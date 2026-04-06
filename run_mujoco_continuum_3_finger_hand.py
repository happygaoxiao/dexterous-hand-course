from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "descriptions" / "continuum_soft_3_finger_hand.xml"


@dataclass(frozen=True)
class FingerMeta:
    name: str
    mount_body: int
    root_body: int
    segment_bodies: list[int]
    close_center: np.ndarray
    rest_positions: np.ndarray


def control_alpha(t: float) -> float:
    open_hold = 1.0
    close_time = 2.0
    close_hold = 1.5
    release_time = 2.0

    period = open_hold + close_time + close_hold + release_time
    phase = t % period

    if phase < open_hold:
        return 0.0
    if phase < open_hold + close_time:
        return (phase - open_hold) / close_time
    if phase < open_hold + close_time + close_hold:
        return 1.0

    release_phase = phase - open_hold - close_time - close_hold
    return 1.0 - release_phase / release_time


def get_body_id(model: mujoco.MjModel, name: str) -> int:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise ValueError(f"Body '{name}' not found in model")
    return body_id


def find_segment_bodies(model: mujoco.MjModel, prefix: str) -> tuple[int, list[int]]:
    root_name = f"{prefix}_B_first"
    root_body = get_body_id(model, root_name)
    segment_bodies: list[int] = []

    index = 1
    while True:
        body_name = f"{prefix}_B_{index}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            break
        segment_bodies.append(body_id)
        index += 1

    last_name = f"{prefix}_B_last"
    last_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, last_name)
    if last_body >= 0:
        segment_bodies.append(last_body)

    if not segment_bodies:
        raise ValueError(f"No movable cable bodies found for prefix '{prefix}'")

    return root_body, segment_bodies


def build_finger_metas(model: mujoco.MjModel, data: mujoco.MjData) -> list[FingerMeta]:
    palm_center = np.array([0.0, 0.0, 0.110], dtype=np.float64)
    definitions = [
        ("thumb", "thumb", "thumb_mount"),
        ("index", "index", "index_mount"),
        ("middle", "middle", "middle_mount"),
    ]

    metas: list[FingerMeta] = []
    for name, prefix, mount_name in definitions:
        root_body, segment_bodies = find_segment_bodies(model, prefix)
        metas.append(
            FingerMeta(
                name=name,
                mount_body=get_body_id(model, mount_name),
                root_body=root_body,
                segment_bodies=segment_bodies,
                close_center=palm_center.copy(),
                rest_positions=np.asarray([data.xpos[body_id].copy() for body_id in segment_bodies], dtype=np.float64),
            )
        )
    return metas


def unit(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1.0e-9:
        return np.zeros(3, dtype=np.float64)
    return vec / norm


def apply_continuum_close_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metas: list[FingerMeta],
    alpha: float,
):
    data.xfrc_applied[:, :] = 0.0
    open_gain = 45.0
    bend_force = 3.20
    contract_force = 1.20
    tip_exponent = 1.65

    for meta in metas:
        root_pos = data.xpos[meta.root_body].copy()
        count = len(meta.segment_bodies)

        for index, body_id in enumerate(meta.segment_bodies, start=1):
            body_pos = data.xpos[body_id].copy()
            tip_weight = (index / count) ** tip_exponent
            open_weight = 0.35 + 0.65 * tip_weight
            inward = unit(meta.close_center - body_pos)
            rootward = unit(root_pos - body_pos)
            restore_scale = 1.0 - 0.75 * alpha
            restore = restore_scale * open_gain * open_weight * (meta.rest_positions[index - 1] - body_pos)
            close_force = alpha * (
                bend_force * tip_weight * inward
                + contract_force * tip_weight * rootward
            )
            data.xfrc_applied[body_id, :3] += restore + close_force


def average_tip_distance(data: mujoco.MjData, metas: list[FingerMeta]) -> float:
    tips = np.asarray([data.xpos[meta.segment_bodies[-1]] for meta in metas], dtype=np.float64)
    center = np.mean(np.asarray([meta.close_center for meta in metas], dtype=np.float64), axis=0)
    distances = np.linalg.norm(tips - center, axis=1)
    return float(np.mean(distances))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a MuJoCo continuum 3-finger hand demo.")
    parser.add_argument(
        "--xml",
        type=Path,
        default=XML_PATH,
        help="Path to the MuJoCo XML model.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=8.0,
        help="Simulation duration in seconds.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def run_headless(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metas: list[FingerMeta],
    duration: float,
):
    sim_time = 0.0
    step = 0
    last_print = -1.0

    while sim_time < duration:
        alpha = control_alpha(sim_time)
        apply_continuum_close_forces(model, data, metas, alpha)
        mujoco.mj_step(model, data)

        if sim_time - last_print >= 0.5:
            tip_distance = average_tip_distance(data, metas)
            ball_pos = data.xpos[get_body_id(model, "target_ball")]
            print(
                f"t={sim_time:5.2f}s | alpha={alpha:4.2f} | tip_distance={tip_distance: .3f} | "
                f"ball=({ball_pos[0]: .3f}, {ball_pos[1]: .3f}, {ball_pos[2]: .3f})"
            )
            last_print = sim_time

        sim_time += model.opt.timestep
        step += 1

    print(f"Finished {step} steps in headless mode.")


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    metas: list[FingerMeta],
    duration: float,
):
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -24
        viewer.cam.distance = 0.42
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.11])

        t0 = time.time()
        last_print = -1.0

        while viewer.is_running():
            sim_time = time.time() - t0
            if sim_time > duration:
                break

            alpha = control_alpha(sim_time)
            apply_continuum_close_forces(model, data, metas, alpha)
            mujoco.mj_step(model, data)
            viewer.sync()

            if sim_time - last_print >= 0.5:
                tip_distance = average_tip_distance(data, metas)
                print(f"t={sim_time:5.2f}s | alpha={alpha:4.2f} | tip_distance={tip_distance: .3f}")
                last_print = sim_time

            time.sleep(model.opt.timestep)


def main():
    args = parse_args()
    xml_path = args.xml.resolve()
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    metas = build_finger_metas(model, data)

    print(f"Loaded XML: {xml_path}")
    print(f"nq={model.nq}, nv={model.nv}, nbody={model.nbody}")
    print("Continuum fingers:", ", ".join(meta.name for meta in metas))

    if args.headless:
        run_headless(model, data, metas, args.seconds)
        return

    run_viewer(model, data, metas, args.seconds)


if __name__ == "__main__":
    main()
