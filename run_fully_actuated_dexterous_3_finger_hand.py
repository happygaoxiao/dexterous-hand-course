from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "descriptions" / "fully_actuated_dexterous_3_finger_hand.xml"

ACTUATOR_NAMES = [
    "thumb_a1",
    "thumb_a2",
    "thumb_a3",
    "index_a1",
    "index_a2",
    "index_a3",
    "middle_a1",
    "middle_a2",
    "middle_a3",
]

JOINT_NAMES = [
    "thumb_j1",
    "thumb_j2",
    "thumb_j3",
    "index_j1",
    "index_j2",
    "index_j3",
    "middle_j1",
    "middle_j2",
    "middle_j3",
]

TIP_SITE_NAMES = ["thumb_tip", "index_tip", "middle_tip"]

OPEN_POSE = np.array([0.00, 0.05, 0.04, 0.00, 0.06, 0.04, 0.00, 0.06, 0.04], dtype=np.float64)
POWER_GRASP_POSE = np.array([0.72, 0.98, 0.78, 0.82, 1.02, 0.88, 0.72, 0.98, 0.84], dtype=np.float64)
PINCH_POSE = np.array([0.92, 1.08, 0.74, 0.62, 1.10, 0.94, 0.10, 0.54, 0.34], dtype=np.float64)


def smoothstep(alpha: float) -> float:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return alpha * alpha * (3.0 - 2.0 * alpha)


def blend_pose(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    return start + smoothstep(alpha) * (end - start)


def pulse(local_t: float, start: float, duration: float) -> float:
    phase = (local_t - start) / duration
    if phase <= 0.0 or phase >= 1.0:
        return 0.0
    return float(np.sin(np.pi * phase))


def wave_pose(local_t: float) -> np.ndarray:
    pose = PINCH_POSE.copy()
    flex_delta = np.array([0.08, 0.16, 0.12], dtype=np.float64)

    pose[3:6] += pulse(local_t, start=0.00, duration=0.80) * flex_delta
    pose[6:9] += pulse(local_t, start=0.55, duration=0.80) * flex_delta
    pose[0:3] += pulse(local_t, start=1.10, duration=0.80) * flex_delta
    return pose


def desired_pose(sim_t: float) -> tuple[str, np.ndarray]:
    period = 11.0
    phase = sim_t % period

    if phase < 1.0:
        return "open", OPEN_POSE
    if phase < 3.5:
        alpha = (phase - 1.0) / 2.5
        return "closing", blend_pose(OPEN_POSE, POWER_GRASP_POSE, alpha)
    if phase < 5.0:
        return "power_grasp", POWER_GRASP_POSE
    if phase < 6.5:
        alpha = (phase - 5.0) / 1.5
        return "pinch_transition", blend_pose(POWER_GRASP_POSE, PINCH_POSE, alpha)
    if phase < 8.5:
        return "finger_wave", wave_pose(phase - 6.5)
    if phase < 10.5:
        alpha = (phase - 8.5) / 2.0
        return "opening", blend_pose(PINCH_POSE, OPEN_POSE, alpha)
    return "reset", OPEN_POSE


def get_ids(model: mujoco.MjModel, names: list[str], obj_type: mujoco.mjtObj) -> np.ndarray:
    ids = []
    for name in names:
        obj_id = mujoco.mj_name2id(model, obj_type, name)
        if obj_id < 0:
            raise ValueError(f"Could not find {obj_type} named '{name}' in model")
        ids.append(obj_id)
    return np.asarray(ids, dtype=np.int32)


def joint_qpos_indices(model: mujoco.MjModel, joint_ids: np.ndarray) -> np.ndarray:
    return np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)


def clamp_ctrl(model: mujoco.MjModel, actuator_ids: np.ndarray, ctrl: np.ndarray) -> np.ndarray:
    ctrlrange = model.actuator_ctrlrange[actuator_ids]
    return np.clip(ctrl, ctrlrange[:, 0], ctrlrange[:, 1])


def average_tip_center(data: mujoco.MjData, tip_site_ids: np.ndarray) -> np.ndarray:
    return np.mean(data.site_xpos[tip_site_ids], axis=0)


def print_status(
    data: mujoco.MjData,
    joint_qpos_ids: np.ndarray,
    tip_site_ids: np.ndarray,
    ball_body_id: int,
    pose_name: str,
) -> None:
    joint_pos = data.qpos[joint_qpos_ids]
    tip_center = average_tip_center(data, tip_site_ids)
    ball_pos = data.xpos[ball_body_id]

    print(
        f"t={data.time:5.2f}s | pose={pose_name:16s} | "
        f"thumb=({joint_pos[0]: .2f}, {joint_pos[1]: .2f}, {joint_pos[2]: .2f}) | "
        f"index=({joint_pos[3]: .2f}, {joint_pos[4]: .2f}, {joint_pos[5]: .2f}) | "
        f"middle=({joint_pos[6]: .2f}, {joint_pos[7]: .2f}, {joint_pos[8]: .2f}) | "
        f"tip_center=({tip_center[0]: .3f}, {tip_center[1]: .3f}, {tip_center[2]: .3f}) | "
        f"ball=({ball_pos[0]: .3f}, {ball_pos[1]: .3f}, {ball_pos[2]: .3f})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a fully actuated 3-finger dexterous hand in MuJoCo.")
    parser.add_argument("--xml", type=Path, default=XML_PATH, help="Path to the MuJoCo XML model.")
    return parser.parse_args()


def run_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    actuator_ids: np.ndarray,
    joint_qpos_ids: np.ndarray,
    tip_site_ids: np.ndarray,
    ball_body_id: int,
) -> None:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -28
        viewer.cam.distance = 0.42
        viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.09], dtype=np.float64)

        wall_t0 = time.time()
        last_print = -1.0

        while viewer.is_running():
            sim_t = time.time() - wall_t0

            pose_name, ctrl = desired_pose(sim_t)
            data.ctrl[actuator_ids] = clamp_ctrl(model, actuator_ids, ctrl)
            mujoco.mj_step(model, data)
            viewer.sync()

            if sim_t - last_print >= 0.5:
                print_status(data, joint_qpos_ids, tip_site_ids, ball_body_id, pose_name)
                last_print = sim_t

            time.sleep(model.opt.timestep)


def main() -> None:
    args = parse_args()
    xml_path = args.xml.resolve()

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    actuator_ids = get_ids(model, ACTUATOR_NAMES, mujoco.mjtObj.mjOBJ_ACTUATOR)
    joint_ids = get_ids(model, JOINT_NAMES, mujoco.mjtObj.mjOBJ_JOINT)
    joint_qpos_ids = joint_qpos_indices(model, joint_ids)
    tip_site_ids = get_ids(model, TIP_SITE_NAMES, mujoco.mjtObj.mjOBJ_SITE)
    ball_body_id = int(get_ids(model, ["target_ball"], mujoco.mjtObj.mjOBJ_BODY)[0])

    data.qpos[joint_qpos_ids] = OPEN_POSE
    data.qvel[:] = 0.0
    data.ctrl[actuator_ids] = OPEN_POSE
    mujoco.mj_forward(model, data)

    print(f"Loaded XML: {xml_path}")
    print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
    print("Actuators:", ", ".join(ACTUATOR_NAMES))
    run_viewer(model, data, actuator_ids, joint_qpos_ids, tip_site_ids, ball_body_id)


if __name__ == "__main__":
    main()
