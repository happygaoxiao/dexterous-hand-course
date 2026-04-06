from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import newton
import newton.examples as newton_examples
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parent
DEFAULT_SCENE = ROOT / "descriptions" / "newton_continuum_soft_gripper_scene.json"
ACTIVE_PARTICLE_MASK = int(newton.ParticleFlags.ACTIVE)


@dataclass(frozen=True)
class FingerMeta:
    label: str
    particle_start: int
    particle_end: int
    origin: np.ndarray
    basis: np.ndarray
    inward_dir: np.ndarray
    root_dir: np.ndarray


def load_scene(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def vec3(values) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def quat_from_basis(ex: np.ndarray, ey: np.ndarray, ez: np.ndarray):
    basis = wp.matrix_from_cols(
        wp.vec3(*map(float, ex)),
        wp.vec3(*map(float, ey)),
        wp.vec3(*map(float, ez)),
    )
    return wp.quat_from_matrix(basis)


def control_profile(t: float, cfg: dict) -> tuple[float, float]:
    open_hold = float(cfg["open_hold"])
    close_time = float(cfg["close_time"])
    close_hold = float(cfg["close_hold"])
    lift_time = float(cfg.get("lift_time", 0.0))
    release_time = float(cfg["release_time"])

    period = open_hold + close_time + close_hold + lift_time + release_time
    phase = t % period

    if phase < open_hold:
        return 0.0, 0.0
    if phase < open_hold + close_time:
        return (phase - open_hold) / close_time, 0.0
    if phase < open_hold + close_time + close_hold:
        return 1.0, 0.0
    if phase < open_hold + close_time + close_hold + lift_time:
        lift_phase = phase - open_hold - close_time - close_hold
        lift_alpha = lift_phase / max(lift_time, 1.0e-6)
        return 1.0, lift_alpha

    release_phase = phase - open_hold - close_time - close_hold - lift_time
    release_alpha = 1.0 - release_phase / release_time
    release_alpha = np.clip(release_alpha, 0.0, 1.0)
    return float(release_alpha), float(release_alpha)


def add_static_environment(builder: newton.ModelBuilder, cfg: dict):
    env = cfg["environment"]
    contact = cfg["contact"]

    builder.add_ground_plane(
        cfg=builder.ShapeConfig(
            ke=float(contact["shape_ke"]),
            kd=float(contact["shape_kd"]),
            mu=float(env["ground_mu"]),
        ),
        label="ground",
    )

    visual_cfg = builder.ShapeConfig(density=0.0, has_shape_collision=False)
    table_cfg = builder.ShapeConfig(
        density=0.0,
        ke=float(contact["shape_ke"]),
        kd=float(contact["shape_kd"]),
        mu=float(env["ground_mu"]),
    )

    table_half = env["table_half_extents"]
    table_center = env["table_center"]
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*map(float, table_center)), wp.quat_identity()),
        hx=float(table_half[0]),
        hy=float(table_half[1]),
        hz=float(table_half[2]),
        cfg=table_cfg,
        label="table",
    )

    crossbar_half = env["crossbar_half_extents"]
    crossbar_center = env["crossbar_center"]
    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(*map(float, crossbar_center)), wp.quat_identity()),
        hx=float(crossbar_half[0]),
        hy=float(crossbar_half[1]),
        hz=float(crossbar_half[2]),
        cfg=visual_cfg,
        label="crossbar_visual",
    )

    builder.add_shape_cylinder(
        body=-1,
        xform=wp.transform(
            wp.vec3(
                float(crossbar_center[0]),
                float(crossbar_center[1]),
                float(crossbar_center[2]) + 0.04,
            ),
            wp.quat_identity(),
        ),
        radius=0.016,
        half_height=0.024,
        cfg=visual_cfg,
        label="wrist_visual",
    )


def add_ball(builder: newton.ModelBuilder, cfg: dict) -> int:
    obj = cfg["object"]
    contact = cfg["contact"]
    position = vec3(obj["position"])

    body = builder.add_body(
        xform=wp.transform(wp.vec3(*map(float, position)), wp.quat_identity()),
        label="target_ball",
    )
    builder.add_shape_sphere(
        body=body,
        radius=float(obj["radius"]),
        cfg=builder.ShapeConfig(
            density=float(obj["density"]),
            ke=float(contact["shape_ke"]),
            kd=float(contact["shape_kd"]),
            mu=float(contact["shape_mu"]),
        ),
        label="target_ball_geom",
    )
    return body


def add_continuum_finger(builder: newton.ModelBuilder, cfg: dict, side: int) -> FingerMeta:
    finger = cfg["finger"]

    length = float(finger["length"])
    width = float(finger["width"])
    thickness = float(finger["thickness"])
    root_center = np.array(
        [
            float(finger["root_x"]),
            side * float(finger["root_spacing"]) * 0.5,
            float(finger["root_height"]),
        ],
        dtype=np.float64,
    )

    ex = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    ey = np.array([1.0 if side > 0 else -1.0, 0.0, 0.0], dtype=np.float64)
    ez = np.cross(ex, ey)

    origin = root_center - 0.5 * width * ey - 0.5 * thickness * ez
    basis = np.column_stack([ex, ey, ez])
    quat = quat_from_basis(ex, ey, ez)

    start = builder.particle_count
    builder.add_soft_grid(
        pos=wp.vec3(*map(float, origin)),
        rot=quat,
        vel=wp.vec3(0.0, 0.0, 0.0),
        dim_x=int(finger["dim_x"]),
        dim_y=int(finger["dim_y"]),
        dim_z=int(finger["dim_z"]),
        cell_x=length / int(finger["dim_x"]),
        cell_y=width / int(finger["dim_y"]),
        cell_z=thickness / int(finger["dim_z"]),
        density=float(finger["density"]),
        k_mu=float(finger["k_mu"]),
        k_lambda=float(finger["k_lambda"]),
        k_damp=float(finger["k_damp"]),
        fix_left=True,
        tri_ke=float(finger["tri_ke"]),
        tri_ka=float(finger["tri_ka"]),
        tri_kd=float(finger["tri_kd"]),
        edge_ke=float(finger["edge_ke"]),
        edge_kd=float(finger["edge_kd"]),
        particle_radius=float(finger["particle_radius"]),
    )
    end = builder.particle_count

    return FingerMeta(
        label="left_finger" if side > 0 else "right_finger",
        particle_start=start,
        particle_end=end,
        origin=origin,
        basis=basis,
        inward_dir=ez,
        root_dir=-ex,
    )


def build_actuation_fields(state: newton.State, metas: list[FingerMeta], cfg: dict):
    finger = cfg["finger"]
    act = cfg["actuation"]
    rest_positions = state.particle_q.numpy()
    particle_count = rest_positions.shape[0]

    bend_dir = np.zeros((particle_count, 3), dtype=np.float32)
    root_dir = np.zeros((particle_count, 3), dtype=np.float32)
    anchor_dir = np.zeros((particle_count, 3), dtype=np.float32)
    bend_weight = np.zeros(particle_count, dtype=np.float32)
    contract_weight = np.zeros(particle_count, dtype=np.float32)
    drive_weight = np.zeros(particle_count, dtype=np.float32)

    half_width = float(finger["width"]) * 0.5
    tip_exponent = float(act["tip_exponent"])
    inner_power = float(act["inner_power"])
    midline_gain = float(act["midline_gain"])
    length = float(finger["length"])
    thickness = float(finger["thickness"])
    drive_exponent = float(act.get("drive_exponent", 1.0))

    for meta in metas:
        rest_world = rest_positions[meta.particle_start : meta.particle_end]
        local = (rest_world - meta.origin) @ meta.basis

        tip = np.clip(local[:, 0] / max(length, 1.0e-6), 0.0, 1.0) ** tip_exponent
        width_center = 1.0 - np.clip(np.abs(local[:, 1] - half_width) / max(half_width, 1.0e-6), 0.0, 1.0)
        inner = np.clip(local[:, 2] / max(thickness, 1.0e-6), 0.0, 1.0) ** inner_power

        finger_bend = tip * (midline_gain + (1.0 - midline_gain) * width_center)
        finger_contract = tip * inner * (0.4 + 0.6 * width_center)
        finger_drive = np.clip(local[:, 0] / max(length, 1.0e-6), 0.0, 1.0) ** drive_exponent

        bend_dir[meta.particle_start : meta.particle_end] = meta.inward_dir.astype(np.float32)
        root_dir[meta.particle_start : meta.particle_end] = meta.root_dir.astype(np.float32)
        anchor_dir[meta.particle_start : meta.particle_end] = meta.inward_dir.astype(np.float32)
        bend_weight[meta.particle_start : meta.particle_end] = finger_bend.astype(np.float32)
        contract_weight[meta.particle_start : meta.particle_end] = finger_contract.astype(np.float32)
        drive_weight[meta.particle_start : meta.particle_end] = finger_drive.astype(np.float32)

    return anchor_dir, bend_dir, root_dir, bend_weight, contract_weight, drive_weight


def create_viewer(args: argparse.Namespace, num_frames: int, fps: int):
    if args.viewer == "gl":
        if args.headless:
            return newton.viewer.ViewerNull(num_frames=num_frames)
        return newton.viewer.ViewerGL(headless=False)
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=num_frames)
    if args.viewer == "usd":
        return newton.viewer.ViewerUSD(output_path=args.output_path, fps=fps, num_frames=num_frames)
    if args.viewer == "rerun":
        return newton.viewer.ViewerRerun(address=args.rerun_address)
    if args.viewer == "viser":
        return newton.viewer.ViewerViser()
    raise ValueError(f"Unsupported viewer '{args.viewer}'")


def parse_args() -> argparse.Namespace:
    parser = newton_examples.create_parser()
    parser.set_defaults(num_frames=None, broad_phase=None)
    parser.add_argument(
        "--scene",
        type=Path,
        default=DEFAULT_SCENE,
        help="Path to the continuum soft gripper JSON scene description.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=None,
        help="Override the simulated duration in seconds.",
    )
    return parser.parse_args()


class Example:
    def __init__(self, viewer, args: argparse.Namespace, cfg: dict):
        self.viewer = viewer
        self.args = args
        self.cfg = cfg

        self.fps = int(cfg["scene"]["fps"])
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = int(cfg["scene"]["sim_substeps"])
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
        builder.default_shape_cfg.ke = float(cfg["contact"]["shape_ke"])
        builder.default_shape_cfg.kd = float(cfg["contact"]["shape_kd"])
        builder.default_shape_cfg.mu = float(cfg["contact"]["shape_mu"])

        add_static_environment(builder, cfg)
        self.ball_body = add_ball(builder, cfg)
        self.finger_metas = [
            add_continuum_finger(builder, cfg, side=1),
            add_continuum_finger(builder, cfg, side=-1),
        ]

        # Soft grids need a colored mesh topology before finalization for VBD rendering/contact bookkeeping.
        builder.color()

        self.model = builder.finalize(device=wp.get_device())
        self.model.soft_contact_ke = float(cfg["contact"]["soft_contact_ke"])
        self.model.soft_contact_kd = float(cfg["contact"]["soft_contact_kd"])
        self.model.soft_contact_mu = float(cfg["contact"]["soft_contact_mu"])

        self.solver = newton.solvers.SolverVBD(
            model=self.model,
            iterations=int(cfg["scene"]["solver_iterations"]),
            particle_enable_self_contact=True,
            particle_self_contact_radius=float(cfg["finger"]["particle_radius"]) * 0.9,
            particle_self_contact_margin=float(cfg["finger"]["particle_radius"]) * 1.15,
            particle_enable_tile_solve=True,
            rigid_body_particle_contact_buffer_size=1024,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        margin = float(cfg["finger"]["particle_radius"]) * float(cfg["contact"]["soft_contact_margin_scale"])
        broad_phase = args.broad_phase or cfg["scene"]["broad_phase"]
        self.collision_pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase=broad_phase,
            soft_contact_margin=margin,
        )
        self.contacts = self.collision_pipeline.contacts()

        rest_particle_q = self.state_0.particle_q.numpy()
        self.rest_particle_q = wp.array(rest_particle_q, dtype=wp.vec3, device=self.model.device)

        anchor_dir, bend_dir, root_dir, bend_weight, contract_weight, drive_weight = build_actuation_fields(
            self.state_0,
            self.finger_metas,
            cfg,
        )
        self.anchor_dir = wp.array(anchor_dir, dtype=wp.vec3, device=self.model.device)
        self.bend_dir = wp.array(bend_dir, dtype=wp.vec3, device=self.model.device)
        self.root_dir = wp.array(root_dir, dtype=wp.vec3, device=self.model.device)
        self.bend_weight = wp.array(bend_weight, dtype=wp.float32, device=self.model.device)
        self.contract_weight = wp.array(contract_weight, dtype=wp.float32, device=self.model.device)
        self.drive_weight = wp.array(drive_weight, dtype=wp.float32, device=self.model.device)

        self.viewer.set_model(self.model)

        camera = cfg.get("camera")
        if camera and hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(
                pos=wp.vec3(*map(float, camera["position"])),
                pitch=float(camera["pitch"]),
                yaw=float(camera["yaw"]),
            )
            if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
                self.viewer.camera.fov = float(camera["fov"])

        self.last_print = -1.0

    def tip_aperture(self) -> float:
        particle_q = self.state_0.particle_q.numpy()
        tip_means = []
        tip_threshold = 0.85 * float(self.cfg["finger"]["length"])

        for meta in self.finger_metas:
            finger_q = particle_q[meta.particle_start : meta.particle_end]
            local = (finger_q - meta.origin) @ meta.basis
            tip_mask = local[:, 0] >= tip_threshold
            if not np.any(tip_mask):
                tip_mask[np.argmax(local[:, 0])] = True
            tip_points = finger_q[tip_mask]
            tip_means.append(np.mean(tip_points, axis=0))

        return float(abs(tip_means[0][1] - tip_means[1][1]))

    def step(self):
        if not self.viewer.is_paused():
            self.simulate()
            self.sim_time += self.frame_dt

    def simulate(self):
        bend_force = float(self.cfg["actuation"]["bend_force"])
        contract_force = float(self.cfg["actuation"]["contract_force"])
        root_close_distance = float(self.cfg["actuation"].get("root_close_distance", 0.0))
        lift_height = float(self.cfg["actuation"].get("lift_height", 0.0))

        for substep in range(self.sim_substeps):
            local_time = self.sim_time + substep * self.sim_dt
            alpha, lift_alpha = control_profile(local_time, self.cfg["control"])

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            wp.launch(
                drive_root_particles,
                dim=self.model.particle_count,
                inputs=[
                    self.rest_particle_q,
                    self.anchor_dir,
                    self.root_dir,
                    self.drive_weight,
                    wp.float32(alpha),
                    wp.float32(root_close_distance),
                    wp.float32(lift_alpha),
                    wp.float32(lift_height),
                    wp.float32(self.sim_dt),
                ],
                outputs=[self.state_0.particle_q, self.state_0.particle_qd],
                device=self.model.device,
            )

            wp.launch(
                apply_continuum_actuation,
                dim=self.model.particle_count,
                inputs=[
                    self.model.particle_flags,
                    self.bend_dir,
                    self.root_dir,
                    self.bend_weight,
                    self.contract_weight,
                    wp.int32(ACTIVE_PARTICLE_MASK),
                    wp.float32(alpha),
                    wp.float32(bend_force),
                    wp.float32(contract_force),
                ],
                outputs=[self.state_0.particle_f],
                device=self.model.device,
            )

            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.sim_time - self.last_print > 0.5:
            ball_pos = self.state_0.body_q.numpy()[self.ball_body][:3]
            alpha, lift_alpha = control_profile(self.sim_time, self.cfg["control"])
            aperture = self.tip_aperture()
            print(
                f"t={self.sim_time:5.2f}s | alpha={alpha:4.2f} | lift={lift_alpha:4.2f} | "
                f"aperture={aperture: .3f} | ball=({ball_pos[0]: .3f}, {ball_pos[1]: .3f}, {ball_pos[2]: .3f})"
            )
            self.last_print = self.sim_time

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        particle_q = self.state_0.particle_q.numpy()
        particle_qd = self.state_0.particle_qd.numpy()

        if not np.isfinite(particle_q).all():
            raise ValueError("Non-finite particle positions detected")
        if not np.isfinite(particle_qd).all():
            raise ValueError("Non-finite particle velocities detected")

        min_pos = np.min(particle_q, axis=0)
        max_pos = np.max(particle_q, axis=0)
        bbox_size = np.linalg.norm(max_pos - min_pos)
        assert bbox_size < 2.0, f"Soft gripper exploded: bbox_size={bbox_size:.3f}"
        assert min_pos[2] > -0.15, f"Excessive penetration: z_min={min_pos[2]:.4f}"


@wp.kernel
def apply_continuum_actuation(
    particle_flags: wp.array(dtype=wp.int32),
    bend_dir: wp.array(dtype=wp.vec3),
    root_dir: wp.array(dtype=wp.vec3),
    bend_weight: wp.array(dtype=wp.float32),
    contract_weight: wp.array(dtype=wp.float32),
    active_mask: wp.int32,
    alpha: wp.float32,
    bend_force: wp.float32,
    contract_force: wp.float32,
    particle_f: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    if (particle_flags[tid] & active_mask) == 0:
        return

    force = (
        bend_dir[tid] * (alpha * bend_force * bend_weight[tid])
        + root_dir[tid] * (alpha * contract_force * contract_weight[tid])
    )
    particle_f[tid] = particle_f[tid] + force


@wp.kernel
def drive_root_particles(
    rest_particle_q: wp.array(dtype=wp.vec3),
    anchor_dir: wp.array(dtype=wp.vec3),
    root_dir: wp.array(dtype=wp.vec3),
    drive_weight: wp.array(dtype=wp.float32),
    close_alpha: wp.float32,
    close_distance: wp.float32,
    lift_alpha: wp.float32,
    lift_height: wp.float32,
    dt: wp.float32,
    particle_q: wp.array(dtype=wp.vec3),
    particle_qd: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    weight = drive_weight[tid]
    if weight <= 0.0:
        return

    target = (
        rest_particle_q[tid]
        + anchor_dir[tid] * (close_alpha * close_distance * weight)
        + root_dir[tid] * (lift_alpha * lift_height * weight)
    )
    velocity = (target - particle_q[tid]) / wp.max(dt, wp.float32(1.0e-6))

    particle_q[tid] = target
    particle_qd[tid] = velocity


def main():
    args = parse_args()
    scene_path = args.scene.resolve()
    cfg = load_scene(scene_path)

    if args.quiet:
        wp.config.quiet = True

    device = args.device or cfg["scene"]["device"]
    wp.set_device(device)

    fps = int(cfg["scene"]["fps"])
    scene_duration = float(args.seconds if args.seconds is not None else cfg["scene"]["duration"])
    num_frames = args.num_frames if args.num_frames is not None else int(math.ceil(scene_duration * fps))
    viewer = create_viewer(args, num_frames, fps)

    actual_viewer = "null" if args.viewer == "gl" and args.headless else args.viewer
    print(f"Loaded scene: {scene_path}")
    print(f"Device: {device}")
    print(f"Viewer: {actual_viewer}")
    print(f"Duration: {scene_duration:.2f}s")

    example = Example(viewer, args, cfg)
    newton_examples.run(example, args)


if __name__ == "__main__":
    main()
