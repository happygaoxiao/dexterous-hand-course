from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

XML_PATH = Path(__file__).resolve().parent / "descriptions" / "underactuated_soft_3_finger_hand.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# 便于按名字访问
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "close_hand")
tendon_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "grasp_length")
force_sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "close_force")

print(f"Loaded XML: {XML_PATH}")
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
print("Actuator ctrlrange:", model.actuator_ctrlrange[act_id])

# 先前向计算一次
mujoco.mj_forward(model, data)

# 运动模式：
# 0~2s 打开
# 2~5s 缓慢闭合
# 5~7s 保持
# 7~10s 缓慢张开
# 然后循环

def desired_ctrl(t: float, theta = 10) -> float:
    period = 10.0
    x = t % period

    if x < 2.0:
        return 0.0
    elif x < 5.0:
        return (x - 2.0) / 3.0 * theta
    elif x < 7.0:
        return theta
    else:
        return theta * (1.0 - (x - 7.0) / 3.0)


with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -35
    viewer.cam.distance = 0.45
    viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.04])

    t0 = time.time()
    last_print = -1.0

    while viewer.is_running():
        sim_t = time.time() - t0
        cmd = desired_ctrl(sim_t)
        
        data.ctrl[act_id] = cmd

        mujoco.mj_step(model, data)
        viewer.sync()
        if sim_t - last_print > 0.5:
            tendon_length = data.sensordata[tendon_sensor_id]
            actuator_force = data.sensordata[force_sensor_id]
            print(
                f"t={sim_t:6.2f}s | ctrl={data.ctrl[act_id]:5.2f} | "
                f"tendon={tendon_length:5.2f} | force={actuator_force:7.3f}"
            )
            last_print = sim_t

        time.sleep(model.opt.timestep)
