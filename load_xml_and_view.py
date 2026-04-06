import mujoco
import mujoco.viewer
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "descriptions" / "fully_actuated_dexterous_3_finger_hand.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.azimuth = 135
    viewer.cam.elevation = -28
    viewer.cam.distance = 0.42

    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
