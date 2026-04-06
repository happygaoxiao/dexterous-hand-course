# 机器人灵巧手课程仓库

这个仓库用于本科生《机器人灵巧手》课程的教学与实验，内容以"能跑起来、能看清楚、便于修改"为目标，包含课程 slides、MuJoCo 仿真示例，以及 Newton 软体夹爪仿真代码。

仓库中的例子覆盖了几类常见研究对象：

- 欠驱动三指手
- 全驱动三指灵巧手
- 连续体/软体手指
- 基于 Newton 的软体夹爪场景

## 仓库内容

- `slides/`
  课程讲义与配套 PDF。
- `descriptions/`
  模型与场景描述文件，包括 MuJoCo XML 和 Newton JSON。
- `run_underactuated_soft_3_finger_hand.py`
  MuJoCo 欠驱动三指手演示脚本。
- `run_fully_actuated_dexterous_3_finger_hand.py`
  MuJoCo 全驱动三指灵巧手演示脚本。
- `load_xml_and_view.py`
  MuJoCo 交互式查看脚本，可在 viewer 右侧 `control` 面板中手动拖动执行器，实时调节关节角度。
- `run_mujoco_continuum_3_finger_hand.py`
  MuJoCo 连续体三指手演示脚本。
- `run_newton_continuum_soft_gripper.py`
  Newton 软体夹爪仿真脚本。

## 使用示例
在 `conda` 的 `newton` 环境下运行

 - `python -m mujoco.viewer --mjcf=descriptions/fully_actuated_dexterous_3_finger_hand.xml` 命令行查看机器人模型
 -  `python load_xml_and_view.py`，可以通过 Python 代码加载模型并打开 viewer，支持在 viewer 右侧 `control` 面板中手动拖动 9 个 actuator，实时调节三指灵巧手关节角度
 

## 环境建议

安装教程见slides

## 扩展资源

- MuJoCo XML Reference: https://mujoco.readthedocs.io/en/stable/overview.html 建议学生在编写 MuJoCo XML 模型时查阅官方文档，重点关注 `joint`、`geom`、`body`、`tendon`、`actuator`、`sensor` 等部分

- MuJoCo Menagerie: https://github.com/google-deepmind/mujoco_menagerie 建议学生通过这个仓库查阅不同类型机器人的模型、关节配置、执行器写法和场景组织方式。如果想参考更多机械手、机械臂、移动机器人或四足机器人模型，这个资源很有帮助




## 课程使用建议

- 先阅读 `slides/` 中的课程讲义，了解灵巧手分类、结构设计与抓取规划基础
- 再运行 MuJoCo 示例，观察欠驱动与全驱动手的差别
- 最后尝试连续体与软体手模型，理解刚体手与软体手在建模与控制上的区别（Newton环境需要GPU）

## 说明

这个仓库偏向课程教学、演示和实验入门，希望学生能设计出自己的灵巧手，并进行多种物体抓取实验
