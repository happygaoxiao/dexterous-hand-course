# servo_host.py 运行说明

本文档对应脚本：`servo_host.py`

## 1. 环境准备

1. 安装 Python 依赖：

```bash
pip3 install pyserial
```

2. 确认串口号（Windows 示例）：

在设备管理器中查看COM端口。

常见是 COM3、COM4 等。

## 2. 脚本参数说明

```bash
python servo_host.py --port <串口号> [--baud 115200] [--cmd "..."] [--interactive]
```

- `--port`：必填，ZLink 对应串口。
- `--baud`：可选，默认 `115200`。
- `--cmd`：可选，执行一条命令后退出。
- `--interactive`：可选，进入交互模式。

## 3. 直接运行（不加 --cmd / --interactive）

```bash
python servo_host.py --port COM3
```

会自动执行内置演示：
- 依次控制 `0~3` 号舵机移动
- 然后回读 `0~3` 号舵机位置

## 4. 单条命令模式（--cmd）

### 4.1 移动舵机

```bash
python servo_host.py --port COM3 --cmd "MOVE 0 1500 800"
```

含义：
- `0`：舵机 ID
- `1500`：目标位置（范围 `500~2500`）
- `800`：动作时间 ms（范围 `0~9999`）

### 4.2 回读位置

```bash
python servo_host.py --port COM3 --cmd "READ 0"
```

若回包正常，脚本会打印：

```text
POS id=0 pos=1500
```

### 4.3 修改舵机 ID

```bash
python servo_host.py --port COM3 --cmd "SETID 1 2"
```

含义：把 ID `1` 改成 `2`。

### 4.4 原始帧透传

```bash
python servo_host.py --port COM3 --cmd "RAW #000PRAD!"
```

用于手工测试协议帧。

## 5. 交互模式（--interactive）

```bash
python servo_host.py --port COM3 --interactive
```

进入后可输入：

- `MOVE id pos time`
- `READ id`
- `SETID old_id new_id`
- `RAW #000PRAD!`
- `MOVE4 p0 p1 p2 p3 time`
- `quit`

示例：

```text
> MOVE 0 1600 500
> READ 0
> quit
```

## 6. 常见问题排查

1. 提示 `could not open port`：
- 串口号不对，重新在设备管理器中确认。
- 串口被其他软件占用（串口助手/IDE），先关闭占用程序。

2. `(no response)`：
- 检查 ZLink 是否在 USB->Bus 直通模式。
- 检查舵机供电和总线接线。
- 确认舵机 ID 是否正确。
- 若设备实际波特率不是 `115200`，加 `--baud` 指定正确值。

3. `ERR pos range 500..2500`：
- `MOVE` 的位置参数超出脚本设定范围。

4. `ERR id range ...`：
- `READ/SETID` 允许 `0..254`。
- `MOVE` 允许 `0..255`，但 `255` 常用于广播，不适合回读。

## 7. 推荐测试顺序

1. 先单条回读确认连通：

```bash
python servo_host.py --port COM3 --cmd "READ 0"
```

2. 再发一条小动作：

```bash
python servo_host.py --port COM3 --cmd "MOVE 0 1550 400"
```

3. 最后进入交互模式连续调试。

