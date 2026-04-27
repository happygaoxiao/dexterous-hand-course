#!/usr/bin/env python3
"""PC direct control for ZLink (USB->Bus mode) and ZP bus servos.

PC(USB COM) -> ZLink -> ZP Bus Servo
"""

import argparse
import re
import sys
import time
from typing import List, Sequence

import serial


RAD_RE = re.compile(r"#(\d{3})P(\d{4})!")
SERVO_MIN_POS = 500
SERVO_MAX_POS = 2500
SERVO_RANGE_DEG = 270.0
DEFAULT_JOINT_IDS = (0, 1, 2, 3)


def angle_deg_to_pos(angle_deg: float) -> int:
    angle_deg = float(angle_deg)
    if angle_deg < 0.0 or angle_deg > SERVO_RANGE_DEG:
        raise ValueError(f"joint angle must be within 0..{SERVO_RANGE_DEG} degrees")

    span = SERVO_MAX_POS - SERVO_MIN_POS
    return int(round(SERVO_MIN_POS + angle_deg * span / SERVO_RANGE_DEG))


def pos_to_angle_deg(pos: int) -> float:
    if pos < SERVO_MIN_POS or pos > SERVO_MAX_POS:
        raise ValueError(f"servo position must be within {SERVO_MIN_POS}..{SERVO_MAX_POS}")

    span = SERVO_MAX_POS - SERVO_MIN_POS
    return (pos - SERVO_MIN_POS) * SERVO_RANGE_DEG / span


class ZlinkHost:
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.15) -> None:
        self.ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        time.sleep(0.2)
        self.drain()

    def close(self) -> None:
        if self.ser.is_open:
            self.ser.close()

    def drain(self) -> None:
        time.sleep(0.03)
        while self.ser.in_waiting:
            self.ser.read(self.ser.in_waiting)

    def send_raw(self, frame: str) -> None:
        self.ser.write(frame.encode("ascii"))

    def recv_raw(self, wait_s: float = 0.2) -> List[str]:
        end_time = time.time() + wait_s
        chunks: List[str] = []
        while time.time() < end_time:
            data = self.ser.read(self.ser.in_waiting or 1)
            if not data:
                continue
            chunks.append(data.decode("ascii", errors="replace"))
        if not chunks:
            return []
        text = "".join(chunks)
        parts = [p for p in text.replace("\r", "\n").split("\n") if p]
        return parts if parts else [text]

    def query_raw(self, frame: str, wait_s: float = 0.2) -> List[str]:
        self.drain()
        self.send_raw(frame)
        return self.recv_raw(wait_s)

    def read_position(self, servo_id: int, wait_s: float = 0.25) -> int:
        if servo_id < 0 or servo_id > 254:
            raise ValueError("servo id must be within 0..254")

        lines = self.query_raw(make_read_frame(servo_id), wait_s=wait_s)
        for line in lines:
            match = RAD_RE.search(line)
            if match and int(match.group(1)) == servo_id:
                return int(match.group(2))

        raise RuntimeError(f"failed to read servo {servo_id}, response={lines!r}")

    def move_to_joints(self, joint_angles: Sequence[float], t: float) -> None:
        import numpy as np

        angles = np.asarray(joint_angles, dtype=float)
        if angles.shape != (len(DEFAULT_JOINT_IDS),):
            raise ValueError(f"joint_angles must have shape ({len(DEFAULT_JOINT_IDS)},)")

        time_ms = int(round(float(t) * 1000.0))
        if time_ms < 0 or time_ms > 9999:
            raise ValueError("move time must convert to an integer number of milliseconds in 0..9999")

        positions = [angle_deg_to_pos(angle) for angle in angles.tolist()]
        frame = "".join(
            make_move_frame(servo_id, pos, time_ms)
            for servo_id, pos in zip(DEFAULT_JOINT_IDS, positions)
        )
        self.query_raw(frame, wait_s=0.1)

    def read_joints(self):
        import numpy as np

        angles = [pos_to_angle_deg(self.read_position(servo_id)) for servo_id in DEFAULT_JOINT_IDS]
        return np.asarray(angles, dtype=float)



def make_move_frame(servo_id: int, pos: int, time_ms: int) -> str:
    return f"#{servo_id:03d}P{pos:04d}T{time_ms:04d}!"



def make_read_frame(servo_id: int) -> str:
    return f"#{servo_id:03d}PRAD!"



def make_setid_frame(old_id: int, new_id: int) -> str:
    return f"#{old_id:03d}PID{new_id:03d}!"



def print_lines(lines: List[str]) -> None:
    if not lines:
        print("(no response)")
        return
    for line in lines:
        print(line)



def parse_int(s: str) -> int:
    return int(s.strip())



def handle_command(host: ZlinkHost, cmd: str) -> None:
    parts = cmd.strip().split()
    if not parts:
        return

    head = parts[0].upper()

    if head == "PING":
        print("ZLink direct mode has no PING protocol; checking by opening COM succeeded.")
        return

    if head == "RAW":
        if len(parts) < 2:
            print("Usage: RAW #000PRAD!")
            return
        frame = cmd.strip()[4:]
        print(f"TX {frame}")
        print_lines(host.query_raw(frame))
        return

    if head == "MOVE":
        if len(parts) != 4:
            print("Usage: MOVE id pos time")
            return
        servo_id = parse_int(parts[1])
        pos = parse_int(parts[2])
        time_ms = parse_int(parts[3])
        if servo_id < 0 or servo_id > 255:
            print("ERR id range 0..255")
            return
        if pos < SERVO_MIN_POS or pos > SERVO_MAX_POS:
            print(f"ERR pos range {SERVO_MIN_POS}..{SERVO_MAX_POS}")
            return
        if time_ms < 0 or time_ms > 9999:
            print("ERR time range 0..9999")
            return
        frame = make_move_frame(servo_id, pos, time_ms)
        print(f"TX {frame}")
        lines = host.query_raw(frame, wait_s=0.1)
        print_lines(lines)
        return

    if head == "READ":
        if len(parts) != 2:
            print("Usage: READ id")
            return
        servo_id = parse_int(parts[1])
        if servo_id < 0 or servo_id > 254:
            print("ERR id range 0..254")
            return
        frame = make_read_frame(servo_id)
        print(f"TX {frame}")
        lines = host.query_raw(frame, wait_s=0.25)
        if not lines:
            print("(no response)")
            return
        for line in lines:
            m = RAD_RE.search(line)
            if m:
                print(f"POS id={int(m.group(1))} pos={int(m.group(2))}")
            else:
                print(line)
        return

    if head == "SETID":
        if len(parts) != 3:
            print("Usage: SETID old_id new_id")
            return
        old_id = parse_int(parts[1])
        new_id = parse_int(parts[2])
        if old_id < 0 or old_id > 254 or new_id < 0 or new_id > 254:
            print("ERR id range 0..254")
            return
        frame = make_setid_frame(old_id, new_id)
        print(f"TX {frame}")
        print_lines(host.query_raw(frame, wait_s=0.25))
        return

    if head == "MOVE4":
        if len(parts) != 6:
            print("Usage: MOVE4 p0 p1 p2 p3 time")
            return
        p0, p1, p2, p3, t = map(parse_int, parts[1:])
        for sid, pos in enumerate([p0, p1, p2, p3]):
            handle_command(host, f"MOVE {sid} {pos} {t}")
        return

    print("Unknown command")



def interactive(host: ZlinkHost) -> None:
    print("Interactive mode (PC -> ZLink -> ZP)")
    print("Commands:")
    print("  MOVE id pos time")
    print("  READ id")
    print("  SETID old_id new_id")
    print("  RAW #000PRAD!")
    print("  MOVE4 p0 p1 p2 p3 time")
    print("  quit")

    while True:
        try:
            cmd = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not cmd:
            continue
        if cmd.lower() in {"q", "quit", "exit"}:
            break

        handle_command(host, cmd)



def main() -> int:
    parser = argparse.ArgumentParser(description="PC direct ZLink host for ZP bus servo")
    parser.add_argument("--port", required=True, help="COM port of ZLink, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--cmd", help="send one command then exit")
    parser.add_argument("--interactive", action="store_true")
    args = parser.parse_args()

    host = ZlinkHost(port=args.port, baudrate=args.baud)
    try:
        if args.cmd:
            handle_command(host, args.cmd)
            return 0

        if args.interactive:
            interactive(host)
            return 0

        demo = [
            "MOVE 0 1500 800",
            "MOVE 1 1300 800",
            "MOVE 2 1700 800",
            "MOVE 3 1500 800",
            "READ 0",
            "READ 1",
            "READ 2",
            "READ 3",
        ]
        for cmd in demo:
            print(">", cmd)
            handle_command(host, cmd)
        return 0
    finally:
        host.close()


if __name__ == "__main__":
    sys.exit(main())
