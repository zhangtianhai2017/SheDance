#!/usr/bin/env python3
"""Kinematically replay a MuscleMimic retargeted trajectory cache on MyoFullBody and
record an mp4 (no policy / no physics -- just sets qpos per frame + forward kinematics).

Used to eyeball whether a retargeted motion mapped correctly onto the muscle skeleton
before committing GPU hours to training.

    uv run python render_ref.py \
        --cache ~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/<name>.npz \
        --out /mnt/c/.../renders/<name>_ref.mp4
"""
import argparse
import os
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from musclemimic.environments.humanoids import MyoFullBody


def get_model(env) -> "mujoco.MjModel":
    for a in ("_model", "model", "_mjmodel", "mjmodel"):
        m = getattr(env, a, None)
        if isinstance(m, mujoco.MjModel):
            return m
    for a in dir(env):
        try:
            m = getattr(env, a)
        except Exception:
            continue
        if isinstance(m, mujoco.MjModel):
            return m
    raise RuntimeError("Could not find MjModel on env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--disable-fingers", action="store_true", default=True)
    args = ap.parse_args()

    c = np.load(args.cache, allow_pickle=True)
    qpos = np.asarray(c["qpos"])
    freq = float(c["frequency"])

    env = MyoFullBody(disable_fingers=args.disable_fingers)
    model = get_model(env)
    assert model.nq == qpos.shape[1], f"nq {model.nq} != qpos width {qpos.shape[1]}"

    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.distance = 3.2
    cam.elevation = -12
    cam.azimuth = 135

    step = max(1, round(freq / args.fps))
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{args.width}x{args.height}", "-r", str(args.fps), "-i", "-",
         "-pix_fmt", "yuv420p", args.out],
        stdin=subprocess.PIPE,
    )
    n = 0
    for i in range(0, len(qpos), step):
        data.qpos[:] = qpos[i]
        mujoco.mj_forward(model, data)
        cam.lookat[:] = qpos[i][:3]
        renderer.update_scene(data, camera=cam)
        proc.stdin.write(renderer.render().tobytes())
        n += 1
    proc.stdin.close()
    proc.wait()
    print(f"wrote {args.out}  frames={n}  src_freq={freq}Hz  playback_fps={args.fps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
