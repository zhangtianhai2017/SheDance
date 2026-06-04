#!/usr/bin/env python3
"""Definitively check whether MuJoCo's EGL render context is on the NVIDIA GPU or software."""
import os
os.environ["MUJOCO_GL"] = "egl"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import mujoco
ctx = mujoco.GLContext(128, 128)
ctx.make_current()
import ctypes
gl = ctypes.CDLL("libGL.so.1")
gl.glGetString.restype = ctypes.c_char_p
for name, code in (("VENDOR", 0x1F00), ("RENDERER", 0x1F01), ("VERSION", 0x1F02)):
    print(name, "=", gl.glGetString(code))
