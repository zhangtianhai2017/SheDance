#!/usr/bin/env python3
"""Ping-pong the 12s qpos cache into a 60s qpos cache (matches how pop60.mot was built),
so the parallel render can actually render 60s instead of extrapolating. -> pop60_qpos.npz"""
import os, numpy as np
SRC = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
OUT = os.path.expanduser("~/shedance/osim/pop60_qpos.npz")
TARGET = 6000
c = np.load(SRC, allow_pickle=True)
q = np.asarray(c["qpos"], float); freq = float(c["frequency"])
cycle = np.vstack([q, q[-2:0:-1]])               # forward + reversed (smooth turn)
reps = int(np.ceil(TARGET / len(cycle)))
allq = np.vstack([cycle] * reps)[:TARGET]
np.savez(OUT, qpos=allq, frequency=freq)
print(f"wrote {OUT}: {allq.shape[0]} frames = {allq.shape[0]/freq:.1f}s, qpos dim={allq.shape[1]}")
