#!/usr/bin/env python3
"""Reusable render core for muscle-driven playback (vectorized).
RenderCtx loads model+qpos cache+activation sto + precomputes index arrays ONCE;
render_frame(fi) returns an HxWx3 uint8 image. Used by render_one (single proc) and
render_parallel (frame-chunked). Vectorized act-assignment + tendon coloring (no Python
per-muscle loop). Frames are pushed to ffmpeg as raw bytes by the drivers (no PNG/disk)."""
import os, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
# osmesa (CPU software) is ~27x faster than egl here: WSL's GPU path (Mesa->D3D12->WSLg)
# has a ~4s/call fixed submit-overhead, independent of resolution. CPU softrender wins.
os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.environ.get("RENDER_CACHE") or os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
W, H = 720, 960


def gaussian_smooth(x, sigma=2.5, radius=6):
    k = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2); k /= k.sum()
    xp = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1)


def read_sto(path):
    lines = open(path).read().splitlines()
    hi = next(i for i, l in enumerate(lines) if l.strip().lower() == "endheader")
    hdr = lines[hi + 1].split("\t")
    data = np.array([l.split("\t") for l in lines[hi + 2:] if l.strip()], float)
    return hdr, data


class RenderCtx:
    def __init__(self, sto_path, color_gain=1.0):
        c = np.load(CACHE, allow_pickle=True)
        self.qpos = gaussian_smooth(np.asarray(c["qpos"], float))
        self.freq = float(c["frequency"])
        env = MyoFullBody(disable_fingers=True)
        self.m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel"))
                      if isinstance(o, mujoco.MjModel))
        m = self.m
        self.d = mujoco.MjData(m)
        self.gain = float(color_gain)

        hdr, sto = read_sto(sto_path)
        self.sto = sto
        self.tcol = sto[:, hdr.index("time")]
        oset = set(hdr)
        mjc_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]

        def mjc_to_osim(n):
            if n in oset: return n
            if n.endswith("_left") and n[:-5] + "_l" in oset: return n[:-5] + "_l"
            if not n.endswith(("_left", "_l", "_r")) and n + "_r" in oset: return n + "_r"
            return None

        # --- precompute index arrays ONCE (vectorization) ---
        nact = self.d.act.shape[0]
        ai, col = [], []
        for i, nm in enumerate(mjc_names):
            o = mjc_to_osim(nm)
            if o is not None and i < nact:
                ai.append(i); col.append(hdr.index(o))
        self.act_idx = np.array(ai, int)            # mujoco act indices to fill from sto
        self.col_idx = np.array(col, int)           # matching sto columns
        act2ten = m.actuator_trnid[:, 0].astype(int)
        valid = (act2ten >= 0) & (act2ten < m.ntendon)
        self.all_mu = np.nonzero(valid)[0]          # muscles with a valid tendon (for coloring)
        self.all_ten = act2ten[valid]               # their tendon ids
        print(f"mapped {len(self.act_idx)}/{m.nu} muscles to activations; "
              f"{len(self.all_ten)} tendons colorable; sto frames={len(self.tcol)}", flush=True)

        m.vis.global_.offwidth = W; m.vis.global_.offheight = H
        self.rend = mujoco.Renderer(m, height=H, width=W)
        self.opt = mujoco.MjvOption(); self.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 2.6; self.cam.elevation = -12; self.cam.azimuth = 120
        self.cam.lookat[:] = [0, 0, 0.95]

    def frames_in_range(self, t0=None, t1=None, step=1):
        idx = np.arange(len(self.tcol))
        if t0 is not None:
            idx = idx[(self.tcol >= t0) & (self.tcol <= t1)]
        return idx[::step]

    def render_frame(self, fi):
        m, d = self.m, self.d
        qi = int(round(self.tcol[fi] * self.freq))
        if qi < 0 or qi >= self.qpos.shape[0]:
            return None
        d.qpos[:] = self.qpos[qi]; d.qvel[:] = 0
        d.act[:] = 0.0
        d.act[self.act_idx] = self.sto[fi, self.col_idx]          # vectorized act fill
        mujoco.mj_forward(m, d)
        a = np.clip(d.act[self.all_mu] * self.gain, 0.0, 1.0)     # vectorized coloring
        rgba = np.empty((a.shape[0], 4))
        rgba[:, 0] = 0.2 + 0.8 * a
        rgba[:, 1] = 0.15 + 0.15 * a
        rgba[:, 2] = 0.4 * (1.0 - a)
        rgba[:, 3] = 0.55 + 0.45 * a
        m.tendon_rgba[self.all_ten] = rgba
        self.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.95]
        self.rend.update_scene(d, self.cam, self.opt)
        return self.rend.render()

    def close(self):
        self.rend.close()
