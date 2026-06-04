#!/usr/bin/env python3
"""Render the muscle-driven Pop: MuJoCo MyoFullBody playing the motion with muscles colored
by the OpenSim-Static-Optimization activations (mjVIS_ACTUATOR: low=blue, high=red).

OpenSim activation.sto -> map muscle names to MuJoCo (side convention: MuJoCo no-suffix=right,
'_left'=left; trunk/leg already share _r/_l) -> set d.act per frame -> render.

Args: [activation_sto out_mp4 t0 t1]   defaults: active-segment sto, t=7.5..8.0
"""
import os, sys, subprocess, numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
from musclemimic.environments.humanoids import MyoFullBody

CACHE = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz")
STO = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/shedance/osim/pop_so_active_7.5_8.0_activation.sto")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/shedance/renders/pop_muscle_active.mp4")
W, H, FPS = 720, 960, 50


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


def main():
    c = np.load(CACHE, allow_pickle=True)
    qpos = gaussian_smooth(np.asarray(c["qpos"], float)); freq = float(c["frequency"])
    env = MyoFullBody(disable_fingers=True)
    m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, mujoco.MjModel))
    d = mujoco.MjData(m)
    mjc_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]

    hdr, sto = read_sto(STO)
    oset = set(hdr)
    tcol = sto[:, hdr.index("time")]

    def mjc_to_osim(n):
        if n in oset: return n
        if n.endswith("_left") and n[:-5] + "_l" in oset: return n[:-5] + "_l"
        if not n.endswith(("_left", "_l", "_r")) and n + "_r" in oset: return n + "_r"
        return None
    colmap = {i: hdr.index(mjc_to_osim(nm)) for i, nm in enumerate(mjc_names) if mjc_to_osim(nm)}
    print(f"mapped {len(colmap)}/{m.nu} MuJoCo muscles to OpenSim activations; sto frames={len(tcol)}")

    so = mujoco.MjvOption()
    so.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
    act2ten = [int(m.actuator_trnid[i, 0]) for i in range(m.nu)]   # muscle -> its tendon id

    def color_by_act():
        for i in range(m.nu):
            tid = act2ten[i]
            if 0 <= tid < m.ntendon:
                a = float(d.act[i]) if i < d.act.shape[0] else 0.0
                a = min(1.0, max(0.0, a))
                m.tendon_rgba[tid] = [0.2 + 0.8 * a, 0.15 + 0.15 * a, 0.4 * (1 - a), 0.55 + 0.45 * a]
    cam = mujoco.MjvCamera(); cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 120; cam.lookat[:] = [0, 0, 0.95]
    m.vis.global_.offwidth = W; m.vis.global_.offheight = H   # enlarge offscreen framebuffer
    rend = mujoco.Renderer(m, height=H, width=W)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = os.path.expanduser("~/shedance/renders/_mframes"); os.makedirs(tmp, exist_ok=True)
    STEP = int(os.environ.get("RENDER_STEP", "1"))   # render every STEP-th frame (speed for long clips)
    nf = 0
    for fi, t in enumerate(tcol):
        if fi % STEP != 0:
            continue
        qi = int(round(t * freq))
        if qi < 0 or qi >= qpos.shape[0]: continue
        d.qpos[:] = qpos[qi]; d.qvel[:] = 0
        d.act[:] = 0.0
        for ai, col in colmap.items():
            if ai < d.act.shape[0]:
                d.act[ai] = sto[fi, col]
        mujoco.mj_forward(m, d)
        color_by_act()
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.95]
        rend.update_scene(d, cam, so)
        img = rend.render()
        # write ppm-ish via imageio? use raw -> let ffmpeg read png via mujoco? simplest: save with numpy->png
        from PIL import Image
        Image.fromarray(img).save(os.path.join(tmp, f"f{nf:05d}.png")); nf += 1
    rend.close()
    print(f"rendered {nf} frames -> ffmpeg")
    subprocess.run(["ffmpeg", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", OUT],
                   check=True, capture_output=True)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
