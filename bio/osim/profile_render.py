#!/usr/bin/env python3
"""Pinpoint the render bottleneck: time mj_forward / update_scene / render() separately,
plus a resolution sweep (is render() pixel-bound or per-call overhead?). Args: sto"""
import time, sys, numpy as np
import render_core as rc
import mujoco

ctx = rc.RenderCtx(sys.argv[1], 1.0)
m, d = ctx.m, ctx.d
frames = list(ctx.frames_in_range())[:13]
tf = ts = tr = 0.0; n = 0
for k, fi in enumerate(frames):
    qi = int(round(ctx.tcol[fi] * ctx.freq))
    if qi >= ctx.qpos.shape[0]:
        continue
    d.qpos[:] = ctx.qpos[qi]; d.qvel[:] = 0; d.act[:] = 0
    d.act[ctx.act_idx] = ctx.sto[fi, ctx.col_idx]
    t = time.time(); mujoco.mj_forward(m, d); dtf = time.time() - t
    a = np.clip(d.act[ctx.all_mu], 0, 1)
    rgba = np.empty((a.shape[0], 4))
    rgba[:, 0] = 0.2 + 0.8 * a; rgba[:, 1] = 0.15 + 0.15 * a; rgba[:, 2] = 0.4 * (1 - a); rgba[:, 3] = 0.55 + 0.45 * a
    m.tendon_rgba[ctx.all_ten] = rgba
    ctx.cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.95]
    t = time.time(); ctx.rend.update_scene(d, ctx.cam, ctx.opt); dts = time.time() - t
    t = time.time(); ctx.rend.render(); dtr = time.time() - t
    if k > 0:
        tf += dtf; ts += dts; tr += dtr; n += 1
    print(f"frame {k}: mj_forward={dtf*1000:.0f}  update_scene={dts*1000:.0f}  render={dtr*1000:.0f} ms", flush=True)
print(f"\nAVG(skip warmup): mj_forward={tf/n*1000:.1f}ms  update_scene={ts/n*1000:.1f}ms  render={tr/n*1000:.1f}ms")

# resolution sweep on render() only
for res in (128, 256, 512, 720):
    r = mujoco.Renderer(m, height=res, width=res)
    r.update_scene(d, ctx.cam, ctx.opt)
    r.render()  # warmup
    t = time.time()
    for _ in range(5):
        r.render()
    print(f"render@{res}x{res} = {(time.time()-t)/5*1000:.0f} ms/frame")
    r.close()
