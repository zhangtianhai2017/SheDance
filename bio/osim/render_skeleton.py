#!/usr/bin/env python3
"""Render MocoTrack solutions as overlaid 3D stick figures -> mp4. Shows 'same dance, different
bodies': strong (blue) / normal (green) / weak (red) F_max archetypes superimposed; where the
weak body deviates from the strong = the embodied imitator. Uses OpenSim forward kinematics to
get joint-center positions. Args: label=sol.sto pairs, e.g. weak=popB_..f0.40.sto norm=..f1.00..
"""
import sys, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclist_min.osim")
OUT = os.path.join(HERE, "vision_overlay.mp4")
# stick-figure bones (joint-center bodies)
BONES = [("pelvis", "femur_r"), ("femur_r", "tibia_r"), ("tibia_r", "talus_r"), ("talus_r", "toes_r"),
         ("pelvis", "femur_l"), ("femur_l", "tibia_l"), ("tibia_l", "talus_l"), ("talus_l", "toes_l"),
         ("pelvis", "lumbar5"), ("lumbar5", "humerus_r"), ("humerus_r", "ulna_r"), ("ulna_r", "radius_r"),
         ("lumbar5", "humerus_l"), ("humerus_l", "ulna_l"), ("ulna_l", "radius_l")]
JOINTS = sorted(set([b for pair in BONES for b in pair]))
COLORS = {"strong": "tab:blue", "norm": "tab:green", "weak": "tab:red"}


def read_sto(p):
    L = open(p).read().splitlines()
    hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    hdr = L[hi + 1].split("\t")
    d = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
    indeg = any("indegrees=yes" in l.lower() for l in L[:hi])
    return hdr, d, indeg


def body_positions(model, state, sol_hdr, sol_row, indeg):
    cs = model.getCoordinateSet()
    val = {sol_hdr[i].split("/")[-2]: i for i in range(len(sol_hdr)) if sol_hdr[i].endswith("/value")}
    for i in range(cs.getSize()):
        nm = cs.get(i).getName()
        if nm in val:
            v = sol_row[val[nm]]
            if indeg and cs.get(i).getMotionType() != 2:
                v = np.radians(v)
            try:
                cs.get(i).setValue(state, float(v), False)
            except Exception:
                pass
    model.assemble(state); model.realizePosition(state)
    bs = model.getBodySet()
    return {b: np.array(bs.get(b).getPositionInGround(state).to_numpy()) for b in JOINTS}


def main():
    items = []
    for a in sys.argv[1:]:
        label, f = a.split("=")
        items.append((label, os.path.join(HERE, f)))
    model = osim.Model(MODEL); state = model.initSystem()
    # load all solutions, precompute body positions per frame
    series = []
    nframes = 1 << 30
    for label, f in items:
        hdr, d, indeg = read_sto(f)
        frames = [body_positions(model, state, hdr, d[k], indeg) for k in range(d.shape[0])]
        series.append((label, frames)); nframes = min(nframes, len(frames))
    print(f"loaded {len(series)} archetypes, {nframes} frames", flush=True)

    fig = plt.figure(figsize=(6, 8)); ax = fig.add_subplot(111, projection="3d")

    def draw(k):
        ax.clear()
        ax.set_xlim(-0.6, 0.6); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 1.9)
        ax.set_box_aspect((1, 1, 1.6)); ax.view_init(elev=8, azim=-70); ax.set_axis_off()
        for label, frames in series:
            P = frames[min(k, len(frames) - 1)]
            c = COLORS.get(label, "gray")
            for b1, b2 in BONES:
                p, q = P[b1], P[b2]
                # OpenSim Y-up -> plot with Z=up: (X, Z, Y)
                ax.plot([p[0], q[0]], [p[2], q[2]], [p[1], q[1]], "-", color=c, lw=2)
            for b in JOINTS:
                p = P[b]; ax.scatter(p[0], p[2], p[1], color=c, s=10)
        ax.set_title(" / ".join(f"{l}" for l, _ in series) + f"   frame {k}", fontsize=9)

    anim = FuncAnimation(fig, draw, frames=nframes, interval=50)
    anim.save(OUT, writer="ffmpeg", fps=20, dpi=80)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
