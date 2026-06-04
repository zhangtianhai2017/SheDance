#!/usr/bin/env python3
"""Effort-vision plot: muscle activation over time per F_max archetype. On a feasible segment the
motions match but the WEAKER body strains more (higher activation) = the embodied imitator in the
effort dimension. Args: fmax values (default 0.2 0.4 1.0)."""
import sys, os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.expanduser("~/shedance/osim")
fmaxes = [float(x) for x in sys.argv[1:]] or [0.4, 1.0]


def read_sto(p):
    L = open(p).read().splitlines()
    hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    hdr = L[hi + 1].split("\t")
    d = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
    return hdr, d


fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
COL = {0.2: "darkred", 0.4: "tab:red", 1.0: "tab:green", 2.5: "tab:blue"}
LAB = {0.2: "Fmax x0.2 (very weak)", 0.4: "x0.4 (weak)", 1.0: "x1.0 (normal)", 2.5: "x2.5 (strong)"}
means = []
for fm in fmaxes:
    f = os.path.join(HERE, f"popB_7.40_7.80_f{fm:.2f}.sto")
    if not os.path.exists(f):
        print(f"skip {fm} (missing)"); continue
    hdr, d = read_sto(f); t = d[:, 0]
    mc = [i for i in range(len(hdr)) if "/forceset/" in hdr[i] and "reserve" not in hdr[i] and "res_" not in hdr[i]]
    A = np.abs(d[:, mc])
    perframe = A.mean(1)
    c = COL.get(fm, "gray")
    ax[0].plot(t, perframe, color=c, lw=2, label=LAB.get(fm, str(fm)))
    means.append((fm, A.mean(), (A > 0.05).mean()))
    print(f"Fmax {fm}: mean act {A.mean():.3f}, frac active {(A>0.05).mean():.2f}")
ax[0].set_xlabel("time (s)"); ax[0].set_ylabel("mean muscle activation"); ax[0].grid(alpha=0.3); ax[0].legend()
ax[0].set_title("Effort over time: weaker body activates more")
if means:
    xs = [str(m[0]) for m in means]
    ax[1].bar(xs, [m[1] for m in means], color=[COL.get(m[0], "gray") for m in means])
    ax[1].set_xlabel("F_max (body strength)"); ax[1].set_ylabel("overall mean activation")
    ax[1].set_title("Same dance, different effort")
fig.suptitle("Embodied imitator (effort dimension) — segment 7.4-7.8s", fontsize=13)
fig.tight_layout()
out = os.path.join(HERE, "effort_vision.png"); fig.savefig(out, dpi=90)
print("wrote", out)
