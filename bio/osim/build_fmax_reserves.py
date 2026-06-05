#!/usr/bin/env python3
"""Build an F_max-scaled reserves model: take cyclist_min_reserves.osim and scale every muscle's
max-isometric-force by FMAX (the character/strength knob). <1 = weaker body (tracks the same dance
with HIGHER activation -> redder muscles in the render; reserves cover what muscles can't).
Args: FMAX_SCALE (e.g. 0.5)"""
import sys, os
import opensim as osim
HERE = os.path.expanduser("~/shedance/osim")
FMAX = float(sys.argv[1])
m = osim.Model(os.path.join(HERE, "cyclist_min_reserves.osim"))
mus = m.getMuscles()
for i in range(mus.getSize()):
    mu = mus.get(i); mu.setMaxIsometricForce(mu.getMaxIsometricForce() * FMAX)
m.initSystem()
out = os.path.join(HERE, f"cyclist_min_reserves_f{int(round(FMAX*100)):02d}.osim")
m.printToXML(out)
print(f"FMAX={FMAX} scaled {mus.getSize()} muscles -> {out}", flush=True)
