#!/usr/bin/env python3
"""Does the source model have muscles that actually actuate the wrist? (decides whether the
wrist can be MUSCLE-driven, not just reserve-driven). Lists muscles with moment arm on wrist coords."""
import os, opensim as osim
HERE = os.path.expanduser("~/shedance/osim")
m = osim.Model(os.path.join(HERE, "cyclistFullBodyMuscle.osim"))
s = m.initSystem(); m.realizePosition(s)
cs = m.getCoordinateSet()
wrist = [cs.get(i) for i in range(cs.getSize())
         if any(k in cs.get(i).getName() for k in ("wrist_flex", "wrist_dev"))]
print("wrist coords:", [c.getName() for c in wrist])
mus = m.getMuscles(); hits = {}
for i in range(mus.getSize()):
    mu = mus.get(i)
    for c in wrist:
        try:
            ma = abs(mu.computeMomentArm(s, c))
            if ma > 1e-3:
                hits.setdefault(mu.getName(), []).append(f"{c.getName()}={ma:.3f}")
        except Exception:
            pass
print(f"total muscles in model: {mus.getSize()}")
print(f"muscles actuating the wrist: {len(hits)}")
for nm, v in list(hits.items())[:25]:
    print("  ", nm, v)
