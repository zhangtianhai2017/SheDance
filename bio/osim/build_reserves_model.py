#!/usr/bin/env python3
"""Pre-build cyclist_min + reserve/residual actuators ONCE -> cyclist_min_reserves.osim,
so parallel SO chunk processes all load the same prebuilt model (no write collision)."""
import os
import opensim as osim
HERE = os.path.expanduser("~/shedance/osim")
model = osim.Model(os.path.join(HERE, "cyclist_min.osim"))
cs = model.getCoordinateSet()
for i in range(cs.getSize()):
    c = cs.get(i)
    if c.getDefaultLocked():
        continue
    nm = c.getName()
    ca = osim.CoordinateActuator(nm); ca.setName("res_" + nm)
    of = 1000.0 if "pelvis" in nm.lower() else 1.0   # pelvis residual cheap; joint reserve expensive (muscles lead)
    ca.setOptimalForce(of); ca.setMinControl(-1e4); ca.setMaxControl(1e4)
    model.addForce(ca)
model.finalizeConnections(); model.initSystem()
OUT = os.path.join(HERE, "cyclist_min_reserves.osim")
model.printToXML(OUT)
print("wrote", OUT, "muscles", model.getMuscles().getSize(), "forces", model.getForceSet().getSize())
