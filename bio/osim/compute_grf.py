#!/usr/bin/env python3
"""Estimate ground reaction force (GRF) from whole-body COM dynamics and write OpenSim
ExternalLoads (forces .mot + .xml). Without force plates: F_grf = M*(a_com - g), distributed
to the planted foot/feet, applied at the foot under-point (COP est). Lets SO/Moco give
PHYSIOLOGICAL leg muscle activation (legs bear weight) and zeroes the pelvis_ty residual.
"""
import os, numpy as np
import opensim as osim

HERE = os.path.expanduser("~/shedance/osim")
MODEL = os.path.join(HERE, "cyclist_min.osim")
MOT = os.path.join(HERE, "pop_ik.mot")          # degrees
OUT_MOT = os.path.join(HERE, "pop_grf.mot")
OUT_XML = os.path.join(HERE, "pop_grf_extloads.xml")
G = np.array([0.0, -9.81, 0.0])                  # OpenSim Y-up


def read_mot(p):
    L = open(p).read().splitlines()
    hi = next(i for i, l in enumerate(L) if l.strip().lower() == "endheader")
    indeg = any("indegrees=yes" in l.lower() for l in L[:hi])
    hdr = L[hi + 1].split("\t")
    d = np.array([l.split("\t") for l in L[hi + 2:] if l.strip()], float)
    return hdr, d, indeg


def main():
    model = osim.Model(MODEL); state = model.initSystem()
    M = model.getTotalMass(state)
    cs = model.getCoordinateSet()
    bs = model.getBodySet()
    hdr, d, indeg = read_mot(MOT)
    ci = {hdr[i]: i for i in range(len(hdr))}
    t = d[:, 0]; T = len(t); dt = t[1] - t[0]

    com = np.zeros((T, 3)); footR = np.zeros((T, 3)); footL = np.zeros((T, 3))
    for k in range(T):
        for i in range(cs.getSize()):
            nm = cs.get(i).getName()
            if nm in ci:
                v = d[k, ci[nm]]
                if indeg and cs.get(i).getMotionType() != 2:
                    v = np.radians(v)
                try: cs.get(i).setValue(state, float(v), False)
                except Exception: pass
        model.assemble(state); model.realizePosition(state)
        com[k] = model.calcMassCenterPosition(state).to_numpy()
        footR[k] = bs.get("calcn_r").getPositionInGround(state).to_numpy()
        footL[k] = bs.get("calcn_l").getPositionInGround(state).to_numpy()

    a_com = np.zeros((T, 3)); a_com[1:-1] = (com[2:] - 2 * com[1:-1] + com[:-2]) / dt ** 2
    grf = M * (a_com - G[None, :])               # total GRF (N), world

    # distribute to feet by how low each is (Y-up: lower y = more planted)
    cols = ["time",
            "R_ground_force_vx", "R_ground_force_vy", "R_ground_force_vz",
            "R_ground_force_px", "R_ground_force_py", "R_ground_force_pz",
            "L_ground_force_vx", "L_ground_force_vy", "L_ground_force_vz",
            "L_ground_force_px", "L_ground_force_py", "L_ground_force_pz",
            "R_ground_torque_x", "R_ground_torque_y", "R_ground_torque_z",
            "L_ground_torque_x", "L_ground_torque_y", "L_ground_torque_z"]
    data = np.zeros((T, len(cols))); data[:, 0] = t
    for k in range(T):
        yr, yl = footR[k, 1], footL[k, 1]
        ymin = min(yr, yl)
        wr = max(0.0, 0.08 - (yr - ymin)); wl = max(0.0, 0.08 - (yl - ymin))
        s = wr + wl + 1e-9; wr, wl = wr / s, wl / s
        Fr = wr * grf[k]; Fl = wl * grf[k]
        pr = footR[k].copy(); pr[1] = 0.0        # COP under foot at ground
        pl = footL[k].copy(); pl[1] = 0.0
        data[k, 1:4] = Fr; data[k, 4:7] = pr
        data[k, 7:10] = Fl; data[k, 10:13] = pl
        # torques 0 (point already at COP)

    with open(OUT_MOT, "w") as f:
        f.write("pop_grf\nversion=1\n")
        f.write(f"nRows={T}\nnColumns={len(cols)}\ninDegrees=no\nendheader\n")
        f.write("\t".join(cols) + "\n")
        for k in range(T):
            f.write("\t".join(f"{v:.6f}" for v in data[k]) + "\n")
    print(f"wrote {OUT_MOT}: mass={M:.1f}kg bodyweight={M*9.81:.0f}N, GRF |F| mean={np.linalg.norm(grf[2:-2],axis=1).mean():.0f}N")

    # ExternalLoads xml (2 ExternalForce: R, L on calcn)
    el = osim.ExternalLoads(); el.setDataFileName(OUT_MOT)
    for side, body in [("R", "calcn_r"), ("L", "calcn_l")]:
        ef = osim.ExternalForce()
        ef.setName(side + "_grf")
        ef.set_applied_to_body(body)
        ef.set_force_expressed_in_body("ground")
        ef.set_point_expressed_in_body("ground")
        ef.set_force_identifier(side + "_ground_force_v")
        ef.set_point_identifier(side + "_ground_force_p")
        ef.set_torque_identifier(side + "_ground_torque_")
        el.cloneAndAppend(ef)
    el.printToXML(OUT_XML)
    print(f"wrote {OUT_XML}")


if __name__ == "__main__":
    main()
