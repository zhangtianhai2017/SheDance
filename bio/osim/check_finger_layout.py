#!/usr/bin/env python3
"""Compare MyoFullBody no-finger(89) vs with-finger(129) qpos layouts, and list the finger DOFs
(name + qpos address) so we can drive fingers synthetically and render them."""
import numpy as np, mujoco
from musclemimic.environments.humanoids import MyoFullBody


def getm(df):
    env = MyoFullBody(disable_fingers=df)
    return next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel"))
                if isinstance(o, mujoco.MjModel))


m89 = getm(True); m129 = getm(False)
print("nq89", m89.nq, "  nq129", m129.nq, "  finger qpos added:", m129.nq - m89.nq)


def jinfo(m):
    return [(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j), int(m.jnt_qposadr[j]), int(m.jnt_type[j]))
            for j in range(m.njnt)]


j89 = jinfo(m89); j129 = jinfo(m129)
match = len(j89) <= len(j129) and all(j89[i][0] == j129[i][0] and j89[i][1] == j129[i][1] for i in range(len(j89)))
print(f"first {len(j89)} joints identical (name+adr) in 129 model:", match)
n89 = set(x[0] for x in j89)
fingers = [x for x in j129 if x[0] not in n89]
print(f"finger joints only-in-129: {len(fingers)}")
for nm, adr, t in fingers:
    print(f"   {nm}  adr={adr} type={t}")
