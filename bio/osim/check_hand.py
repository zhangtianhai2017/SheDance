#!/usr/bin/env python3
"""Check what hand/wrist/finger motion actually exists in the qpos cache (decides whether
adding the hand back is meaningful). Also which MyoFullBody variant matches the cache."""
import os, numpy as np, mujoco
from musclemimic.environments.humanoids import MyoFullBody
c = np.load(os.path.expanduser(
    "~/.musclemimic/caches/AMASS/MyoFullBody/gmr/AIST/gPO_sBM_cAll_d10_mPO0_ch01.npz"), allow_pickle=True)
q = np.asarray(c["qpos"], float)
print("cache qpos shape:", q.shape)
for df in (True, False):
    try:
        env = MyoFullBody(disable_fingers=df)
        m = next(o for o in (getattr(env, a, None) for a in ("_model", "model", "_mjmodel"))
                 if isinstance(o, mujoco.MjModel))
        tag = "  <== MATCHES cache" if m.nq == q.shape[1] else ""
        print(f"disable_fingers={df}: nq={m.nq} njnt={m.njnt}{tag}")
        if m.nq == q.shape[1]:
            for j in range(m.njnt):
                nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
                if any(k in nm.lower() for k in ("wrist", "hand", "finger", "thumb", "mcp", "pip",
                                                 "dip", "carp", "radius", "lunate", "flex", "dev")):
                    adr = int(m.jnt_qposadr[j])
                    rng = float(q[:, adr].max() - q[:, adr].min())
                    print(f"    joint {nm}: motion range = {np.degrees(rng):.2f} deg")
    except Exception as e:
        print(f"disable_fingers={df}: error {e}")
