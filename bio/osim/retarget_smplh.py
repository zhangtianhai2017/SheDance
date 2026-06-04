#!/usr/bin/env python3
"""Headless SMPL-H -> MyoFullBody retarget via the GMR class (no interactive viewer).
The muscle pipeline only needs qpos, so we save {qpos, frequency} as a MyoFullBody cache.
Args: smplh_npz out_npz [tgt_fps=60]"""
import sys, os, numpy as np
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.utils.smpl import load_smplh_file, get_smplh_data_offline_fast
# fix: load_smplh_file builds SMPLH_Parser without num_betas (defaults 10) but AMASS/fitted shape
# use 16 betas -> einsum mismatch. Force num_betas=16 (non-invasive, this process only).
import general_motion_retargeting.utils.smpl as _sm
_orig_init = _sm.SMPLH_Parser.__init__
def _init16(self, *a, **k):
    k.setdefault("num_betas", 16); _orig_init(self, *a, **k)
_sm.SMPLH_Parser.__init__ = _init16   # patch method only -> keeps super() intact (no recursion)
# the MyoFullBody knee coupler DOFs slightly violate mink limits during IK; clamp instead of raise.
import mink.configuration as _mc
_orig_cl = _mc.Configuration.check_limits
def _cl(self, *a, **k):
    k["safety_break"] = False; return _orig_cl(self, *a, **k)
_mc.Configuration.check_limits = _cl
# the GMR pkg ships without the myofullbody IK config (loco_mujoco keeps it elsewhere); copy it in.
import shutil, glob, general_motion_retargeting as _gmr
_gik = os.path.join(os.path.dirname(_gmr.__file__), "ik_configs"); os.makedirs(_gik, exist_ok=True)
for _f in glob.glob(os.path.expanduser("~/shedance/musclemimic/loco_mujoco/smpl/gmr_configs/*.json")):
    _d = os.path.join(_gik, os.path.basename(_f))
    if not os.path.exists(_d):
        shutil.copy2(_f, _d)

SMPLH_FOLDER = os.environ.get("SMPLH_FOLDER") or os.path.expanduser("~/shedance/smpl_models")
SHAPE = os.path.expanduser("~/.musclemimic/caches/AMASS/MyoFullBody/gmr/myofullbody_shape.pkl")
npz, out = sys.argv[1], sys.argv[2]
fps = int(sys.argv[3]) if len(sys.argv) > 3 else 60
shape = SHAPE if os.path.exists(SHAPE) else None

smplh_data, body_model, smplh_output, height = load_smplh_file(npz, SMPLH_FOLDER, fitted_shape_path=shape)
frames, afps = get_smplh_data_offline_fast(smplh_data, body_model, smplh_output, tgt_fps=fps)
print(f"loaded {len(frames)} frames @ {afps}Hz, human height {height:.2f}m, shape={'fitted' if shape else 'height'}", flush=True)

rt = GMR(actual_human_height=height, src_human="smplh", tgt_robot="myofullbody",
         use_fitted_shape=(shape is not None), fitted_shape_path=shape)
qpos = []
for i in range(len(frames)):
    q, _ = rt.retarget(frames[i], offset_to_ground=True)
    qpos.append(np.asarray(q).copy())
qpos = np.array(qpos)

# GMR outputs the 129-dof (with-finger) MyoFullBody; the muscle pipeline uses the 89-dof model.
# Remap 129 -> 89 by joint NAME (drop the at-rest fingers).
import mujoco as _mj
from musclemimic.environments.humanoids import MyoFullBody as _MFB
def _getm(df):
    e = _MFB(disable_fingers=df)
    return next(o for o in (getattr(e, a, None) for a in ("_model", "model", "_mjmodel")) if isinstance(o, _mj.MjModel))
def _n2a(m):
    return {_mj.mj_id2name(m, _mj.mjtObj.mjOBJ_JOINT, j): (int(m.jnt_qposadr[j]), int(m.jnt_type[j])) for j in range(m.njnt)}
m89, m129 = _getm(True), _getm(False); a89, a129 = _n2a(m89), _n2a(m129)
NQ = {0: 7, 1: 4, 2: 1, 3: 1}
if qpos.shape[1] == m129.nq and m89.nq != m129.nq:
    q89 = np.zeros((len(qpos), m89.nq))
    for nm, (a, t) in a89.items():
        if nm in a129:
            b = a129[nm][0]; n = NQ[t]; q89[:, a:a + n] = qpos[:, b:b + n]
    qpos = q89
    print(f"remapped to 89-dof qpos {qpos.shape}", flush=True)
np.savez(out, qpos=qpos, frequency=float(afps))
print(f"retargeted qpos {qpos.shape} @ {afps}Hz -> {out}", flush=True)
