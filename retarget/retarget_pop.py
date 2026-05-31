#!/usr/bin/env python3
"""Retarget one AMASS-format motion to a MuscleMimic muscle model via GMR-Fit.

WHY THE MONKEYPATCH:
The installed `general_motion_retargeting` SMPLH_Parser is instantiated WITHOUT
num_betas=16 (smpl.py / shape_fitting.py / loco_mujoco retargeting.py all omit it),
so smplx falls back to num_betas=10 -> shapedirs has 10 components. But the pipeline
feeds a 16-dim fitted MyoFullBody shape (and 16-beta AMASS data), giving:
    RuntimeError: einsum(): subscript l has size 10 ... does not broadcast with size 16
The code comments even say "model created with num_betas=16" -- the instantiation just
forgets to pass it. We force num_betas=16 on the GMR parser only (loco's own parser
keeps its default 10 + 16->10 slicing, so it is unaffected).

Run (inside the musclemimic uv env):
    uv run python retarget_pop.py --motion AIST/gPO_sBM_cAll_d10_mPO0_ch01 --clear-cache
"""
import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

import argparse

# --- monkeypatch: force GMR SMPLH_Parser to num_betas=16 (see module docstring) ---
import general_motion_retargeting.utils.smpl as _gmr_smpl

_orig_smplh_init = _gmr_smpl.SMPLH_Parser.__init__


def _patched_smplh_init(self, *args, **kwargs):
    kwargs.setdefault("num_betas", 16)
    return _orig_smplh_init(self, *args, **kwargs)


_gmr_smpl.SMPLH_Parser.__init__ = _patched_smplh_init
# ----------------------------------------------------------------------------------

from loco_mujoco.task_factories import ImitationFactory, AMASSDatasetConf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True, help="AMASS-root-relative key, e.g. AIST/gPO_...")
    ap.add_argument("--model", default="MyoFullBody")
    ap.add_argument("--target-fps", type=int, default=30)
    ap.add_argument("--clear-cache", action="store_true")
    args = ap.parse_args()

    conf = AMASSDatasetConf([args.motion])
    conf.clear_cache = args.clear_cache
    conf.retargeting_method = "gmr"
    conf.gmr_config = {
        "src_human": "smplh",
        "target_fps": args.target_fps,
        "solver": "daqp",
        "damping": 0.5,
        "offset_to_ground": False,
        "use_velocity_limit": False,
        "verbose": False,
    }

    env = ImitationFactory.make(args.model, amass_dataset_conf=conf, headless=True)
    del env
    print("RETARGET_OK", args.motion)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
