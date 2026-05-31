#!/usr/bin/env python3
"""AIST++ SMPL motion (.pkl) -> AMASS-style SMPL-H .npz for MuscleMimic GMR-Fit retargeting.

AIST++ motion pkl keys:
  smpl_poses   (N, 72) float32  axis-angle, 24 SMPL joints (global + 23 body), no hands
  smpl_trans   (N, 3)  float32  root translation (in AIST++ scaled units)
  smpl_scaling (1,)    float32  scale factor; meters = smpl_trans / smpl_scaling
AIST++ is sampled at 60 fps.

MuscleMimic loco_mujoco.smpl.retargeting.load_amass_data expects an .npz with:
  poses (N, >=66)   -- it reads poses[:, :66] then pads 6 zeros -> 72
  trans (N, 3)
  betas             -- shape params (we use mean body = zeros)
  gender            -- we only built SMPLH_NEUTRAL.pkl, so 'neutral'
  mocap_framerate   -- fps

Place the output under the configured AMASS root (musclemimic-set-amass-path),
e.g. <amass_root>/AIST/<name>.npz, so load_amass_data resolves the key 'AIST/<name>'.
"""
import argparse
import os
import pickle

import numpy as np

AIST_FPS = 60.0


def convert(pkl_path: str, out_npz: str) -> None:
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)

    poses72 = np.asarray(d["smpl_poses"], dtype=np.float32)         # (N, 72) SMPL 24-joint axis-angle
    n = poses72.shape[0]
    # GMR load_smplh_file takes the 'poses' branch only when shape[1]==156
    # (SMPL-H = root(3)+body(63)+left_hand(45)+right_hand(45)). AIST++ has no fingers,
    # so copy root+21 body joints (first 66) and leave the 90 hand dims as zeros.
    poses = np.zeros((n, 156), dtype=np.float32)
    poses[:, :66] = poses72[:, :66]

    scaling = float(np.asarray(d["smpl_scaling"]).reshape(-1)[0])
    trans = np.asarray(d["smpl_trans"], dtype=np.float32) / scaling  # (N, 3) meters
    # 16 betas (zeros = mean body) to match AMASS convention. The GMR SMPLH_Parser
    # is forced to num_betas=16 via monkeypatch in retarget_pop.py (the installed
    # GMR/loco code omits num_betas=16 at instantiation -> latent shapedirs(10) vs
    # betas(16) mismatch). With that patch, 16-dim betas are consistent on all paths.
    betas = np.zeros(16, dtype=np.float32)

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(
        out_npz,
        poses=poses,
        trans=trans,
        betas=betas,
        gender="neutral",
        mocap_framerate=np.float32(AIST_FPS),
    )
    print(f"wrote {out_npz}  frames={poses.shape[0]}  dur={poses.shape[0] / AIST_FPS:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, help="AIST++ motion .pkl path")
    ap.add_argument("--out", required=True, help="output AMASS-style .npz path")
    args = ap.parse_args()
    convert(args.pkl, args.out)
