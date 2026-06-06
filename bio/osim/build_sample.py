import os, shutil, numpy as np
O = os.path.expanduser("~/shedance/osim")
TOP = "/mnt/c/work/2026/Claude/SheDance"
S = TOP + "/SheDance_sample_dance1"; DS = TOP + "/SheDance_data_structure"; R = TOP + "/renders"
mp4 = R + "/our_humanoid_dance1_locked.mp4"
sz = os.path.getsize(mp4)
assert sz > 10000, f"skin mp4 looks broken: {sz} bytes"
print("skin preview:", sz, "bytes  OK")
shutil.rmtree(S, ignore_errors=True)
for d in ("data", "ref", "preview"):
    os.makedirs(S + "/" + d, exist_ok=True)
shutil.copy(O + "/our_dance1_poses_locked.npz", S + "/data/dance1_poses.npz")
shutil.copy(O + "/dance1_joints_world.npz", S + "/data/dance1_joints_world.npz")
z = np.load(O + "/ourdance1_locked_cache.npz", allow_pickle=True); q = np.asarray(z["qpos"])
np.savez(S + "/data/dance1_qpos.npz", qpos=q, frequency=float(z["frequency"]),
         has_fingers=bool(q.shape[1] >= 129), dof=int(q.shape[1]))   # self-describing flags per the contract
shutil.copy(O + "/ourdance1_locked_par_activation.sto", S + "/data/dance1_activation.sto")
shutil.copy(O + "/ourdance1_locked_ik.mot", S + "/data/dance1_ik.mot")
shutil.copy(TOP + "/SheDance_DATA_CONTRACT.md", S + "/README.md")     # locked format contract as the readme
for f in os.listdir(DS + "/ref"):
    shutil.copy(DS + "/ref/" + f, S + "/ref/" + f)
shutil.copy(mp4, S + "/preview/dance1_skin.mp4")
print("sample rebuilt | qpos", q.shape, "has_fingers", bool(q.shape[1] >= 129), "dof", q.shape[1])
for root, _, files in os.walk(S):
    for f in sorted(files):
        print("  ", os.path.relpath(os.path.join(root, f), S))
