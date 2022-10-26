import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from KITTI import KITTINMPairDataset

# parser = argparse.ArgumentParser()
# parser.add_argument("")

# get ppair of point clouds from the dataset
dataset = KITTINMPairDataset(phase="train")
loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
eps = 10  # threshold for distances

N = len(loader)
pc_pair, _ = dataset.__getitem__(0)
n_seq = 10
n_pts = pc_pair.shape[1]
all_inliers = torch.full((N, n_pts, n_pts), False)

print(n_pts)
print("Started computing inliers.")
for idx, (pc_pair, gt_pose) in enumerate(loader):

    # for testing purposes only
    if idx == 1:
        break

    print(f"[BATCH : {idx}]")

    # send all tensors to device
    pc_pair = pc_pair.to(device).float()
    R, t = utils.get_rot_trans(gt_pose)
    R, t = R.to(device).float(), t.to(device).float()

    # get source and target point clouds from data
    src = pc_pair[0, 0, :, :]
    tgt = pc_pair[0, 1, :, :]

    distances = torch.zeros((src.shape[0], tgt.shape[0])).to(device)
    inliers = torch.zeros((src.shape[0], tgt.shape[0])).to(device)

    for i, src_pt in enumerate(src):
        print(f"Computing Distances for point {i} in source cloud")
        src_pt = torch.mul(R, src_pt) + t

        for j, tgt_pt in enumerate(tgt):
            distances[i][j] = torch.dist(src_pt, tgt_pt, 2)
            # all_inliers[idx][i][j] = torch.lt(distances[i][j], eps)exi
            all_inliers[idx][i][j] = distances[i][j]

# convert pytorch tensors to numpy
all_inliers = all_inliers.numpy()  # <N x n_pts x n_pts x 3>
test = distances.cpu().numpy()
print(test.shape)
print(test)

# save dataset
np.save("inliers.npy", test)
