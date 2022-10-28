from matplotlib.scale import InvertedSymmetricalLogTransform
import numpy as np
import torch
from torch.utils.data import DataLoader
import open3d as o3d

import utils
from KITTI import KITTINMPairDataset

# get pair of point clouds from the dataset
dataset = KITTINMPairDataset(phase="train")
loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)

# load inlier coordinates
pc_pair, gt_pose = next(iter(loader))

# get rotation and translation
pc_pair = pc_pair.float()
R, t = utils.get_rot_trans(gt_pose)
R, t = R.float(), t.float()
R, t = R[-1, :, :], t[-1, :]

# get source and target point clouds from data
src = pc_pair[0, 0, :, :]
tgt = pc_pair[0, 1, :, :]

# print(src.shape)
# transform source cloud using gt
# src_trans = R @ src.T + t

# get overlapping point indices from stored vals
all_overlapping_pts = np.load("/scratch/amw9425/projects/bcpcr/overlapping_pts_1.0.npy")

# print(f"Shape of all_ov_pts : {all_overlapping_pts[:, 1]}")
# fetch overlapping indices from all_overlapping_pts
src_ovidx = np.array(all_overlapping_pts[:, 1])
tgt_ovidx = np.array(all_overlapping_pts[:, 2])

src_ovidx = torch.from_numpy(src_ovidx).long().unsqueeze(1)
tgt_ovidx = torch.from_numpy(tgt_ovidx).long().unsqueeze(1)

print(type(src_ovidx))
print(src_ovidx.shape)
print(tgt_ovidx.shape)

src_ovpts = src[src_ovidx].squeeze(1)
tgt_ovpts = tgt[tgt_ovidx].squeeze(1)

print(src_ovpts)
src_ovpts, tgt_ovpts = src_ovpts.cpu().numpy(), tgt_ovpts.cpu().numpy()
src, tgt = src.cpu().numpy(), tgt.cpu().numpy()

print(type(src_ovpts))
print(src_ovpts.shape)
# draw src, tgt, and src & tgt overlapping points using open3d
src_pcd = o3d.geometry.PointCloud()
src_pcd.points = o3d.utility.Vector3dVector(src)

tgt_pcd = o3d.geometry.PointCloud()
tgt_pcd.points = o3d.utility.Vector3dVector(tgt)

src_ovpts_pcd = o3d.geometry.PointCloud()
src_ovpts_pcd.points = o3d.utility.Vector3dVector(src_ovpts)

tgt_ovpts_pcd = o3d.geometry.PointCloud()
tgt_ovpts_pcd.points = o3d.utility.Vector3dVector(tgt_ovpts)

# color src red, tgt green and overlap blue
src_pcd.paint_uniform_color([1, 0, 0])
tgt_pcd.paint_uniform_color([0, 1, 0])
src_ovpts_pcd.paint_uniform_color([0, 0, 1])
tgt_ovpts_pcd.paint_uniform_color([0, 0, 1])

# show visualization
o3d.visualization.draw_geometries(
    [src_pcd, tgt_pcd, src_ovpts_pcd, tgt_ovpts_pcd], zoom=0.7, front=[0.5439, -0.2333, -0.8060], lookat=[2.4615, 2.1331, 1.338], up=[-0.1781, -0.9708, 0.1608]
)
