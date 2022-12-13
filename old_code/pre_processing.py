import argparse

import open3d as o3d
import torch
from easydict import EasyDict as edict
from torch.utils.data import DataLoader

import lib.utils as lu
from kitti import KITTIDataset

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=float, default=1.0, help="Sets the distance threshold to compute overlapping points")
args = parser.parse_args()

torch.cuda.empty_cache()

# threshold for distances
config = edict(lu.load_config("configs/kitti.yaml"))

# get a pair of point clouds from the dataset
dataset = KITTIDataset(split="train", data_augmentation=False, config=config)
loader = DataLoader(dataset, batch_size=100, shuffle=False, drop_last=True)

# find appropriate dimensions of gt
B = len(loader)
N = dataset.__getitem__(0)[0].shape[0]

ground_truth = torch.zeros((B, N, 2)).cuda()  # batch x npts x 2(src, tgt)
print(B)
# # find overlapping points
# for idx, (src, tgt, rot, trans) in enumerate(loader):
#
#     if idx == 1:
#         break
#
#     # send all things to device
#     src, tgt, rot, trans = src.cuda().float(), tgt.cuda().float(), rot.cuda().float(), trans.cuda().float()
#
#     # find overlapping points
#     print(src.shape, tgt.shape, rot.shape, trans.shape)
#     src_t = torch.zeros_like(src)
#     src_t[idx] = (rot[idx] @ src[idx].T + trans[idx]).T
#
#     # Computing the euclidean distance between the source and target point clouds.
#     bool_ovlp = torch.cdist(src_t[idx], tgt[idx], 2, compute_mode="donot_use_mm_for_euclid_dist") < args.eps
#     ovlp_idx = bool_ovlp.nonzero()
#     ground_truth[idx,
#
# # Getting the first item from the dataset.
# src_pcd, tgt_pcd, rot, trans = dataset.__getitem__(0)
#
# # Converting the numpy array to a tensor.
# src, tgt = torch.Tensor(src_pcd), torch.Tensor(tgt_pcd)
#
# # transform source point cloud by ground truth transformation
# src_t = rot @ src.T + trans
#
# # Visualizing the point clouds.
# src_p = o3d.geometry.PointCloud()
# src_p.points = o3d.utility.Vector3dVector(src)
# src_p.paint_uniform_color([1, 0, 0])
#
# src_tp = o3d.geometry.PointCloud()
# src_tp.points = o3d.utility.Vector3dVector(src_t.T)
# src_tp.paint_uniform_color([0, 0, 1])
#
# tgt_p = o3d.geometry.PointCloud()
# tgt_p.points = o3d.utility.Vector3dVector(tgt)
# tgt_p.paint_uniform_color([0, 1, 0])
#
# o3d.visualization.draw_geometries([tgt_p, src_p], window_name="T vs S")
# o3d.visualization.draw_geometries([tgt_p, src_tp], window_name="T vs S transformed")
