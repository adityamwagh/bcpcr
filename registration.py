import torch
from torch.utils.data import DataLoader
from utils import draw_registration_result
import open3d as o3d
import KITTI
import numpy as np
import copy



dataset = KITTI.KITTINMPairDataset(phase="train")
loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

inliers = np.load("inliers.npy")
print(inliers.shape)
print(inliers)
# # loop through the dataset and plot pointclouds
# for idx, (pc_pair, _) in enumerate(loader):

#     pc_pair = pc_pair.to(device).float()


#     # get src and tgt point clouds from data
#     src = pc_pair[0, 0, :, :]
#     tgt = pc_pair[0, 1, :, :]
    
#     # TODO: Read inliers from npy file, and sample min(src, tgt) points from the dataset

#     # convert to numpy arrays
#     src, tgt = src.detach().cpu().numpy(), tgt.detach().cpu().numpy()

    # numpy to open3d structure
    # o3d_src = o3d.geometry.PointCloud()
    # o3d_src.points = o3d.utility.Vector3dVector(src)

    # o3d_tgt = o3d.geometry.PointCloud()
    # o3d_tgt.points = o3d.utility.Vector3dVector(tgt)

    # threshold = 0.02
    # trans_init = np.eye((4, 4))

    # # print("Apply point-to-point ICP")
    # # reg_p2p = o3d.open3d.pipelines.registration.registration_icp(src, tgt, threshold, trans_init,
    # # o3d.pipelines.registration.TransformationEstimationPointToPoint())
    # # print(reg_p2p)
    # # print(f"Transformation is: {reg_p2p.transformation}")
    # # draw_registration_result(src, tgt, reg_p2p.transformation)

    # print("Apply point-to-plane ICP")
    # reg_p2l = o3d.pipelines.registration.registration_icp(src, tgt, threshold, trans_init,
    # o3d.pipelines.registration.TransformationEstimationPointToPlane())
    # print(reg_p2l)
    # print(f"Transformation is: {reg_p2l.transformation}")
    # draw_registration_result(src, tgt, reg_p2l.transformation)


