import torch
from torch.utils.data import DataLoader
import numpy as np
import open3d as o3d
import os
import h5py
from kitti import KITTIDataset
from utils import *
from easydict import EasyDict as edict

CFG_DIR = os.path.join(os.getcwd(), "config", "kitti.yaml")
config = edict(load_config(CFG_DIR))
dataset = KITTIDataset(split="test", data_augmentation=False, config=config)
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

# directory to store labels
os.makedirs("labels", exist_ok=True)

# get a pair of point clouds from the dataset
for idx, data in enumerate(loader):
    
    # read the source and target point clouds
    # and the ground truth transformation
    src = data[0].float().squeeze(0)
    tgt = data[1].float().squeeze(0)
    rot = data[2].float().squeeze(0)
    trans = data[3].float().squeeze(0)

    src_t = (rot @ src.T + trans).T
    
    l2_dist = torch.cdist(src_t, tgt)
    threshold = 15 * torch.min(l2_dist) # set to 15 after plotting outlier count for values 1...100

    overlap_indexes = torch.nonzero(l2_dist < threshold)

    gt = torch.zeros(src_t.shape[0], 2)
    gt[overlap_indexes[:, 0], 0] = 1
    gt[overlap_indexes[:, 0], 1] = 1
    
    GT_PATH = os.path.join("labels", f"{idx}.npy")
    gt.numpy().tofile(GT_PATH)
    
    # # mark overlapping points as 1
    # src_gt[overlap_indexes[:, 0]] = 1
    # tgt_gt[overlap_indexes[:, 1]] = 1