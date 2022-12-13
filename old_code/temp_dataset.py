import copy
import glob
import os
import random

import numpy as np
import open3d
from scipy.spatial.transform import Rotation

# Dataset parent class
from torch.utils.data import Dataset

from lib.utils import to_o3d_pcd

# for reproducible sampling in stage 2
np.random.seed(42)


class KITTIDataset(Dataset):
    """
    We follow D3Feat to add data augmentation part.
    We first voxelize the pcd and get matches
    Then we apply data augmentation to pcds. KPConv runs over processed pcds, but later for loss computation, we use pcds before data augmentation
    """

    DATA_FILES = {
        "train": "./configs/train_kitti.txt",
        "val": "./configs/val_kitti.txt",
        "test": "./configs/test_kitti.txt",
    }

    def __init__(self, config, split, data_augmentation=True):
        super(KITTIDataset, self).__init__()
        self.config = config
        self.root = os.path.join(config.root, "dataset")
        self.icp_path = os.path.join(config.root, "icp")
        if not os.path.exists(self.icp_path):
            os.makedirs(self.icp_path)
        self.voxel_size = config.first_subsampling_dl
        self.matching_search_voxel_size = config.overlap_radius
        self.data_augmentation = data_augmentation
        self.augment_noise = config.augment_noise
        self.IS_ODOMETRY = True
        self.max_corr = config.max_points
        self.augment_shift_range = config.augment_shift_range
        self.augment_scale_max = config.augment_scale_max
        self.augment_scale_min = config.augment_scale_min

        # Initiate containers
        self.files = []
        self.kitti_icp_cache = {}
        self.kitti_cache = {}
        self.prepare_kitti_ply(split)
        self.split = split

    def prepare_kitti_ply(self, split):
        assert split in ["train", "val", "test"]

        subset_names = open(self.DATA_FILES[split]).read().split()
        for dirname in subset_names:
            drive_id = int(dirname)
            fnames = glob.glob(self.root + "/sequences/%02d/velodyne/*.bin" % drive_id)
            assert len(fnames) > 0, f"Make sure that the path {self.root} has data {dirname}"
            inames = sorted([int(os.path.split(fname)[-1][:-4]) for fname in fnames])
            for iname in inames:
                self.files.append((drive_id, iname))

        # remove bad pairs
        if split == "test":
            self.files.remove((8, 15, 58))
        print(f"Num_{split}: {len(self.files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        drive = self.files[idx][0]  # drive id
        iname = self.files[idx][1]  # current point cloud frame

        # Read point cloud and remove reflectance
        fname = self._get_velodyne_fn(drive, iname)

        # XYZ and reflectance
        xyzr = np.fromfile(fname, dtype=np.float32).reshape(-1, 4)
        xyz = xyzr[:, :3]

        # downsample
        xyz = to_o3d_pcd(xyz)
        xyz = xyz.voxel_down_sample(self.voxel_size)
        xyz = np.asarray(xyz.points)

        # Sample 8192 points randomly
        xyz_idx = np.random.choice(xyz.shape[0], size=8192, replace=False)
        return xyz[xyz_idx]

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        pts = pts @ R.T + T
        return pts

    @property
    def velo2cam(self):
        try:
            velo2cam = self._velo2cam
        except AttributeError:
            R = np.array(
                [
                    7.533745e-03,
                    -9.999714e-01,
                    -6.166020e-04,
                    1.480249e-02,
                    7.280733e-04,
                    -9.998902e-01,
                    9.998621e-01,
                    7.523790e-03,
                    1.480755e-02,
                ]
            ).reshape(3, 3)
            T = np.array([-4.069766e-03, -7.631618e-02, -2.717806e-01]).reshape(3, 1)
            velo2cam = np.hstack([R, T])
            self._velo2cam = np.vstack((velo2cam, [0, 0, 0, 1])).T
        return self._velo2cam

    def get_video_odometry(self, drive, indices=None, ext=".txt", return_all=False):
        if self.IS_ODOMETRY:
            data_path = self.root + "/poses/%02d.txt" % drive
            if data_path not in self.kitti_cache:
                self.kitti_cache[data_path] = np.genfromtxt(data_path)
            if return_all:
                return self.kitti_cache[data_path]
            else:
                return self.kitti_cache[data_path][indices]

    def odometry_to_positions(self, odometry):
        if self.IS_ODOMETRY:
            T_w_cam0 = odometry.reshape(3, 4)
            T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
            return T_w_cam0

    def _get_velodyne_fn(self, drive, t):
        if self.IS_ODOMETRY:
            fname = self.root + "/sequences/%02d/velodyne/%06d.bin" % (
                drive,
                t,
            )
        return fname

    def get_position_transform(self, pos0, pos1, invert=False):
        T0 = self.pos_transform(pos0)
        T1 = self.pos_transform(pos1)
        return np.dot(T1, np.linalg.inv(T0)).T if not invert else np.dot(np.linalg.inv(T1), T0).T
