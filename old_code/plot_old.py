"""
Plots number of inliers for various values of eps
"""
import numpy as np
import matplotlib.pyplot as plt

# load all point clouds with different val of eps
ovp_1 = np.load("/home/aditya/bcpcr/overlapping_pts_0.1_old.npy")
ovp_2 = np.load("/home/aditya/bcpcr/overlapping_pts_0.2_old.npy")
ovp_3 = np.load("/home/aditya/bcpcr/overlapping_pts_0.3_old.npy")
ovp_4 = np.load("/home/aditya/bcpcr/overlapping_pts_0.4_old.npy")
ovp_5 = np.load("/home/aditya/bcpcr/overlapping_pts_0.5_old.npy")
ovp_6 = np.load("/home/aditya/bcpcr/overlapping_pts_0.6_old.npy")
ovp_7 = np.load("/home/aditya/bcpcr/overlapping_pts_0.7_old.npy")
ovp_8 = np.load("/home/aditya/bcpcr/overlapping_pts_0.8_old.npy")
ovp_9 = np.load("/home/aditya/bcpcr/overlapping_pts_0.9_old.npy")
ovp_10 = np.load("/home/aditya/bcpcr/overlapping_pts_1.0_old.npy")

eps = np.linspace(1, 10, 10)

# compute number of points in overlap - # pts in src + # pts in tgt
ovp_1_uniq = len(np.unique(ovp_1[:, 1])) + len(np.unique(ovp_1[:, 2]))
ovp_2_uniq = len(np.unique(ovp_2[:, 1])) + len(np.unique(ovp_2[:, 2]))
ovp_3_uniq = len(np.unique(ovp_3[:, 1])) + len(np.unique(ovp_3[:, 2]))
ovp_4_uniq = len(np.unique(ovp_4[:, 1])) + len(np.unique(ovp_4[:, 2]))
ovp_5_uniq = len(np.unique(ovp_5[:, 1])) + len(np.unique(ovp_5[:, 2]))
ovp_6_uniq = len(np.unique(ovp_6[:, 1])) + len(np.unique(ovp_6[:, 2]))
ovp_7_uniq = len(np.unique(ovp_7[:, 1])) + len(np.unique(ovp_7[:, 2]))
ovp_8_uniq = len(np.unique(ovp_8[:, 1])) + len(np.unique(ovp_8[:, 2]))
ovp_9_uniq = len(np.unique(ovp_9[:, 1])) + len(np.unique(ovp_9[:, 2]))
ovp_10_uniq = len(np.unique(ovp_10[:, 1])) + len(np.unique(ovp_10[:, 2]))

npts_overlap = [
    ovp_1_uniq,
    ovp_2_uniq,
    ovp_3_uniq,
    ovp_4_uniq,
    ovp_5_uniq,
    ovp_6_uniq,
    ovp_7_uniq,
    ovp_8_uniq,
    ovp_9_uniq,
    ovp_10_uniq,
]

plt.plot(eps, npts_overlap)
plt.show()
