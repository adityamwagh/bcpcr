import torch
import open3d as o3d
import copy


def get_rot_trans(H):

    R = torch.zeros((H.shape[0], 3, 3))
    t = torch.zeros((H.shape[0], 3, 1))

    R = H[:, 0:3, 0:3]
    t = H[:, 0:3, 3]

    return R, t


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.pipelines.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )
