from typing import Tuple
import numpy as np
from numpy import ndarray
import torch


def point_cloud_process(point_cloud: ndarray) -> Tuple[ndarray, ndarray]:
    """point cloud coordinate transform and center calculation

    Parameters
    ----------
    point_cloud : ndarray
        The point cloud in the world coordinate system
        size [batchsize, 3, point number]

    Returns
    -------
    Tuple[ndarray, ndarray]
        point_cloud_trans: transformed point cloud
        size [batchsize, point number, 3]
        xyz_mean: point cloud mean point
        size [batchsize, 3]
    """
    num_point = point_cloud.shape[2]
    xyz_sum = point_cloud.sum(2, keepdim=True)
    xyz_mean = xyz_sum / num_point
    point_cloud_trans = point_cloud - xyz_mean.repeat(1, 1, num_point)
    xyz_mean = xyz_mean.squeeze(2)
    return point_cloud_trans, xyz_mean


def point_cloud_normalization(pc):
    """_summary_

    Parameters
    ----------
    pc : tensor
        The point cloud in the object coordinate system without rotation
        [batchsize, 3, point number]

    Returns
    -------
    pc: tensor
        normalized point cloud
    """
    m = torch.max(torch.sqrt(torch.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
