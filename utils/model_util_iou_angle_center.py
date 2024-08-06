import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
IOU_DIR = os.path.join(PARENT_DIR, "Rotated_IoU")
sys.path.append(IOU_DIR)


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.params import *

from oriented_iou_loss import cal_diou, cal_giou  # type: ignore


def angle_thres(angle):
    while angle < 0:
        angle = angle + 2 * np.pi
    while angle > np.pi:
        angle = angle - np.pi
    return angle


def label_to_tensors(label_dicts):
    center_boxnet = label_dicts.get('box3d_center')
    bs = center_boxnet.shape[0]
    size_class_label = label_dicts.get('size_class')
    size_residual_label = label_dicts.get('size_residual')
    heading_class_label = label_dicts.get('angle_class')
    heading_residual = label_dicts.get('angle_residual')

    heading_residual_normalized = heading_residual / np.pi

    scls_onehot = torch.eye(NUM_SIZE_CLUSTER).cuda()[
        size_class_label.squeeze().long()]
    scls_onehot_repeat = scls_onehot.unsqueeze(2).repeat(1, 1, 3)
    size_residual_label = size_residual_label.unsqueeze(1)
    size_residual = size_residual_label * scls_onehot_repeat

    size_residual_normalized = size_residual / \
        torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()
    return center_boxnet, heading_residual_normalized, heading_residual, \
        size_residual_normalized, size_residual


def parse_output_to_tensors(box_pred, one_hot):
    '''
    :param box_pred: (bs,59)
    :return:
        center_boxnet:(bs,3)
        heading_scores:(bs,12)
        heading_residual_normalized:(bs,12),-1 to 1
        heading_residual:(bs,12)
        size_scores:(bs,8)
        size_residual_normalized:(bs,8)
        size_residual:(bs,8)
    '''
    bs = box_pred.shape[0]
    # center
    center_boxnet = box_pred[:, :3]  # 0:3
    c = 3

    # heading
    heading_residual_normalized = box_pred[:, c:c + 1]  # 3+12 : 3+2*12
    heading_residual = heading_residual_normalized * (np.pi)
    c += 1

    size_residual_normalized = box_pred[:, c:c + 3 * 1].contiguous()
    size_residual_normalized = size_residual_normalized.view(
        bs, 1, 3)  # [32,8,3]
    one_hot_array = one_hot.unsqueeze(2).repeat(1, 1, 3)
    size_residual_normalized = size_residual_normalized * one_hot_array
    size_residual = size_residual_normalized * \
        torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()
    return center_boxnet, heading_residual_normalized, heading_residual, \
        size_residual_normalized, size_residual


def huber_loss(error, delta=1.0):  # (32,), ()
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)


def get_box3d_corners_helper(centers, headings, sizes):
    """ Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    # print '-----', centers
    N = centers.shape[0]
    l = sizes[:, 0].view(N, 1)
    w = sizes[:, 1].view(N, 1)
    h = sizes[:, 2].view(N, 1)
    # print l,w,h
    x_corners = torch.cat([l / 2, l / 2, -l / 2, -l / 2,
                          l / 2, l / 2, -l / 2, -l / 2], dim=1)  # (N,8)
    z_corners = torch.cat(
        [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2], dim=1)  # (N,8)
    y_corners = torch.cat([w / 2, -w / 2, -w / 2, w / 2,
                          w / 2, -w / 2, -w / 2, w / 2], dim=1)  # (N,8)
    corners = torch.cat([x_corners.view(N, 1, 8), y_corners.view(N, 1, 8),
                         z_corners.view(N, 1, 8)], dim=1)  # (N,3,8)

    c = torch.cos(headings).cuda()
    s = torch.sin(headings).cuda()
    ones = torch.ones([N], dtype=torch.float32).cuda()
    zeros = torch.zeros([N], dtype=torch.float32).cuda()
    row1 = torch.stack([c, -s, zeros], dim=1)  # (N,3)
    row2 = torch.stack([s, c, zeros], dim=1)
    row3 = torch.stack([zeros, zeros, ones], dim=1)
    R = torch.cat([row1.view(N, 1, 3), row2.view(N, 1, 3),
                   row3.view(N, 1, 3)], axis=1)  # (N,3,3)
    # print row1, row2, row3, R, N
    corners_3d = torch.bmm(R, corners)  # (N,3,8)
    corners_3d += centers.view(N, 3, 1).repeat(1, 1, 8)  # (N,3,8)
    corners_3d = torch.transpose(corners_3d, 1, 2)  # (N,8,3)
    return corners_3d


def get_box3d_corners(center, heading_residual, size_residual):
    """
    Inputs:
        center: (bs,3)
        heading_residual: (bs,NH)
        size_residual: (bs,NS,3)
    Outputs:
        box3d_corners: (bs,NH,NS,8,3) tensor
    """
    bs = center.shape[0]
    heading_bin_centers = torch.from_numpy(
        np.arange(0, np.pi, np.pi / NUM_HEADING_BIN)).float()  # (12,) (NH,)
    headings = heading_residual + \
        heading_bin_centers.view(1, -1).cuda()  # (bs,12)

    sizes = torch.from_numpy(g_mean_size_arr).float().view(1, NUM_SIZE_CLUSTER, 3).cuda()\
        + size_residual.cuda()  # (1,8,3)+(bs,8,3) = (bs,8,3)
    # sizes = mean_sizes + size_residual  # (bs,8,3)
    sizes = sizes.view(bs, 1, NUM_SIZE_CLUSTER, 3)\
        .repeat(1, NUM_HEADING_BIN, 1, 1).float()  # (B,12,8,3)
    headings = headings.view(bs, NUM_HEADING_BIN, 1).repeat(
        1, 1, NUM_SIZE_CLUSTER)  # (bs,12,8)
    centers = center.view(bs, 1, 1, 3).repeat(
        1, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1)  # (bs,12,8,3)
    N = bs * NUM_HEADING_BIN * NUM_SIZE_CLUSTER
    corners_3d = get_box3d_corners_helper(centers.view(N, 3), headings.view(N),
                                          sizes.view(N, 3))

    return corners_3d.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 8, 3)


g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]


class TransformerLoss(nn.Module):
    def __init__(self, delta_norm_loss_weight=0.01, center_loss_weight=0.5):
        super(TransformerLoss, self).__init__()
        self.delta_norm_loss_weight = delta_norm_loss_weight
        self.center_loss_weight = center_loss_weight

    def forward(self, mask_xyz_mean, point_cloud, x_delta, center_label,
                size_class_label, size_residual_label,
                heading_class_label, heading_residual_label):
        bs = mask_xyz_mean.shape[0]

        # Center Regression Loss
        center_dist = torch.norm(mask_xyz_mean - center_label, dim=1)
        center_loss = self.center_loss_weight * \
            huber_loss(center_dist, delta=1.0)

        # Calculate the groundtruth corners in xy plane
        hcls_onehot = torch.eye(NUM_HEADING_BIN).cuda()[
            heading_class_label.squeeze().long()]
        scls_onehot = torch.eye(NUM_SIZE_CLUSTER).cuda()[
            size_class_label.squeeze().long()]
        mean_sizes = torch.from_numpy(g_mean_size_arr)\
            .float().view(1, NUM_SIZE_CLUSTER, 3).cuda()  # (1,NS,3)
        size_label = mean_sizes + size_residual_label.view(bs, 1, 3)
        size_label = torch.sum(
            scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])

        heading_bin_centers = torch.from_numpy(
            np.arange(0, np.pi, np.pi / NUM_HEADING_BIN)).float().cuda()
        heading_label = heading_residual_label.view(bs, 1) + \
            heading_bin_centers.view(1, NUM_HEADING_BIN)
        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        corners_3d_gt = get_box3d_corners_helper(
            center_label, heading_label, size_label)
        corners_2d_gt = corners_3d_gt[:, 4:8, :2]

        v = center_label[:, :2]
        # horizantal vector
        center_2d_p_label = torch.stack([-v[:, 1], v[:, 0]], dim=1)
        p_norms = torch.norm(center_2d_p_label, dim=1, keepdim=True)
        center_2d_p_label = center_2d_p_label / p_norms
        center_2d_p_label = center_2d_p_label.unsqueeze(2)

        pp_value_label = (corners_2d_gt @ center_2d_p_label).squeeze()
        py_max_label = torch.max(pp_value_label, dim=1).values
        py_min_label = torch.min(pp_value_label, dim=1).values
        std_y_label = (py_max_label - py_min_label) / 4
        mean_y_label = (py_max_label + py_min_label) / 2

        # vertical case
        center_2d_label = v.unsqueeze(2)  # (bs,4,1)
        center_2d_norm = torch.norm(center_2d_label, dim=1)

        projection_value_label = (corners_2d_gt @ center_2d_label).squeeze()
        projection_value_label = projection_value_label / center_2d_norm
        p_max_label = torch.max(projection_value_label, dim=1).values
        p_min_label = torch.min(projection_value_label, dim=1).values
        std_label = (p_max_label - p_min_label) / 4
        mean_label = (p_max_label + p_min_label) / 2

        # pointcloud in xy plane
        pc_xy = point_cloud.permute(0, 2, 1)
        pc_xy = pc_xy[:, :, :2]

        # horizantal
        pp_value_pc = (pc_xy @ center_2d_p_label).squeeze()
        std_y_pc = torch.std(pp_value_pc, dim=1, keepdim=True)
        mean_y_pc = torch.mean(pp_value_pc, dim=1, keepdim=True)
        std_dist = torch.norm(std_y_label.unsqueeze(1) - std_y_pc, dim=1)
        y_std_loss = huber_loss(std_dist, delta=1.0)

        mean_dist = torch.norm(mean_y_label.unsqueeze(1) - mean_y_pc, dim=1)
        y_mean_loss = huber_loss(mean_dist, delta=1.0)

        # vertical
        projection_value_pc = (pc_xy @ center_2d_label).squeeze()
        projection_value_pc = projection_value_pc / center_2d_norm
        # p_max_pc = torch.max(projection_value_pc, dim=1).values
        # p_min_pc = torch.min(projection_value_pc, dim=1).values

        std_pc = torch.std(projection_value_pc, dim=1, keepdim=True)
        mean_pc = torch.mean(projection_value_pc, dim=1, keepdim=True)

        std_dist = torch.norm(std_label.unsqueeze(1) - std_pc, dim=1)
        x_std_loss = huber_loss(std_dist, delta=1.0)

        mean_dist = torch.norm(mean_label.unsqueeze(1) - mean_pc, dim=1)
        x_mean_loss = huber_loss(mean_dist, delta=1.0)

        delta_norm = torch.norm(x_delta, dim=[1, 2])
        delta_norm_loss = self.delta_norm_loss_weight * \
            huber_loss(delta_norm, delta=1.0)
        total_loss = center_loss + x_std_loss + x_mean_loss + \
            delta_norm_loss + y_mean_loss + y_std_loss
        return 0.4 * total_loss


class PointNetLoss(nn.Module):
    def __init__(self, hyper_parameter=(1, 1, 1)):
        super(PointNetLoss, self).__init__()
        self.p = hyper_parameter

    def forward(self, center, center_label, stage1_center,
                heading_residual_normalized, heading_residual,
                heading_class_label, heading_residual_label,
                size_residual_normalized, size_residual,
                size_class_label, size_residual_label, mask_xyz_mean, transformerloss,
                corner_loss_weight=0.2, box_loss_weight=10):
        """_summary_

        Parameters
        ----------
        center : array size [bs, 3]
            predicted box center
        center_label : array size [bs, 3]
            _description_
        stage1_center : array size [bs,3]
            predicted box center in the middle of the network
        heading_residual_normalized : array size [bs,1]
            heading residual normalized value
        heading_residual : array size [bs, 1]
            heading residual value
        heading_class_label : array size [bs, 1]
            heading class label
        heading_residual_label : array [bs, 1]
            heading residual label
        size_residual_normalized : array [bs, num_class, 3]
            size residual normalized value
        size_residual : array [bs, num_class, 3]
            size residual value
        size_class_label : array [bs, 1]
            size class label
        size_residual_label : array [16, 3]
            size residual label
        mask_xyz_mean : array [16, 3]
            box center after transformer 
        corner_loss_weight : int, optional
            _description_, by default 0
        box_loss_weight : int, optional
            _description_, by default 10

        Returns
        -------
        _type_
            _description_
        """

        bs = center.shape[0]

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(
            stage1_center - center_label, dim=1)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        mask_center_dist = torch.norm(mask_xyz_mean - center_label, dim=1)
        mask_center_loss = huber_loss(mask_center_dist, delta=1.0)

        # Size Loss
        hcls_onehot = torch.eye(NUM_HEADING_BIN).cuda()[
            heading_class_label.squeeze(-1).long()]

        scls_onehot = torch.eye(NUM_SIZE_CLUSTER).cuda()[
            size_class_label.squeeze(-1).long()]

        scls_onehot_repeat = scls_onehot.unsqueeze(2).repeat(1, 1, 3)

        predicted_size_residual_normalized_dist = torch.sum(
            size_residual_normalized, dim=1)
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda() \
            .view(1, NUM_SIZE_CLUSTER, 3)
        mean_size_label = torch.sum(
            scls_onehot_repeat * mean_size_arr_expand, dim=1)
        size_residual_label_normalized = size_residual_label / mean_size_label

        size_normalized_dist = torch.norm(size_residual_label_normalized -
                                          predicted_size_residual_normalized_dist, dim=1)
        size_residual_normalized_loss = huber_loss(
            size_normalized_dist, delta=1.0)

        predicted_size_residual = predicted_size_residual_normalized_dist * mean_size_label
        size_residual_dist = torch.norm(size_residual_label -
                                        predicted_size_residual, dim=1)
        size_residual_loss = huber_loss(size_residual_dist, delta=2.0)

        # Center normalized loss
        normalize_center_dist = torch.norm(
            (center - center_label) / mean_size_label, dim=1)
        nomalize_center_loss = huber_loss(normalize_center_dist, delta=2.0)

        # Heading Loss
        heading_residual_label_normalized = heading_residual_label / \
            (np.pi / NUM_HEADING_BIN)

        heading_mask = size_class_label == 0
        heading_mask = torch.where(
            heading_mask, torch.tensor(1), torch.tensor(0))
        heading_residual_normalized_v1 = heading_residual_normalized + heading_mask * 1 / 2
        mask = heading_residual_normalized_v1 > 1
        heading_residual_normalized_v1 = heading_residual_normalized_v1 - mask.int()

        heading_residual_normalized_dist_v1 = torch.norm(
            heading_residual_label_normalized - heading_residual_normalized_v1, dim=1)
        heading_residual_normalized_dist_v0 = torch.norm(
            heading_residual_label_normalized - heading_residual_normalized, dim=1)
        heading_residual_normalized_dist = torch.min(heading_residual_normalized_dist_v0,
                                                     heading_residual_normalized_dist_v1)
        heading_loss = huber_loss(heading_residual_normalized_dist, delta=1.0)

        # Corner Loss
        corners_3d = get_box3d_corners(center,
                                       heading_residual, size_residual).cuda()  # (bs,NH,NS,8,3)(32, 12, 8, 8, 3)
        gt_mask = hcls_onehot.view(bs, NUM_HEADING_BIN, 1).repeat(1, 1, NUM_SIZE_CLUSTER) * \
            scls_onehot.view(bs, 1, NUM_SIZE_CLUSTER).repeat(
                1, NUM_HEADING_BIN, 1)  # (bs,NH=12,NS=8)
        corners_3d_pred = torch.sum(
            gt_mask.view(bs, NUM_HEADING_BIN, NUM_SIZE_CLUSTER, 1, 1)
            .float() * corners_3d,
            dim=[1, 2])  # (bs,8,3)
        heading_bin_centers = torch.from_numpy(
            np.arange(0, np.pi, np.pi / NUM_HEADING_BIN)).float().cuda()  # (NH,)
        heading_label = heading_residual_label.view(bs, 1) + \
            heading_bin_centers.view(
                1, NUM_HEADING_BIN)  # (bs,1)+(1,NH)=(bs,NH)

        heading_label = torch.sum(hcls_onehot.float() * heading_label, 1)
        mean_sizes = torch.from_numpy(g_mean_size_arr)\
            .float().view(1, NUM_SIZE_CLUSTER, 3).cuda()  # (1,NS,3)
        size_label = mean_sizes + size_residual_label.view(bs, 1, 3)
        size_label = torch.sum(
            scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])  # (B,3)

        corners_3d_gt = get_box3d_corners_helper(
            center_label, heading_label, size_label)  # (B,8,3)
        corners_3d_gt_flip = get_box3d_corners_helper(
            center_label, heading_label + np.pi, size_label)  # (B,8,3)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # iou angle loss
        loc_size_label = torch.cat(
            (center_label[:, :2], size_label[:, :2]), dim=-1)
        loc_size_angle_label = torch.cat(
            (loc_size_label, heading_residual_label), dim=-1)
        loc_size_angle_label = loc_size_angle_label.unsqueeze(1)
        loc_size_angle = torch.cat(
            (loc_size_label, heading_residual), dim=-1)
        loc_size_angle = loc_size_angle.unsqueeze(1)
        iou_loss, _ = cal_diou(loc_size_angle_label, loc_size_angle)
        iou_value_loss = 0 * huber_loss(iou_loss.squeeze())

        total_loss = box_loss_weight * (0 * center_loss +
                                        self.p[0] * 1 / 4 * nomalize_center_loss +
                                        iou_value_loss +
                                        # size_residual_normalized_loss +
                                        self.p[2] * heading_loss +
                                        self.p[1] * size_residual_loss +
                                        stage1_center_loss +
                                        transformerloss +
                                        corner_loss_weight * corners_loss)

        losses = {
            'total_loss': total_loss,
            'center_loss': box_loss_weight * 1 / 4 * nomalize_center_loss,
            'iou_value_loss': box_loss_weight * iou_value_loss,
            'size_residual_normalized_loss': box_loss_weight * size_residual_loss,
            'stage1_center_loss': box_loss_weight * stage1_center_loss,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
            'mask_center_loss': box_loss_weight * mask_center_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_loss,
            'transformer_loss': box_loss_weight * transformerloss
        }
        return losses
