import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.params import *


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

    heading_residual = heading_residual.repeat(1, 4)
    heading_residual_normalized = heading_residual / (np.pi / NUM_HEADING_BIN)
    # heading_scores = heading_class_label

    heading_scores = torch.eye(NUM_HEADING_BIN).cuda()[
        heading_class_label.squeeze().long()]

    scls_onehot = torch.eye(NUM_SIZE_CLUSTER).cuda()[
        size_class_label.squeeze().long()]
    scls_onehot_repeat = scls_onehot.unsqueeze(2).repeat(1, 1, 3)
    size_residual_label = size_residual_label.unsqueeze(1)
    size_residual = size_residual_label * scls_onehot_repeat

    size_residual_normalized = size_residual / \
        torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()
    return center_boxnet, heading_residual_normalized, heading_residual, \
        size_residual_normalized, size_residual, heading_scores


def parse_output_to_tensors(box_pred, one_hot):
    '''
    :param box_pred: (bs,59)
    :param logits: (bs,1024,2)
    :param mask: (bs,1024)
    :param stage1_center: (bs,3)
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
    heading_scores = box_pred[:, c:c + NUM_HEADING_BIN]  # 3:3+12
    c += NUM_HEADING_BIN
    heading_residual_normalized = \
        box_pred[:, c:c + NUM_HEADING_BIN]  # 3+12 : 3+2*12
    heading_residual = \
        heading_residual_normalized * (np.pi / NUM_HEADING_BIN)
    c += NUM_HEADING_BIN

    # [32,24] 3+2*12+8 : 3+2*12+4*8
    size_residual_normalized = box_pred[:, c:c + 3 * 1].contiguous()
    size_residual_normalized = size_residual_normalized.view(
        bs, 1, 3)  # [32,8,3]
    one_hot_array = one_hot.unsqueeze(2).repeat(1, 1, 3)
    size_residual_normalized = size_residual_normalized * one_hot_array
    size_residual = size_residual_normalized * \
        torch.from_numpy(g_mean_size_arr).unsqueeze(0).repeat(bs, 1, 1).cuda()
    return center_boxnet, \
        heading_scores, heading_residual_normalized, heading_residual, \
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


class PointNetLoss(nn.Module):
    def __init__(self):
        super(PointNetLoss, self).__init__()

    def forward(self, center, center_label, stage1_center,
                heading_scores, heading_residual_normalized, heading_residual,
                heading_class_label, heading_residual_label,
                size_residual_normalized, size_residual,
                size_class_label, size_residual_label,
                corner_loss_weight=1, box_loss_weight=10):
        '''
        1.InsSeg
        logits: torch.Size([32, 1024, 2]) torch.float32
        mask_label: [32, 1024]
        2.Center
        center: torch.Size([32, 3]) torch.float32
        stage1_center: torch.Size([32, 3]) torch.float32
        center_label:[32,3]
        3.Heading
        heading_scores: torch.Size([32, 12]) torch.float32
        heading_residual_snormalized: torch.Size([32, 12]) torch.float32
        heading_residual: torch.Size([32, 12]) torch.float32
        heading_class_label:(32)
        heading_residual_label:(32)
        4.Size
        size_scores: torch.Size([32, 8]) torch.float32
        size_residual_normalized: torch.Size([32, 8, 3]) torch.float32
        size_residual: torch.Size([32, 8, 3]) torch.float32
        size_class_label:(32)
        size_residual_label:(32,3)
        5.Corner
        6.Weight
        corner_loss_weight: float scalar
        box_loss_weight: float scalar

        '''
        bs = center.shape[0]

        # Center Regression Loss
        center_dist = torch.norm(center - center_label, dim=1)  # (32,)
        center_loss = huber_loss(center_dist, delta=2.0)

        stage1_center_dist = torch.norm(
            stage1_center - center_label, dim=1)  # (32,)
        stage1_center_loss = huber_loss(stage1_center_dist, delta=1.0)

        # Heading Loss
        heading_class_loss_v1 = F.nll_loss(F.log_softmax(heading_scores, dim=1),
                                           heading_class_label.squeeze(-1).long(), reduction='none')  # tensor(2.4505, grad_fn=<NllLossBackward>)

        heading_mask = size_class_label == 0
        heading_mask = torch.where(
            heading_mask, torch.tensor(1), torch.tensor(0))
        heading_class_label_flip = heading_class_label + 2 * heading_mask
        mask = heading_class_label_flip > 3
        heading_class_label_flip = heading_class_label_flip - 4 * mask.int()
        heading_class_loss_v2 = F.nll_loss(F.log_softmax(heading_scores, dim=1),
                                           heading_class_label_flip.squeeze(-1).long(), reduction='none')
        heading_class_loss = torch.min(
            heading_class_loss_v1, heading_class_loss_v2)
        heading_class_loss = torch.mean(heading_class_loss)

        hcls_onehot = torch.eye(NUM_HEADING_BIN).cuda()[
            heading_class_label.squeeze(-1).long()]  # 32,12
        heading_residual_normalized_label = \
            heading_residual_label / (np.pi / NUM_HEADING_BIN)  # 32,
        heading_residual_normalized_dist = torch.sum(
            heading_residual_normalized * hcls_onehot.float(), dim=1, keepdim=True)  # 32,
        # Only compute reg loss on gt label
        heading_residual_normalized_loss = \
            huber_loss(heading_residual_normalized_dist -
                       heading_residual_normalized_label, delta=1.0)  # fix,2020.1.14
        # # Size loss
        # size_class_loss = F.nll_loss(F.log_softmax(size_scores, dim=1),
        #                              size_class_label.squeeze().long())  # tensor(2.0240, grad_fn=<NllLossBackward>)

        scls_onehot = torch.eye(NUM_SIZE_CLUSTER).cuda()[
            size_class_label.squeeze(-1).long()]  # 32,8
        # 32,8,3
        scls_onehot_repeat = scls_onehot.view(-1,
                                              NUM_SIZE_CLUSTER, 1).repeat(1, 1, 3)

        predicted_size_residual_normalized_dist = torch.sum(
            size_residual_normalized * scls_onehot_repeat.cuda(), dim=1)  # 32,3
        mean_size_arr_expand = torch.from_numpy(g_mean_size_arr).float().cuda() \
            .view(1, NUM_SIZE_CLUSTER, 3)  # 1,8,3
        mean_size_label = torch.sum(
            scls_onehot_repeat * mean_size_arr_expand, dim=1)  # 32,3
        size_residual_label_normalized = size_residual_label / mean_size_label

        size_normalized_dist = torch.norm(size_residual_label_normalized -
                                          predicted_size_residual_normalized_dist, dim=1)  # 32

        size_residual_normalized_loss = huber_loss(
            size_normalized_dist, delta=1.0)

        predicted_size_residual = predicted_size_residual_normalized_dist * mean_size_label
        size_residual_dist = torch.norm(size_residual_label -
                                        predicted_size_residual, dim=1)
        size_residual_loss = huber_loss(size_residual_dist, delta=1.0)

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
        size_label = mean_sizes + \
            size_residual_label.view(bs, 1, 3)  # (1,NS,3)+(bs,1,3)=(bs,NS,3)
        size_label = torch.sum(
            scls_onehot.view(bs, NUM_SIZE_CLUSTER, 1).float() * size_label, axis=[1])  # (B,3)

        corners_3d_gt = get_box3d_corners_helper(
            center_label, heading_label, size_label)  # (B,8,3)
        corners_3d_gt_flip = get_box3d_corners_helper(
            center_label, heading_label + np.pi, size_label)  # (B,8,3)

        corners_dist = torch.min(torch.norm(corners_3d_pred - corners_3d_gt, dim=-1),
                                 torch.norm(corners_3d_pred - corners_3d_gt_flip, dim=-1))
        corners_loss = huber_loss(corners_dist, delta=1.0)

        # Weighted sum of all losses
        # total_loss = box_loss_weight * (center_loss +
        #                                 heading_class_loss*0.1 + size_class_loss +
        #                                 heading_residual_normalized_loss * 20 +
        #                                 size_residual_normalized_loss * 20 +
        #                                 stage1_center_loss +
        #                                 corner_loss_weight * corners_loss)

        total_loss = box_loss_weight * (center_loss +
                                        heading_class_loss * 0.1 +
                                        heading_residual_normalized_loss +
                                        size_residual_loss +
                                        stage1_center_loss +
                                        corner_loss_weight * corners_loss)

        losses = {
            'total_loss': total_loss,
            'center_loss': box_loss_weight * center_loss,
            'heading_class_loss': box_loss_weight * heading_class_loss,
            'heading_residual_normalized_loss': box_loss_weight * heading_residual_normalized_loss,
            'size_residual_normalized_loss': box_loss_weight * size_residual_loss,
            'stage1_center_loss': box_loss_weight * stage1_center_loss,
            'corners_loss': box_loss_weight * corners_loss * corner_loss_weight,
        }
        return losses
