import sys
import os
from typing import Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(ROOT_DIR)

import ipdb
import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy import ndarray
from torch.nn import init
from torch.utils.data import DataLoader

from utils.model_util_transformer import PointNetLoss, parse_output_to_tensors, TransformerLoss, point_cloud_masking
from utils.point_cloud_process import point_cloud_process
from utils.compute_box3d_iou import compute_box3d_iou, calculate_corner
from utils.stereo_custom_dataset import StereoCustomDataset
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from point_transformer.model import PointTransformerSeg
from src.params import *
import yaml


class PointNetEstimationv2(nn.Module):
    def __init__(self, n_classes: int = 3, conv_dim: list = [64, 128, 128, 256, 512]):
        """Model estimate the 3D bounding box

        Parameters
        ----------
        n_classes : int, optional
            Number of the object type, by default 1
        """
        # conv_dim = [64, 128, 256, 512, 1024]
        super(PointNetEstimationv2, self).__init__()
        preencoder_mpl_dims = [0, 64, 128, 256, 512]
        self.preencoder = PointnetSAModuleVotes(
            radius=0.2,
            nsample=32,
            npoint=256,
            mlp=preencoder_mpl_dims,
            normalize_xyz=True,
        )

        self.n_classes = n_classes

        self.class_fc = nn.Linear(n_classes, 64)
        self.dist_fc = nn.Linear(3, 64)
        self.fcbn_dist = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(conv_dim[4] + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3 + 1 + 1 * 3)  # center, angle, size
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)
        self.fcbn3 = nn.BatchNorm1d(64)
        self.dropout12 = nn.Dropout(0.2)
        self.dropout13 = nn.Dropout(0.2)

    def forward(self, pts: ndarray, one_hot_vec: ndarray, stage1_center: ndarray) -> tensor:
        """
        Parameters
        ----------
        pts : ndarray
            point cloud 
            size bsx3xnum_point
        one_hot_vec : ndarray
            one hot vector type 
            size bsxn_classes

        Returns
        -------
        tensor
            size 3x3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        """
        bs = pts.size()[0]

        pointcloud = pts.permute(0, 2, 1).contiguous()
        _, pre_enc_features, _ = self.preencoder(pointcloud)
        global_feat = torch.max(pre_enc_features, 2, keepdim=False)[
            0]  # bs,512

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        one_hot_embed = F.relu(self.class_fc(expand_one_hot_vec))
        center_embed = F.relu(self.fcbn_dist(self.dist_fc(stage1_center)))
        expand_global_feat = torch.cat(
            [global_feat, one_hot_embed, center_embed], 1)  # bs,518
        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))  # bs,512
        x = self.dropout12(F.relu(self.fcbn2(self.fc2(x))))  # bs,256
        x = self.dropout13(F.relu(self.fcbn3(self.fc3(x))))
        box_pred = self.fc4(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred


class PointNetEstimation(nn.Module):
    def __init__(self, n_classes: int = 3, conv_dim: list = [64, 128, 128, 256, 512]):
        """Model estimate the 3D bounding box

        Parameters
        ----------
        n_classes : int, optional
            Number of the object type, by default 1
        """
        # conv_dim = [64, 128, 256, 512, 1024]
        super(PointNetEstimation, self).__init__()
        self.conv1 = nn.Conv1d(3, conv_dim[0], 1)
        self.conv2 = nn.Conv1d(conv_dim[0], conv_dim[1], 1)
        self.conv3 = nn.Conv1d(conv_dim[1], conv_dim[2], 1)
        self.conv4 = nn.Conv1d(conv_dim[2], conv_dim[3], 1)
        self.conv5 = nn.Conv1d(conv_dim[3], conv_dim[4], 1)
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(conv_dim[0])
        self.bn2 = nn.BatchNorm1d(conv_dim[1])
        self.bn3 = nn.BatchNorm1d(conv_dim[2])
        self.bn4 = nn.BatchNorm1d(conv_dim[3])
        self.bn5 = nn.BatchNorm1d(conv_dim[4])

        self.n_classes = n_classes

        self.class_fc = nn.Linear(n_classes, 64)
        self.dist_fc = nn.Linear(3, 64)
        self.fcbn_dist = nn.BatchNorm1d(64)

        self.fc1 = nn.Linear(conv_dim[4] + 64 + 64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3 + 1 + 1 * 3)  # center, angle, size
        self.fcbn1 = nn.BatchNorm1d(512)
        self.fcbn2 = nn.BatchNorm1d(256)
        self.fcbn3 = nn.BatchNorm1d(64)
        self.dropout12 = nn.Dropout(0.2)
        self.dropout13 = nn.Dropout(0.2)

    def forward(self, pts: ndarray, one_hot_vec: ndarray, stage1_center: ndarray) -> tensor:
        """
        Parameters
        ----------
        pts : ndarray
            point cloud 
            size bsx3xnum_point
        one_hot_vec : ndarray
            one hot vector type 
            size bsxn_classes

        Returns
        -------
        tensor
            size 3x3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        """
        bs = pts.size()[0]
        n_pts = pts.size()[2]

        out1 = self.dropout(F.relu(self.bn1(self.conv1(pts))))  # bs,128,n
        out2 = self.dropout(F.relu(self.bn2(self.conv2(out1))))  # bs,128,n
        out3 = self.dropout(F.relu(self.bn3(self.conv3(out2))))  # bs,256,n
        out4 = self.dropout(F.relu(self.bn4(self.conv4(out3))))  # bs,512,n
        out5 = self.dropout(F.relu(self.bn5(self.conv5(out4))))  # bs,512,n
        global_feat = torch.max(out5, 2, keepdim=False)[0]  # bs,512

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3
        one_hot_embed = F.relu(self.class_fc(expand_one_hot_vec))
        center_embed = F.relu(self.fcbn_dist(self.dist_fc(stage1_center)))
        expand_global_feat = torch.cat(
            [global_feat, one_hot_embed, center_embed], 1)  # bs,518
        x = F.relu(self.fcbn1(self.fc1(expand_global_feat)))  # bs,512
        x = self.dropout12(F.relu(self.fcbn2(self.fc2(x))))  # bs,256
        x = self.dropout13(F.relu(self.fcbn3(self.fc3(x))))
        box_pred = self.fc4(x)  # bs,3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4
        return box_pred


class STNxyz(nn.Module):
    def __init__(self, n_classes: int = 3):
        """transformation network

        Parameters
        ----------
        n_classes : int, optional
            Number of the object type, by default 1
        """
        super(STNxyz, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        # self.conv4 = torch.nn.Conv1d(256, 512, 1)

        self.fc1 = nn.Linear(256 + n_classes, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

        init.zeros_(self.fc3.weight)
        init.zeros_(self.fc3.bias)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.fcbn1 = nn.BatchNorm1d(256)
        self.fcbn2 = nn.BatchNorm1d(128)

    def forward(self, pts: tensor, one_hot_vec: tensor) -> tensor:
        """transformation network forward

        Parameters
        ----------
        pts : tensor
            point cloud
            size [bs,3,num_point]
        one_hot_vec : tensor
            type of the object
            size [bs,3]

        Returns
        -------
        tensor
            Translation center
            size [bs,3]
        """
        bs = pts.shape[0]
        x = F.relu(self.bn1(self.conv1(pts)))  # bs,128,n
        x = F.relu(self.bn2(self.conv2(x)))  # bs,128,n
        x = F.relu(self.bn3(self.conv3(x)))  # bs,256,n
        x = torch.max(x, 2)[0]  # bs,256

        expand_one_hot_vec = one_hot_vec.view(bs, -1)  # bs,3

        x = torch.cat([x, expand_one_hot_vec], 1)
        x = F.relu(self.fcbn1(self.fc1(x)))  # bs,256
        x = F.relu(self.fcbn2(self.fc2(x)))  # bs,128
        x = self.fc3(x)  # bs,
        return x


class TransformerBasedFilter(nn.Module):
    def __init__(self, yaml_file):
        super().__init__()
        with open(yaml_file, 'rb') as f:
            conf = yaml.safe_load(f.read())

        from collections import namedtuple

        def convert_to_namedtuple(dictionary):
            """Converts a dictionary to a namedtuple."""
            return namedtuple('GenericDict', dictionary.keys())(**dictionary)
        args = convert_to_namedtuple(conf)

        self.pt_seg = PointTransformerSeg(args)

    def forward(self, pc, one_hot_vec: tensor, xyz_mean: tensor):
        """_summary_

        Parameters
        ----------
        pointcloud : tensor
            [bs, 3, num_point]
        one_hot_vec : tensor
            [bs, num_category]
        xyz_mean : tensor
            [bs, 3]
        """
        pointcloud = pc.permute(0, 2, 1)
        ns = pointcloud.shape[1]
        one_hot_expand = one_hot_vec.unsqueeze(1).repeat(1, ns, 1)
        xyz_expand = xyz_mean.unsqueeze(1).repeat(1, ns, 1)
        input_feature = torch.cat([pointcloud, one_hot_expand, xyz_expand], -1)
        output = self.pt_seg(input_feature)
        pc_delta = output.contiguous().permute(0, 2, 1)
        # pred_choice = seg_pred.data.max(-1)[1]
        # pc, xyz_mean2, _ = point_cloud_masking(pc, seg_pred)
        pc = pc + pc_delta
        _, xyz_mean2 = point_cloud_process(pc)
        return pc, xyz_mean + xyz_mean2, pc_delta


def huber_loss(error, delta=1.0):  # (32,), ()
    abs_error = torch.abs(error)
    quadratic = torch.clamp(abs_error, max=delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return torch.mean(losses)


class Amodal3DModel(nn.Module):
    def __init__(self, n_classes: int = 3, n_channel: int = 3, hyper_parameter=None):
        """amodal 3D estimation model 

        Parameters
        ----------
        n_classes : int, optional
            Number of classes, by default 1
        n_channel : int, optional
            Number of channel used in the point cloud, by default 3
        """
        super(Amodal3DModel, self).__init__()
        self.n_classes = n_classes
        self.n_channel = n_channel
        self.transformer = TransformerBasedFilter(
            "./config/PointTransformer.yaml")
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimationv2(n_classes=3)
        # self.est = PointNetEstimation(n_classes=3)
        if hyper_parameter is not None:
            self.Loss = PointNetLoss(hyper_parameter=hyper_parameter)
        else:
            self.Loss = PointNetLoss()
        self.transformer_loss = TransformerLoss()

    def forward(self, features: ndarray, one_hot: ndarray, label_dicts: dict = {}):
        """Amodal3DModel forward

        Parameters
        ----------
        features : ndarray
            object point cloud
            size [bs, num_point, 3]
        one_hot: ndarray
            object class
            size [bs, num_class]
        label_dicts : dict
            labeled result of the 3D bounding box

        Returns
        -------
        Tuple[dict, dict]
            losses: all the loss values stored in the dictionary
            metrics: iou and corner calculation
        """
        # point cloud after the instance segmentation
        bs = features.shape[0]  # batch size
        one_hot = one_hot.to(torch.float)

        features = features.contiguous()
        color = features[:, :, 3:6]
        point_cloud = features[:, :, :self.n_channel].contiguous()

        point_cloud = point_cloud.permute(0, 2, 1)
        color = color.permute(0, 2, 1)
        # object_pts_xyz size (batchsize, 3, number object point)
        object_pts_xyz, mask_xyz_mean = point_cloud_process(point_cloud)

        object_pts_xyz, mask_xyz_mean, pc_delta = self.transformer(
            object_pts_xyz, one_hot, mask_xyz_mean)

        # T-net
        center_delta = self.STN(object_pts_xyz, one_hot)  # (32,3)
        stage1_center = center_delta + mask_xyz_mean  # (32,3)

        if (np.isnan(stage1_center.cpu().detach().numpy()).any()):
            ipdb.set_trace()
        object_pts_xyz_new = object_pts_xyz - \
            center_delta.view(bs, -1, 1).repeat(1, 1, object_pts_xyz.shape[-1])

        # object_pts_xyz_new = torch.cat((object_pts_xyz_new, color), 1)
        # 3D Box Estimation
        box_pred = self.est(object_pts_xyz_new, one_hot, stage1_center)
        center_boxnet, heading_residual_normalized, heading_residual, \
            size_residual_normalized, size_residual = parse_output_to_tensors(
                box_pred, one_hot)

        box3d_center = stage1_center + center_boxnet  # bs,3
        heading_scores = torch.ones((bs, 1)).cuda()

        # center_boxnet, heading_residual_normalized, heading_residual, \
        #     size_residual_normalized, size_residual = label_to_tensors(
        #         label_dicts)
        # box3d_center = center_boxnet
        # stage1_center = center_boxnet

        if len(label_dicts) == 0:
            with torch.no_grad():
                corners = calculate_corner(box3d_center.detach().cpu().numpy(),
                                           heading_scores.detach().cpu().numpy(),
                                           heading_residual.detach().cpu().numpy(),
                                           one_hot.detach().cpu().numpy(),
                                           size_residual.detach().cpu().numpy())

            return corners

        else:
            box3d_center_label = label_dicts.get('box3d_center')
            size_class_label = label_dicts.get('size_class')
            size_residual_label = label_dicts.get('size_residual')
            heading_class_label = label_dicts.get('angle_class')
            heading_residual_label = label_dicts.get('angle_residual')

            # transformerloss = self.transformer_loss(mask_xyz_mean, point_cloud,
            #                                         pc_delta, box3d_center_label,
            #                                         size_class_label, size_residual_label,
            #                                         heading_class_label, heading_residual_label)

            delta_norm = torch.norm(pc_delta, dim=[1, 2])
            delta_norm_loss = 0.05 * huber_loss(delta_norm, delta=1.0)
            transformerloss = delta_norm_loss
            # transformerloss = torch.tensor(0)

            losses = self.Loss(box3d_center, box3d_center_label, stage1_center,
                               heading_residual_normalized,
                               heading_residual, heading_class_label,
                               heading_residual_label,
                               size_residual_normalized,
                               size_residual, size_class_label,
                               size_residual_label, mask_xyz_mean, transformerloss)

            with torch.no_grad():
                iou2ds, iou3ds, corners = compute_box3d_iou(
                    box3d_center.detach().cpu().numpy(),
                    heading_scores.detach().cpu().numpy(),
                    heading_residual.detach().cpu().numpy(),
                    one_hot.detach().cpu().numpy(),
                    size_residual.detach().cpu().numpy(),
                    box3d_center_label.detach().cpu().numpy(),
                    heading_class_label.detach().cpu().numpy().squeeze(-1),
                    heading_residual_label.detach().cpu().numpy().squeeze(-1),
                    size_class_label.detach().cpu().numpy().squeeze(-1),
                    size_residual_label.detach().cpu().numpy())
            metrics = {
                'corners': corners,
                'iou2d': iou2ds.mean(),
                'iou3d': iou3ds.mean(),
                'iou3d_0.25': np.sum(iou3ds >= 0.25) / bs,
                'iou3d_0.5': np.sum(iou3ds >= 0.5) / bs,
                'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs,
            }
            return losses, metrics


if __name__ == "__main__":
    pc_path = os.path.join(PARENT_DIR, "datasets", "pointclouds", "train")
    label_path = os.path.join(PARENT_DIR, "datasets", "labels", "train")

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = Amodal3DModel()
    model.to(device)

    dataset = StereoCustomDataset(pc_path, label_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=True)
    train_features, train_labels, img_dir = next(iter(train_dataloader))

    model = model.train()
    features = train_features.to(device, dtype=torch.float32)
    data_dicts_var = {key: value.to(device)
                      for key, value in train_labels.items()}
    one_hot = data_dicts_var.get('one_hot')
    losses, metrics = model(features, one_hot, data_dicts_var)
    print("This is a test!")
