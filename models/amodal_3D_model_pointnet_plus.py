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

from utils.model_util_iou_angle_center import PointNetLoss, parse_output_to_tensors, label_to_tensors, TransformerLoss
from utils.point_cloud_process import point_cloud_process
from utils.compute_box3d_iou import compute_box3d_iou, calculate_corner, compute_iou_class
from utils.stereo_custom_dataset import StereoCustomDataset
from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from pointnet2.pointnet2_utils import furthest_point_sample
from models.position_embedding import PositionEmbeddingCoordsSine
from models.helpers import GenericMLP
from src.params import *


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

        pointcloud = pts.permute(0, 2, 1)
        new_xyz, pre_enc_features, _ = self.preencoder(pointcloud)
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
    def __init__(self,
                 num_token: int = 256,
                 dim_model: int = 128,
                 num_heads: int = 2,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 position_embedding: str = "fourier",
                 dropout_p: float = 0.2):
        super().__init__()
        preencoder_mpl_dims = [0, 64, 128, dim_model]
        self.num_token = num_token
        self.preencoder = PointnetSAModuleVotes(
            radius=0.2,
            nsample=64,
            npoint=num_token,
            mlp=preencoder_mpl_dims,
            normalize_xyz=True,
        )
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
            batch_first=True
        )
        self.query_projection = GenericMLP(
            input_dim=dim_model,
            hidden_dims=[dim_model],
            output_dim=dim_model,
            use_conv=True,
            output_use_activation=True,
            hidden_use_bias=True,
        )

        self.pos_embedding = PositionEmbeddingCoordsSine(
            d_pos=dim_model, pos_type=position_embedding, normalize=True
        )

        self.fc1 = nn.Linear(dim_model + 3, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 3)

        self.fcbn1 = nn.BatchNorm1d(64)
        self.fcbn2 = nn.BatchNorm1d(16)

    def get_query_embeddings(self, encoder_xyz, point_cloud_dims, num_queries=256):
        query_inds = furthest_point_sample(encoder_xyz, num_queries)
        query_inds = query_inds.long()
        query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds)
                     for x in range(3)]
        query_xyz = torch.stack(query_xyz)
        query_xyz = query_xyz.permute(1, 2, 0)

        pos_embed = self.pos_embedding(query_xyz, input_range=point_cloud_dims)
        query_embed = self.query_projection(pos_embed)
        return query_xyz, query_embed

    def forward(self, pointcloud, one_hot_vec: tensor):
        bs = pointcloud.shape[0]
        pc_max = torch.tensor(PC_MAX).repeat(bs, 1)
        pc_min = torch.tensor(PC_MIN).repeat(bs, 1)
        point_cloud_dims = torch.stack((pc_min, pc_max)).cuda()

        xyz, pre_enc_features, pre_enc_inds = self.preencoder(pointcloud)
        src = pre_enc_features.permute(0, 2, 1)
        query_xyz, query_embed = self.get_query_embeddings(
            pointcloud, point_cloud_dims)
        tgt = query_embed.permute(0, 2, 1)
        transformer_out = self.transformer(src, tgt)

        one_hot_expand = one_hot_vec.unsqueeze(1).repeat(1, self.num_token, 1)
        out = torch.cat([transformer_out, one_hot_expand], 2)

        x = F.relu(self.fcbn1(self.fc1(out).permute(0, 2, 1)))
        x = x.permute(0, 2, 1)
        x = F.relu(self.fcbn2(self.fc2(x).permute(0, 2, 1)))
        x = x.permute(0, 2, 1)
        x = self.fc3(x)
        pc = query_xyz - x
        return pc, x


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
        self.transformer = TransformerBasedFilter()
        self.STN = STNxyz(n_classes=3)
        self.est = PointNetEstimationv2(n_classes=3)
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
        # point_cloud, x_delta = self.transformer(point_cloud, one_hot)

        point_cloud = point_cloud.permute(0, 2, 1)
        color = color.permute(0, 2, 1)
        # object_pts_xyz size (batchsize, 3, number object point)
        object_pts_xyz, mask_xyz_mean = point_cloud_process(point_cloud)

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
            #                                         x_delta, box3d_center_label,
            #                                         size_class_label, size_residual_label,
            #                                         heading_class_label, heading_residual_label)
            transformerloss = torch.tensor(0)

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
                label = size_class_label.detach().cpu().numpy().squeeze(-1)
            iou3d_class = compute_iou_class(iou3ds, label)
            # metrics = {
            #     'corners': corners,
            #     'iou2d': iou2ds.mean(),
            #     'iou3d': iou3ds.mean(),
            #     'iou3d_0.25': np.sum(iou3ds >= 0.25) / bs,
            #     'iou3d_0.5': np.sum(iou3ds >= 0.5) / bs,
            #     'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs,
            # }
            metrics = {
                'corners': corners,
                'iou2d': iou2ds.mean(),
                'iou3d': iou3ds.mean(),
                'iou3d_0.25': np.sum(iou3ds >= 0.25) / bs,
                'iou3d_0.5': np.sum(iou3ds >= 0.5) / bs,
                'iou3d_0.7': np.sum(iou3ds >= 0.7) / bs,
                'iou3d_roadcone_0.25': iou3d_class['iou3d_roadcone_0.25'],
                'iou3d_roadcone_0.5': iou3d_class['iou3d_roadcone_0.5'],
                'iou3d_box_0.25': iou3d_class['iou3d_box_0.25'],
                'iou3d_box_0.5': iou3d_class['iou3d_box_0.5'],
                'iou3d_human_0.25': iou3d_class['iou3d_human_0.25'],
                'iou3d_human_0.5': iou3d_class['iou3d_human_0.5'],
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
