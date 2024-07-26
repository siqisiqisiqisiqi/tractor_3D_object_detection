import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)
print(BASE_DIR)

from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from pointnet2.pointnet2_utils import furthest_point_sample

import torch
import torch.nn as nn



is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

def build_preencoder():
    mlp_dims = [0, 64, 128, 128]
    preencoder = PointnetSAModuleVotes(
        radius=0.2,
        nsample=64,
        npoint=256,
        mlp=mlp_dims,
        normalize_xyz=True,
    )
    return preencoder

precoder = build_preencoder().to(device)

# # non-square kernels and unequal stride and with padding
input = torch.randn(8, 1024, 3).to(device)
xyz, pre_enc_features, pre_enc_inds = precoder(input)
print(xyz.shape)
print(pre_enc_features.shape)