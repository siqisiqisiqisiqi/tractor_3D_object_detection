import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
PARENT_DIR = os.path.dirname(ROOT_DIR)
sys.path.append(BASE_DIR)
print(BASE_DIR)

from pointnet2.pointnet2_modules import PointnetSAModuleVotes
from pointnet2.pointnet2_utils import furthest_point_sample
from models.position_embedding import PositionEmbeddingCoordsSine
from models.helpers import GenericMLP

import torch
import torch.nn as nn


# is_cuda = torch.cuda.is_available()
# if is_cuda:
#     device = torch.device("cuda")
# else:
#     device = torch.device("cpu")


# def build_preencoder():
#     mlp_dims = [0, 64, 128, 128]
#     preencoder = PointnetSAModuleVotes(
#         radius=0.2,
#         nsample=64,
#         npoint=256,
#         mlp=mlp_dims,
#         normalize_xyz=True,
#     )
#     return preencoder


# precoder = build_preencoder().to(device)

# # # non-square kernels and unequal stride and with padding
# input = torch.randn(8, 1024, 3).to(device)
# xyz, pre_enc_features, pre_enc_inds = precoder(input)
# print(xyz.shape)
# print(pre_enc_features.shape)

# decoder_dim = 256
# position_embedding = "fourier"

# pos_embedding = PositionEmbeddingCoordsSine(
#     d_pos=decoder_dim, pos_type=position_embedding, normalize=True)

# query_projection = GenericMLP(
#     input_dim=decoder_dim,
#     hidden_dims=[decoder_dim],
#     output_dim=decoder_dim,
#     use_conv=True,
#     output_use_activation=True,
#     hidden_use_bias=True,
# )
# query_projection.to(device)


# def get_query_embeddings(encoder_xyz, point_cloud_dims, num_queries=64):
#     query_inds = furthest_point_sample(encoder_xyz, num_queries)
#     query_inds = query_inds.long()
#     query_xyz = [torch.gather(encoder_xyz[..., x], 1, query_inds)
#                  for x in range(3)]
#     query_xyz = torch.stack(query_xyz)
#     query_xyz = query_xyz.permute(1, 2, 0)

#     pos_embed = pos_embedding(query_xyz, input_range=point_cloud_dims)
#     query_embed = query_projection(pos_embed)
#     return query_xyz, query_embed


# pc_input = torch.randn(8, 1024, 3).to(device)
# # pc_input = torch.randn(8, 1024, 3)
# point_cloud_dims_min = torch.min(pc_input, 1).values
# point_cloud_dims_max = torch.max(pc_input, 1).values
# point_cloud_dims = torch.stack(
#     (point_cloud_dims_min, point_cloud_dims_max)).to(device)
# # print(point_cloud_dims.shape)
# query_xyz, query_embed = get_query_embeddings(pc_input, point_cloud_dims)
# print(query_embed.shape)

a = torch.randn(8,3, 1024)
b = nn.Linear(1024, 64)
c = b(a)
print(c.shape)
