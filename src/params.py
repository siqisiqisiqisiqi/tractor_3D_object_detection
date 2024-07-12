import numpy as np


############################################################################
BASE_LR = 0.001
WEIGHT_DECAY = 0.0000
LR_STEPS = 10
GAMMA = 0.3
BATCH_SIZE = 32
MAX_EPOCH = 51
MIN_LR = 1e-5

############################################################################
NUM_HEADING_BIN = 12
NUM_SIZE_CLUSTER = 3  # one cluster for each type
NUM_OBJECT_POINT = 1024

###########################################################################
g_type2class = {'road cone': 0, 'box': 1, 'human': 2}
g_class2type = {g_type2class[t]: t for t in g_type2class}
g_type2onehotclass = {'road cone': 0, 'box': 1, 'human': 2}
g_type_mean_size = {'road cone': np.array([0.25, 0.25, 0.4]),
                    'box': np.array([0.6, 0.4, 0.5]),
                    'human': np.array([0.5, 0.5, 1.8])}  # uniot in meter
g_mean_size_arr = np.zeros((NUM_SIZE_CLUSTER, 3))  # size clustrs
for i in range(NUM_SIZE_CLUSTER):
    g_mean_size_arr[i, :] = g_type_mean_size[g_class2type[i]]
