import json
import glob
import matplotlib.pyplot as plt
import numpy as np
import re

with np.load('camera_params/camera_param.npz') as X:
    mtx, Mat, tvecs = [X[i] for i in ('mtx', 'Mat', 'tvecs')]

print(mtx)
print(Mat)
print(tvecs)