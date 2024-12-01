import torch
import numpy as np
import matplotlib.pyplot as plt
from FlowNet import *
from trainer_func import Trainer
from LoadDataset import *
from tqdm import tqdm
from collections import deque
import pickle
import os
from sklearn.model_selection import KFold

torch.autograd.set_detect_anomaly(True)

h = 360
w = 720
c = 1
fps = 30
downsampling_factor = 5.625

frame_per_window = 16
frame_per_sliding = 16
input_ch = 1 

model_string = "64_to_256_3layers.ckpt"
#model_string += f"{frame_per_window}frames_"

folder_path = "./naturalistic"
mat_file_name = "experimental_data.mat"
checkpoint_name = "fly_model"

# hyperparameter 
batch_size = 50
lr = 1e-3
epochs = 100
fold_factor = 8

layer_configs = [[64, 2], [128, 2], [256, 2]]


    
    
