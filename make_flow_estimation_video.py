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

model_string = "64x128_opticflow_64t512"
model_string += f"{frame_per_window}frames_"

folder_path = "./naturalistic"
mat_file_name = "experimental_data.mat"
checkpoint_name = "fly_model"

model_name = f"./model/{model_string}"
os.makedirs(model_name, exist_ok=True)
result_save_path = f"./model/{model_string}/result_data.h5"

# hyperparameter 
batch_size = 50
lr = 1e-3
epochs = 100
fold_factor = 8

layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]

video_data, total_frame = load_video_data(folder_path, downsampling_factor)


# Split period and split for training / test data set
recent_losses = deque(maxlen=100)
test_losses_per_epoch = []

batch_tuples = np.array(generate_tuples_flow(total_frame, frame_per_window, frame_per_sliding, video_data.shape[0]))
kf = KFold(n_splits=fold_factor, shuffle=True, random_state=1)

all_fold_losses = []
model = flownet3d(layer_configs, num_classes=2)
trainer = Trainer(model, loss_function, lr)
#%%
test_tuples = []
#for i in range(3):
test_tuples.extend([(0, frame_num) for frame_num in range(frame_per_window, total_frame, 1)])
#%%
checkpoint_name = '64x128_opticflow_64t51216frames'
fold = 0
#for fold in range(5):
fold_path = f"./best_model/{checkpoint_name}/fold{fold+1}"
_ = trainer.load(f"{fold_path}/best_model.ckpt")
video = []
#%%
for idx, t in enumerate(test_tuples):
    print(f"{idx} / {len(test_tuples)}")
    batch = [t]
    input_data, target_data = get_data_from_batch_flow_estimate(video_data, batch, frame_per_window)
    batch_input_data = torch.tensor(input_data, dtype=torch.float32).to(trainer.device)
    prediction = trainer.pred(batch_input_data)
    video.append(prediction[0,0])
#%%
    
    
    
    
