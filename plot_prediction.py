import torch
import numpy as np
import matplotlib.pyplot as plt
from VCL import *
from LoadDataset import *
from tqdm import tqdm
from collections import deque
import pickle
import os

#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from p3d_resnet import p3d_resnet, loss_function
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

frame_per_window = 10
frame_per_sliding = 5
input_ch = 1 

model_string = "p3d_"
model_string += f"{frame_per_window}frames_"

folder_path = "./naturalistic"
mat_file_name = "saccade_prediction_data.mat"
checkpoint_name = "fly_model"

model_name = f"./model/{model_string}"
os.makedirs(model_name, exist_ok=True)
result_save_path = f"./model/{model_string}/result_data.h5"



# hyperparameter 
batch_size = 10
lr = 1e-3
epochs = 200
fold_factor = 8


#### p3d renet18 spec ####
block_list = [[64, 2], [128, 2], [256, 2], [512, 2]]
feature_output_dims = 400
##########################

# ### custom parameters ###
# block_list = [[]]
# feature_output_dims = 1024
# ##########################

# create model
model = p3d_resnet(input_ch, block_list, feature_output_dims)
trainer = Trainer(model, loss_function, lr)
current_epoch = trainer.load(f"{model_name}/{checkpoint_name}.ckpt")


video_data, sac_data, total_frame = sac_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
sac_data = sac_data[:,:total_frame]

# %%

# Split period and split for training / test data set
# and save to model folder
# Load same tuples for performance comparison

recent_losses = deque(maxlen=100)
test_losses_per_epoch = []

batch_tuples = np.array(generate_tuples_sac(total_frame, frame_per_window, frame_per_sliding, 3))
kf = KFold(n_splits=fold_factor)

for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):
    last_val_index = val_index


prediction_path = f"{model_name}"
os.makedirs(prediction_path, exist_ok=True)

fig, axes = plt.subplots(2, 1, figsize=(16, 16))  # Adjusted figure size to show all videos in one plot

all_predictions = []
all_targets = []

val_tuples = batch_tuples[val_index]
val_start_frame = val_tuples[0][1]
#%%

predictions = predict_with_model_sac(trainer, video_data, val_start_frame, frame_per_window, fps)
#%%
target = sac_data[0, val_start_frame:]
#%%

# Plot results
plt.legend()
axes[0].plot(range(val_start_frame, total_frame), target[:,0], label='Target', color='gray')
axes[0].plot(range(val_start_frame, total_frame), predictions[:,0], label='Prediction', color='black')
axes[0].legend()

axes[1].plot(range(val_start_frame, total_frame), target[:,1], label='Target', color='gray')
axes[1].plot(range(val_start_frame, total_frame), predictions[:,1], label='Prediction', color='black')
axes[1].legend()

plt.tight_layout()
plt.show()
plt.close()
print("Prediction phase completed.")

# %%
