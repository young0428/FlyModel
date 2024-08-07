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

model_string = "p3d"
model_string += f"{frame_per_window}frames"

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
    print(f"Fold {fold+1}")
    for epoch in range(current_epoch, epochs):
        training_tuples = batch_tuples[train_index]
        val_tuples = batch_tuples[val_index]
        
        batches = list(get_batches(training_tuples, batch_size))
        print("")
        print(f"Epoch {epoch + 1}/{epochs}")

        progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
        
        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_sac(
                video_data, sac_data, batch, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32)
            
            loss, pred = trainer.step(batch_input_data, batch_target_data)
            
            recent_losses.append(loss.item())

            avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            progress_bar.set_postfix(loss=f"{loss.item():.5f}", avg_recent_loss=f"{avg_recent_loss:.5f}")
            
            del batch_input_data, batch_target_data, loss, pred
        
        trainer.save(f"{model_name}/{checkpoint_name}.ckpt", epoch)
        
        # validation phase after each epoch
        val_batches = list(get_batches(val_tuples, batch_size))
        total_test_loss = 0.0
        progress_bar = tqdm(val_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=150)

        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_v2(
                video_data, sac_data, batch, frame_per_window, fps)
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32)
            
            loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
            total_test_loss = total_test_loss + loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
            
            del batch_input_data, batch_target_data, loss, pred

        avg_test_loss = total_test_loss / len(val_batches)
        test_losses_per_epoch.append(avg_test_loss)
        print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Save the test losses to a file
with open(f"{model_name}/test_losses_per_epoch.pkl", 'wb') as f:
    pickle.dump(test_losses_per_epoch, f)



#### result plotting phase ####
# Plot and save the test losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(test_losses_per_epoch) + 1), test_losses_per_epoch, marker='o', color='b', label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss per Epoch')
plt.legend()
plt.grid(True)
plt.savefig(f"{model_name}/test_loss_per_epoch.png")
plt.close()

print("Training completed.")

# Prediction phase after training
print("Starting prediction phase...")
prediction_path = f"{model_name}"
os.makedirs(prediction_path, exist_ok=True)

fig, axes = plt.subplots(3, 1, figsize=(16, 16))  # Adjusted figure size to show all videos in one plot

final_test_loss = test_losses_per_epoch[-1] if test_losses_per_epoch else float('nan')
all_predictions = []
all_targets = []
for video_num in range(0):
    print(f"Processing video {video_num} for prediction...")
    predictions = predict_with_model(model, trainer, video_data[video_num:video_num+1], frame_per_window, fps)
    target = optic_power[video_num]

    # Save the predictions
    np.save(f"{prediction_path}/predictions_video_{video_num}.npy", predictions)
    np.save(f"{prediction_path}/target_video_{video_num}.npy", target)

    # Plot results
    all_predictions.append(predictions)
    all_targets.append(target)
    plot_results(axes[video_num], video_num, predictions, target, training_tuples, val_tuples, total_frame, fps, frame_per_window, result_patience, final_test_loss)

save_results(result_save_path, all_targets, all_predictions)

plt.tight_layout()
plt.savefig(f"{prediction_path}/predictions_all_videos.png")
plt.close()

print("Prediction phase completed.")
