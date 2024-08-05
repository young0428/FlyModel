import torch
import numpy as np
import matplotlib.pyplot as plt
from VCL import VCL, VCL_Trainer
from LoadDataset import *
from tqdm import tqdm
from collections import deque
import pickle
import os

def save_tuples(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_tuples(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def predict_with_model(model, trainer, video_data, frame_per_window, fps, result_patience, batch_size=10):
    total_frame = video_data.shape[1]
    step = int(10 * fps)  # 10 seconds window
    predictions = []

    print("Starting predictions...")


    video_data = torch.tensor(video_data, dtype=torch.float32).to(trainer.device)
    
    batch_predictions = []

    with torch.no_grad():
        for i in range(0, video_data.shape[1] - frame_per_window + 1, frame_per_window - result_patience):
            batch_input = video_data[:, i:i+frame_per_window]
            
            if batch_input.dim() == 6:
                batch_input = batch_input.squeeze(2)  # Remove the extra dimension
            
            pred = trainer.model(batch_input)
            batch_predictions.extend(pred.squeeze(0).cpu().numpy())

    predictions.extend(batch_predictions)

    print("Predictions complete.")
    return np.array(predictions)

def plot_results(axes, video_num, predictions, target, training_tuples, test_tuples, total_frame, fps, fpw, result_patience, final_test_loss):
    # Adjust target and predictions to align with result_patience

    axes.plot(range(total_frame), target, label='Target', color='gray')
    axes.plot(range(len(predictions)), predictions, label='Prediction', color='black')

    for tup in training_tuples:
        if tup[0] == video_num:
            start_frame = max(0, tup[1] - result_patience)
            end_frame = start_frame + fpw - result_patience
            axes.axvspan(start_frame, end_frame, color='red', alpha=0.05)

    for tup in test_tuples:
        if tup[0] == video_num:
            start_frame = max(0, tup[1] - result_patience)
            end_frame = start_frame + fpw - result_patience
            axes.axvspan(start_frame, end_frame, color='blue', alpha=0.05)

    video_name = ['Bird', 'City', 'Forest']
    axes.set_title(f'Video {video_name[video_num]}')
    axes.set_xlabel('Frame')
    axes.set_ylabel('Value')
    axes.legend(loc='upper right')
    axes.grid(True)
    # Add final test loss as text
    axes.text(0.95, 0.95, f'Final Test Loss: {final_test_loss:.5f}', transform=axes.transAxes, fontsize=12, verticalalignment='top', horizontalalignment='right')

# Training and evaluation code
batch_size = 50
frame_num = 30
lr = 1e-3
epochs = 200

h = 360
w = 720
c = 1
fps = 30
downsampling_factor = 8
fc = 0.7

channel_num_list = [128,128,128]

window_size = 0.067
sliding_size = window_size/2
frame_per_sliding = int(fps * sliding_size)
frame_per_window = int(fps * window_size)
result_patience = 1

model_string = f"frame_diff_CL{len(channel_num_list)}_"
for c in channel_num_list:
    model_string += f"{c}_"
model_string += f"{frame_per_window}frames"

folder_path = "./naturalistic"
mat_file_name = "experimental_data.mat"
checkpoint_name = "fly_model"
model_name = f"./model/{model_string}"
os.makedirs(model_name, exist_ok=True)

input_dims = 1  # [origin, up, down, right, left]
model = VCL(input_dims=input_dims, video_size=(h//downsampling_factor, w//downsampling_factor), result_patience=result_patience, channel_num_list=channel_num_list)
trainer = VCL_Trainer(model, lr)
current_epoch = trainer.load(f"{model_name}/{checkpoint_name}.ckpt")

video_data, optic_power, total_frame = seq_for_optic_cal(folder_path, downsampling_factor)

# Split period and split for training / test data set
# and save to model folder
# Load same tuples for performance comparison
filename = f'{model_name}/dataset_tuples.pkl'
if os.path.exists(filename):
    training_tuples, test_tuples = load_tuples(filename)
    print("Loaded tuples from file.")
else:
    training_tuples, test_tuples = generate_tuples_optic(total_frame, frame_per_sliding, int(fps * window_size))
    save_tuples(filename, (training_tuples, test_tuples))
    print("Generated and saved tuples.")
    
print(f"training_tuples : {np.shape(training_tuples)}")
print(f"test_tuples : {np.shape(test_tuples)}")

recent_losses = deque(maxlen=100)
test_losses_per_epoch = []

for epoch in range(current_epoch, epochs):
    batches = list(get_batches(training_tuples, batch_size))
    print("")
    print(f"Epoch {epoch + 1}/{epochs}")

    progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
    
    for batch in progress_bar:
        batch_input_data, batch_target_data = get_data_from_batch_v2(
            video_data, optic_power, batch, frame_per_window, fps)
        
        batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
        batch_target_data = torch.tensor(batch_target_data[:,result_patience:], dtype=torch.float32)
        
        loss, pred = trainer.step(batch_input_data, batch_target_data)
        
        recent_losses.append(loss.item())

        avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
        
        progress_bar.set_postfix(loss=f"{loss.item():.5f}", avg_recent_loss=f"{avg_recent_loss:.5f}")
        
        del batch_input_data, batch_target_data, loss, pred
    
    trainer.save(f"{model_name}/{checkpoint_name}.ckpt", epoch)
    
    # Test phase after each epoch
    test_batches = list(get_batches(test_tuples, batch_size))
    total_test_loss = 0.0
    progress_bar = tqdm(test_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=150)

    for batch in progress_bar:
        batch_input_data, batch_target_data = get_data_from_batch_v2(
            video_data, optic_power, batch, frame_per_window, fps)
        
        batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
        batch_target_data = torch.tensor(batch_target_data[:,result_patience:], dtype=torch.float32)
        
        loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
        total_test_loss = total_test_loss + loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.5f}")
        
        del batch_input_data, batch_target_data, loss, pred

    avg_test_loss = total_test_loss / len(test_batches)
    test_losses_per_epoch.append(avg_test_loss)
    print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")
    
    # Save test results
    epoch_results_path = f"{model_name}/results"
    os.makedirs(epoch_results_path, exist_ok=True)
    
    selected_tuples = []
    for video_num in range(3):
        video_specific_tuples = [t for t in test_tuples if t[0] == video_num]
        selected_tuples.extend(video_specific_tuples[:3])

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))  # Adjusted figure size
    
    for i in range(0, len(selected_tuples), batch_size):
        batch_tuples = selected_tuples[i:min(i + batch_size, 9)]
        
        batch_input_data, batch_target_data = get_data_from_batch_v2(
            video_data, optic_power, batch_tuples, frame_per_window, fps)

        batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
        batch_target_data = torch.tensor(batch_target_data[:,result_patience:], dtype=torch.float32)
        
        _, pred = trainer.evaluate(batch_input_data, batch_target_data)
        
        for j in range(len(batch_tuples)):
            video_num, start_frame = batch_tuples[j]
            ax = axes.flatten()[i + j]
            ax.plot(batch_target_data[j].cpu().numpy(), label="Target", color="gray")
            ax.plot(pred[j].cpu().numpy(), label="Prediction", color="black")  
            ax.set_title(f"Tuple: ( video {video_num}, frame {start_frame})")
            ax.legend()
        
        del batch_input_data, batch_target_data, pred

    plt.tight_layout()
    plt.savefig(f"{epoch_results_path}/epoch_{epoch + 1}.png")
    plt.close(fig)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Save the test losses to a file
with open(f"{model_name}/test_losses_per_epoch.pkl", 'wb') as f:
    pickle.dump(test_losses_per_epoch, f)

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

for video_num in range(3):
    print(f"Processing video {video_num} for prediction...")
    predictions = predict_with_model(model, trainer, video_data[video_num:video_num+1], frame_per_window, fps, result_patience)
    target = optic_power[video_num]

    # Save the predictions
    np.save(f"{prediction_path}/predictions_video_{video_num}.npy", predictions)
    np.save(f"{prediction_path}/target_video_{video_num}.npy", target)

    # Plot results
    plot_results(axes[video_num], video_num, predictions, target, training_tuples, test_tuples, total_frame, fps, frame_per_window, result_patience, final_test_loss)

plt.tight_layout()
plt.savefig(f"{prediction_path}/predictions_all_videos.png")
plt.close()

print("Prediction phase completed.")
