import torch
import numpy as np
from VCL import VCL, VCL_Trainer
from LoadDataset import *
from tqdm import tqdm
from collections import deque
import pickle
import os
import matplotlib.pyplot as plt

def save_tuples(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_tuples(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

batch_size = 7
frame_num = 30
lr = 1e-6
epochs = 50

h = 360
w = 720
c = 5
fps = 30
downsampling_factor = 4
fc = 0.7

window_size = 1
sliding_size = 0.5
frame_per_sliding = int(fps * sliding_size)
frame_per_window = int(fps * window_size)
result_patience = 15

folder_path = "./naturalistic"
mat_file_name = "experimental_data.mat"
checkpoint_name = "fly_model"
model_name = "./model/noPool_fc0.7_Forest"
os.makedirs(model_name, exist_ok=True)

input_dims = 5  # [origin, up, down, right, left]
model = VCL(input_dims=input_dims, video_size=(h//downsampling_factor, w// downsampling_factor), result_patience=result_patience)
trainer = VCL_Trainer(model, lr)
current_epoch = trainer.load(f"{model_name}/{checkpoint_name}.ckpt")

video_data, wba_data, total_frame = load_filtered_diff_data(folder_path, mat_file_name, downsampling_factor, fc=0.4)

filename = f'{model_name}/tuples_avg_trial.pkl'
if os.path.exists(filename):
    training_tuples, test_tuples = load_tuples(filename)
    print("Loaded tuples from file.")
else:
    training_tuples, test_tuples = generate_tuples(total_frame, frame_per_sliding, int(fps * window_size))
    save_tuples(filename, (training_tuples, test_tuples))
    print("Generated and saved tuples.")
    
print(f"training_tuples : {np.shape(training_tuples)}")
print(f"test_tuples : {np.shape(test_tuples)}")

recent_losses = deque(maxlen=100)
for epoch in range(current_epoch, epochs):
    batches = list(get_batches(training_tuples, batch_size))
    print("")
    print(f"Epoch {epoch + 1}/{epochs}")

    progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
    
    for batch in progress_bar:
        batch_input_data, batch_target_data = get_data_from_batch(
            video_data, wba_data, batch, frame_per_window, fps)
        
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
        batch_input_data, batch_target_data = get_data_from_batch(
            video_data, wba_data, batch, frame_per_window, fps)
        
        batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
        batch_target_data = torch.tensor(batch_target_data[:,result_patience:], dtype=torch.float32)
        
        loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
        total_test_loss = total_test_loss + loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.5f}")
        
        del batch_input_data, batch_target_data, loss, pred

    avg_test_loss = total_test_loss / len(test_batches)
    print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")
    
    # Save test results
    epoch_results_path = f"{model_name}/results"
    os.makedirs(epoch_results_path, exist_ok=True)
    
    selected_tuples = []
    for video_num in range(3):
        video_specific_tuples = [t for t in test_tuples if t[1] == video_num]
        selected_tuples.extend(video_specific_tuples[:3])

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for i in range(0, len(selected_tuples), batch_size):
        batch_tuples = selected_tuples[i:min(i + batch_size, 9)]
        
        batch_input_data, batch_target_data = get_data_from_batch(
            video_data, wba_data, batch_tuples, frame_per_window, fps)

        batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32)
        batch_target_data = torch.tensor(batch_target_data[:,result_patience:], dtype=torch.float32)
        
        _, pred = trainer.evaluate(batch_input_data, batch_target_data)
        
        for j in range(len(batch_tuples)):
            fly_num, video_num, start_frame = batch_tuples[j]
            ax = axes.flatten()[i + j]
            ax.plot(batch_target_data[j].cpu().numpy(), label="Target", color="gray", alpha=0.5)
            ax.plot(pred[j].cpu().numpy(), label="Prediction", color="black")  
            ax.set_title(f"Tuple: (fly {fly_num}, video {video_num}, frame {start_frame})")
            ax.legend()
        
        del batch_input_data, batch_target_data, pred

    plt.tight_layout()
    plt.savefig(f"{epoch_results_path}/epoch_{epoch + 1}.png")
    plt.close(fig)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("Training completed.")