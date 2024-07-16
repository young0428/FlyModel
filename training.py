#%%
import torch
import numpy as np
from VCL import VCL, VCL_Trainer
from LoadDataset import *
from tqdm import tqdm  # Import tqdm for progress bars
from collections import deque
import pickle
import os

def save_tuples(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_tuples(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

batch_size = 3
frame_num = 30
lr = 1e-4
epochs = 30

h = 360
w = 720
c = 5
fps = 30
window_size = 1
sliding_size = 0.5
frame_per_sliding = int(fps * sliding_size)
frame_per_window = int(fps * window_size)
result_delay = 15


folder_path = "./naturalistic"
mat_file_name = "experimental_data.mat"
# test_input = torch.randn(batch_size, frame_num, h, w, c).half() # B, T, H, W, C

input_dims = 5  # [origin, up, down, right, left]
model = VCL(input_dims=input_dims, video_size=(h, w))
trainer = VCL_Trainer(model, lr)



video_data = LoadVideo(folder_path)  # (video_num, frame_num, h, w, c)
                                        # video_num : 0 = Bird, 1 = City, 2 = Forest
wba_data = convert_mat_to_array(f"{folder_path}/{mat_file_name}")
downsampled_wba_data = interpolate_wba_data(wba_data, original_freq=1000, target_freq=30)
total_frame = np.shape(video_data)[1]




# Make tuple for random batching and random sampling
# batch_set = (fly#, video#, trial#, start_frame)

#%%


filename = 'tuples.pkl'
if os.path.exists(filename):
    # 파일이 존재하면 튜플을 불러옴
    training_tuples, test_tuples = load_tuples(filename)
    print("Loaded tuples from file.")
else:
    # 파일이 존재하지 않으면 튜플을 생성하고 저장
    training_tuples, test_tuples = generate_tuples(total_frame, frame_per_sliding, int(fps * window_size))
    save_tuples(filename, (training_tuples, test_tuples))
    print("Generated and saved tuples.")
    
    
recent_losses = deque(maxlen=100)
for epoch in range(epochs):
    batches = list(get_batches(training_tuples, batch_size))
    print("")
    print(f"Epoch {epoch + 1}/{epochs}")

    progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=100)
    
    for batch in progress_bar:
        batch_input_data, batch_target_data = get_data_from_batch(
            video_data, downsampled_wba_data, batch, frame_per_window, fps, result_delay)
        
        batch_input_data = torch.tensor(batch_input_data)
        batch_target_data = torch.tensor(batch_target_data)
        
        loss, pred = trainer.step(batch_input_data, batch_target_data, result_delay=result_delay)
        
        recent_losses.append(loss.item())

        if len(recent_losses) > 0:
            avg_recent_loss = sum(recent_losses) / len(recent_losses)
        else:
            avg_recent_loss = 0
        
        progress_bar.set_postfix(loss=f"{loss.item():.5f}", avg_recent_loss=f"{avg_recent_loss:.5f}")
    
    trainer.save("fly_model.ckpt")
    
    # Test phase after each epoch
    test_batches = list(get_batches(test_tuples, batch_size))
    total_test_loss = 0
    progress_bar = tqdm(test_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=100)

    for batch in progress_bar:
        batch_input_data, batch_target_data = get_data_from_batch(
            video_data, downsampled_wba_data, batch, frame_per_window, fps, result_delay)
        
        batch_input_data = torch.tensor(batch_input_data)
        batch_target_data = torch.tensor(batch_target_data)
        
        loss, pred = trainer.evaluate(batch_input_data, batch_target_data, result_delay=result_delay)
        total_test_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.5f}")

    avg_test_loss = total_test_loss / len(test_batches)
    print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")


# %%
