import torch
import numpy as np
from VCL import VCL, VCL_Trainer
from LoadDataset import *
from tqdm import tqdm  # Import tqdm for progress bars

if __name__ == '__main__':
    batch_size = 3
    frame_num = 30
    lr = 1e-3
    epochs = 100
    
    h = 360
    w = 720
    c = 5
    fps = 30
    window_size = 1
    sliding_size = 0.5
    frame_per_sliding = int(fps * sliding_size)
    frame_per_window = int(fps * window_size)
    result_delay = 15
    
    
    folder_path = "/host/c/Users/AHNSOJUNG/Downloads/naturalistic"
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
    
    
    for epoch in range(epochs):
        video_start_point_tuples = generate_tuples(total_frame, frame_per_sliding, int(fps * window_size))
        batches = list(get_batches(video_start_point_tuples, batch_size))
        print("")
        print(f"Epoch {epoch + 1}/{epochs}")
        
        # Use tqdm to create a progress bar for the batches with reduced width
        progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=80)
        
        for batch in progress_bar:
            batch_input_data = []
            batch_target_data = []
            
            batch_input_data, batch_target_data = get_data_from_batch(video_data, batch, frame_per_window, fps)
            
            batch_input_data = torch.tensor(batch_input_data)
            batch_target_data = torch.tensor(batch_target_data)
            
            loss, pred = trainer.step(batch_input_data, batch_target_data, result_delay=result_delay)
            
            # Update the progress bar with the current loss
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")
        
        print(pred)
    
    exit()
