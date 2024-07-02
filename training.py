import torch
import numpy as np
from VCL import VCL, VCL_Trainer
from LoadVideo import *



if __name__ == '__main__':
    batch_size = 3
    frame_num = 30
    lr = 1e-6
    epochs = 100
    
    h = 360
    w = 720
    c = 5
    fps = 30
    window_size = 1
    sliding_size = 0.5
    frame_per_sliding = int(fps*sliding_size)
    frame_per_window = int(fps*window_size)
    
    
    #test_input = torch.randn(batch_size, frame_num, h, w, c).half() # B, T, H, W, C
    test_output = torch.randn(batch_size, frame_num).half()
    
    input_dims = 5 # [origin, up, down, right, left]
    model = VCL(input_dims = input_dims, video_size=(h, w))
    trainer = VCL_Trainer(model, lr)

    folder_path = "C:/Users/전영규/Desktop/naturalistic"
    
    
    video_data = LoadVideo(folder_path) # (video_num, frame_num, h, w, c)
                                        # video_num : 0 = Bird, 1 = City, 2 = Forest
    total_frame = np.shape(video_data)[1]
    
    print(np.shape(video_data))
    
    # make tuple for random batching and random sampling 
    # (video_num, frame_num)
    # to get data, video_data[ batch[0] ][ batch[1]:batch[1]+frame_per_window ]
    
    
    
    # for epoch in range(epochs):
    #     video_start_point_tuples = generate_tuples(total_frame, frame_per_sliding, int(fps*window_size))
    #     batches = list(get_batches(video_start_point_tuples, batch_size))
    #     batch_input_data = []
    #     batch_target_data = []
    #     print(f"Epoch {epoch+1}")
    #     for batch in batches:
    #         for start_point in batch:
    #             batch_input_data.append(video_data[start_point[0]][ start_point[1]:start_point[1]+frame_per_window ])
    #             batch_target_data.append(get_wba_from_time(start_point // fps, window_size))
                
    #     batch_input_data = torch.tensor(batch_input_data).half()
    #     batch_target_data = torch.tensor(batch_target_data).half()
        
    #     loss = trainer.step(batch_input_data, batch_target_data)
    
    exit()
    