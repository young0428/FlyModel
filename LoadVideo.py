import cv2
import numpy as np
import os
import time
import random

def load_videos_to_tensor(video_paths):
    video_tensors = []
    first = True
    for video_path in video_paths:
        if not os.path.exists(video_path):
            print(f"File {video_path} does not exist.")
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if first :
                first = False
                continue
            if not ret:
                break
            # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)

        cap.release()

        if frames:
            # Stack frames and add a channel dimension for grayscale images
            video_tensor = np.stack(frames, axis=0)[..., np.newaxis]  # [time, height, width, 1]
            video_tensors.append(video_tensor)
        else:
            print(f"Failed to load video {video_path}")
            
    print(f"{video_paths[0]:60s} Loaded : {np.shape(video_tensors)}")
    return video_tensors

def combine_videos_to_tensor(video_paths_list):
    # Load each video to tensor
    video_tensors_list = [load_videos_to_tensor(video_paths) for video_paths in video_paths_list]

    # Combine videos by channel
    combined_videos = []
    for i in range(len(video_paths_list)):
        combined_tensor = np.concatenate(
            video_tensors_list[i], axis=-1)  # Combine on the channel dimension
        
        
        combined_videos.append(combined_tensor)
    
    return np.array(combined_videos)


def LoadVideo(folder_path):
    
    type_list = ['01_Bird', '02_City', '03_Forest']

    appendix = ['','_upward','_downward','_leftward','_rightward']
    video_paths_list = [[f"{folder_path}/{type}{ap}.avi" for ap in appendix] for type in type_list ]

    combined_video_tensors = combine_videos_to_tensor(video_paths_list)
    print("Video is allocated on memory!")
    print(f"Shape : {np.shape(combined_video_tensors)}")
    
    return combined_video_tensors

def generate_tuples(frame_num, frame_per_sliding, fps=30):
    tuples_list = []
    for n in range(3):  # n = 0, 1, 2
        for m in range(0, frame_num - fps, frame_per_sliding):
            tuples_list.append((n, m))
    return tuples_list

def get_batches(tuples_list, batch_size):
    random.shuffle(tuples_list)
    for i in range(0, len(tuples_list), batch_size):
        yield tuples_list[i:i + batch_size]
        
def get_data_from_batch(video_tensor, set, frame_per_window):
    video_data = []
    for set in batches:
        video_num, start_frame = set
        video_data.append(video_tensor[video_num, start_frame:start_frame + frame_per_window])
    return video_data
def get_wba_from_time(time, duration):
    pass


if __name__ == "__main__":
    frame_num = 3600
    frame_per_sliding = 15
    batch_size = 3
    
    
    folder_path = "C:/Users/dudgb2380/Downloads/naturalistic_video"
    video_data = LoadVideo(folder_path)
    print(video_data.shape)
    
    tuples = generate_tuples(frame_num, frame_per_sliding, 30)
    batches = list(get_batches(tuples, batch_size))
    

    # 결과 출력
    for t in batches:
        pass





