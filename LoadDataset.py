import cv2
import numpy as np
import os
import random

from scipy.interpolate import interp1d

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

def generate_tuples(frame_num, frame_per_sliding, fps=30, fly_num = 38, video_num = 3, trial_num = 4):
    training_tuples_list = []
    test_tuples_list = []
    for video_n in range(video_num):  # n = 0, 1, 2, video#
        for fly_n in range(fly_num):
            for trial_n in range(trial_num):
                test_period = list(random.sample(range(10), 3))
                for start_frame in range(0, frame_num - fps, frame_per_sliding): # start_frame
                    if (start_frame / (frame_num - fps)) // 0.1 in test_period:
                        test_tuples_list.append((fly_n, video_n, trial_n, start_frame))
                    else:
                        training_tuples_list.append((fly_n, video_n, trial_n, start_frame))
                    
    return training_tuples_list, test_tuples_list

def get_batches(tuples_list, batch_size):
    random.shuffle(tuples_list)
    for i in range(0, len(tuples_list), batch_size):
        yield tuples_list[i:i + batch_size]

def convert_mat_to_array(mat_file_path):
    with h5py.File(mat_file_path, 'r') as mat_file:
        experimental_data = mat_file['experimental_data']
        
        num_flies = experimental_data.shape[3]
        num_videos = experimental_data.shape[2]
        num_trials = experimental_data.shape[1]
        
        
        wba_data = np.zeros((num_flies, num_videos, num_trials, 120000))
        
        for fly in range(num_flies):
            for video in range(num_videos):
                for trial in range(num_trials):
                    # LWBA 데이터 (index 3)
                    lwba_ref = experimental_data[3][trial][video][fly]
                    lwba_data = mat_file[lwba_ref][:]
                    
                    # RWBA 데이터 (index 4)
                    rwba_ref = experimental_data[4][trial][video][fly]
                    rwba_data = mat_file[rwba_ref][:]
                    
                    wba_data[fly, video, trial, :] = lwba_data - rwba_data
                    
    return wba_data

def interpolate_wba_data(wba_data, original_freq=1000, target_freq=30):
    original_freq = 1000 

    duration = wba_data.shape[-1] / original_freq 
    original_time = np.arange(0, wba_data.shape[-1]) / original_freq
    new_time = np.arange(0, duration, 1 / target_freq)


    new_data_shape = wba_data.shape[:-1] + (new_time.size,)
    wba_data_interpolated = np.zeros(new_data_shape)

    # 모든 데이터에 대해 인터폴레이션 수행
    for fly in range(wba_data.shape[0]):
        for video in range(wba_data.shape[1]):
            for trial in range(wba_data.shape[2]):
                # 데이터 추출
                wba_diff = wba_data[fly, video, trial, :]

                # 인터폴레이션 함수 생성
                interpolator = interp1d(original_time, wba_diff, kind='linear')

                wba_data_interpolated[fly, video, trial, :] = interpolator(new_time)

    return wba_data_interpolated

def get_wba_from_time(video_num, start_time, duration):
    pass
def get_data_from_batch(video_tensor, wba_tensor, batch_set, frame_per_window=1, fps=30, result_delay = 15):
    video_data = []
    wba_data = []
    # batch_set = (fly#, video#, trial#, start_frame)
    for set in batch_set:
        fly_num, video_num, trial_num, start_frame = set
        video_data.append(video_tensor[video_num, start_frame:start_frame + frame_per_window])
        #wba_data.append(np.ones(frame_per_window)) # for test
        #wba_data.append(get_wba_from_time(video_num, start_frame // fps, frame_per_window / fps))
        wba_data.append(wba_tensor[fly_num][video_num][trial_num][start_frame + 1 : start_frame + frame_per_window + 1])
        
    return np.array(video_data, dtype=np.float32), np.array(wba_data, dtype=np.float32)


import h5py
import matplotlib.pyplot as plt
if __name__ == "__main__":
    mat_file_path = "./experimental_data.mat"
    wba_data = convert_mat_to_array(mat_file_path)
    wba_data_interpolated = interpolate_wba_data(wba_data)
    
    
        
        
        
    
    # frame_num = 3600
    # frame_per_sliding = 15
    # batch_size = 3
    
    
    # folder_path = "C:/Users/dudgb2380/Downloads/naturalistic_video"
    # video_data = LoadVideo(folder_path)
    # print(video_data.shape)
    
    # tuples = generate_tuples(frame_num, frame_per_sliding, 30)
    # batches = list(get_batches(tuples, batch_size))
    

    # # 결과 출력
    # for t in batches:
    #     pass





