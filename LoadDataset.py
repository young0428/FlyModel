import cv2
import numpy as np
import os
import random
import numpy as np
from scipy.signal import butter, filtfilt

from scipy.interpolate import interp1d

def load_videos_to_tensor(video_paths, downsampling_factor = 1):
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
            if first:
                first = False
                continue
            if not ret:
                break

            # Convert frame to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (frame.shape[1] // downsampling_factor, frame.shape[0] // downsampling_factor))
            frames.append(frame)

        cap.release()

        if frames:
            video_tensor = np.stack(frames, axis=0)[..., np.newaxis]  # [time, height, width, 1]
            video_tensors.append(video_tensor)
        else:
            print(f"Failed to load video {video_path}")

    print(f"{video_paths[0]:60s} Loaded : {np.shape(video_tensors)}")
    return video_tensors

def combine_videos_to_tensor(video_paths_list, downsampling_factor = 1):

    video_tensors_list = [load_videos_to_tensor(video_paths, downsampling_factor) for video_paths in video_paths_list]


    combined_videos = []
    for i in range(len(video_paths_list)):
        combined_tensor = np.concatenate(
            video_tensors_list[i], axis=-1) 
        
        
        combined_videos.append(combined_tensor)
    
    return np.array(combined_videos)


def LoadVideo(folder_path, downsampling_factor = 1):
    
    type_list = ['01_Bird', '02_City', '03_Forest']
    appendix = ['','_upward','_downward','_leftward','_rightward']
    video_paths_list = [[f"{folder_path}/{type}{ap}.avi" for ap in appendix] for type in type_list ]

    combined_video_tensors = combine_videos_to_tensor(video_paths_list, downsampling_factor)
    print("Video is allocated on memory!")
    print(f"Shape : {np.shape(combined_video_tensors)}")
    
    return combined_video_tensors


def get_batches(tuples_list, batch_size):
    random.shuffle(tuples_list)
    for i in range(0, len(tuples_list), batch_size):
        yield tuples_list[i:i + batch_size]
        
def interpolate_and_diff_wba_data(wba_data, original_freq=1000, target_freq=30):
    original_freq = 1000

    duration = wba_data.shape[-1] / original_freq
    original_time = np.arange(0, wba_data.shape[-1]) / original_freq
    new_time = np.arange(0, duration, 1 / target_freq)

    new_data_shape = wba_data.shape[:-1] + (new_time.size,)
    wba_data_interpolated = np.zeros(new_data_shape)

    for fly in range(wba_data.shape[0]):
        for video in range(wba_data.shape[1]):
            # 데이터 추출
            wba_diff = wba_data[fly, video, :]

            # 인터폴레이션 함수 생성
            interpolator = interp1d(original_time, wba_diff, kind='linear')

            wba_data_interpolated[fly, video, :] = interpolator(new_time)

    # 마지막 차원에 대해 차분을 구함
    wba_data_diff = np.diff(wba_data_interpolated, axis=-1)

    return wba_data_interpolated, wba_data_diff


########################### average 안함 ###########################
# def generate_tuples(frame_num, frame_per_sliding, fps=30, fly_num = 38, video_num = 3, trial_num = 4):
#     training_tuples_list = []
#     test_tuples_list = []
#     for video_n in range(video_num):  # n = 0, 1, 2, video#
#         for fly_n in range(fly_num):
#             for trial_n in range(trial_num):
#                 test_period = list(random.sample(range(10), 3))
#                 for start_frame in range(0, frame_num - fps, frame_per_sliding): # start_frame
#                     if (start_frame / (frame_num - fps)) // 0.1 in test_period:
#                         test_tuples_list.append((fly_n, video_n, trial_n, start_frame))
#                     else:
#                         training_tuples_list.append((fly_n, video_n, trial_n, start_frame))
                    
#     return training_tuples_list, test_tuples_list


# def convert_mat_to_array(mat_file_path):
#     with h5py.File(mat_file_path, 'r') as mat_file:
#         experimental_data = mat_file['experimental_data']
        
#         num_flies = experimental_data.shape[3]
#         num_videos = experimental_data.shape[2]
#         num_trials = experimental_data.shape[1]
        
        
#         wba_data = np.zeros((num_flies, num_videos, num_trials, 120000))
        
#         for fly in range(num_flies):
#             for video in range(num_videos):
#                 for trial in range(num_trials):
#                     # LWBA 데이터 (index 3)
#                     lwba_ref = experimental_data[3][trial][video][fly]
#                     lwba_data = mat_file[lwba_ref][:]
                    
#                     # RWBA 데이터 (index 4)
#                     rwba_ref = experimental_data[4][trial][video][fly]
#                     rwba_data = mat_file[rwba_ref][:]
                    
#                     wba_data[fly, video, trial, :] = lwba_data - rwba_data
                    
#     return wba_data

# def get_data_from_batch(video_tensor, wba_tensor, batch_set, frame_per_window=1, fps=30, result_delay = 15):
#     video_data = []
#     wba_data = []
#     # batch_set = (fly#, video#, trial#, start_frame)
#     for set in batch_set:
#         fly_num, video_num, trial_num, start_frame = set
#         video_data.append(video_tensor[video_num, start_frame:start_frame + frame_per_window])
#         #wba_data.append(np.ones(frame_per_window)) # for test
#         #wba_data.append(get_wba_from_time(video_num, start_frame // fps, frame_per_window / fps))
#         wba_data.append(wba_tensor[fly_num][video_num][trial_num][start_frame + 1 : start_frame + frame_per_window + 1])
        
#     return np.array(video_data, dtype=np.float32), np.array(wba_data, dtype=np.float32)

# def interpolate_wba_data(wba_data, original_freq=1000, target_freq=30):
#     original_freq = 1000 

#     duration = wba_data.shape[-1] / original_freq 
#     original_time = np.arange(0, wba_data.shape[-1]) / original_freq
#     new_time = np.arange(0, duration, 1 / target_freq)


#     new_data_shape = wba_data.shape[:-1] + (new_time.size,)
#     wba_data_interpolated = np.zeros(new_data_shape)

#     # 모든 데이터에 대해 인터폴레이션 수행
#     for fly in range(wba_data.shape[0]):
#         for video in range(wba_data.shape[1]):
#             for trial in range(wba_data.shape[2]):
#                 # 데이터 추출
#                 wba_diff = wba_data[fly, video, trial, :]

#                 # 인터폴레이션 함수 생성
#                 interpolator = interp1d(original_time, wba_diff, kind='linear')

#                 wba_data_interpolated[fly, video, trial, :] = interpolator(new_time)

#     return wba_data_interpolated
# def get_data_from_batch(video_tensor, wba_tensor, batch_set, frame_per_window=1, fps=30, result_delay = 15):
#     video_data = []
#     wba_data = []
#     # batch_set = (fly#, video#, trial#, start_frame)
#     for set in batch_set:
#         #fly_num, video_num, trial_num, start_frame = set
#         fly_num, video_num, start_frame = set
#         video_data.append(video_tensor[video_num, start_frame:start_frame + frame_per_window])
#         wba_data.append(wba_tensor[fly_num][video_num][start_frame + 1 : start_frame + frame_per_window + 1])
        
#     return np.array(video_data, dtype=np.float32), np.array(wba_data, dtype=np.float32)
#################################################################################

########################### trial average ###########################
def generate_tuples(frame_num, frame_per_sliding, fps=30, fly_num = 38, video_num = 3, trial_num = 4):
    training_tuples_list = []
    test_tuples_list = []
    for video_n in range(0,3):  # n = 0, 1, 2, video#
        test_period = list(random.sample(range(10), 3))
        for fly_n in range(fly_num):
            for start_frame in range(0, frame_num - fps, frame_per_sliding): # start_frame
                if (start_frame / (frame_num - fps)) // 0.1 in test_period:
                    test_tuples_list.append((fly_n, video_n, start_frame))
                else:
                    training_tuples_list.append((fly_n, video_n, start_frame))
                    
    return training_tuples_list, test_tuples_list



def low_pass_filter(data, cutoff_freq, sample_rate=1000):
    nyquist_rate = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist_rate
    b, a = butter(N=4, Wn=normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=-1)
    return filtered_data

def apply_low_pass_filter_to_wba_data(wba_data, cutoff_freq, sample_rate=1000):
    num_flies, num_videos, _ = wba_data.shape
    filtered_wba_data = np.zeros_like(wba_data)
    
    for fly in range(num_flies):
        for video in range(num_videos):
            filtered_wba_data[fly, video, :] = low_pass_filter(wba_data[fly, video, :], cutoff_freq, sample_rate)
            
    return filtered_wba_data

def convert_mat_to_array(mat_file_path):
    with h5py.File(mat_file_path, 'r') as mat_file:
        experimental_data = mat_file['experimental_data']
        
        num_flies = experimental_data.shape[3]
        num_videos = experimental_data.shape[2]
        num_trials = experimental_data.shape[1]
        
        # trial 별 평균을 저장할 배열
        wba_data = np.zeros((num_flies, num_videos, 120000))
        
        for fly in range(num_flies):
            for video in range(num_videos):
                lwba_sum = np.zeros(120000)
                rwba_sum = np.zeros(120000)
                
                for trial in range(num_trials):
                    # LWBA 데이터 (index 3)
                    lwba_ref = experimental_data[3][trial][video][fly]
                    lwba_data = mat_file[lwba_ref][:]
                    
                    # RWBA 데이터 (index 4)
                    rwba_ref = experimental_data[4][trial][video][fly]
                    rwba_data = mat_file[rwba_ref][:]
                    
                    # 배열을 1차원으로 변환
                    lwba_data = np.squeeze(lwba_data)
                    rwba_data = np.squeeze(rwba_data)
                    
                    lwba_sum += lwba_data
                    rwba_sum += rwba_data
                
                # trial 평균 계산
                lwba_avg = lwba_sum / num_trials
                rwba_avg = rwba_sum / num_trials
                
                wba_data[fly, video, :] = lwba_avg - rwba_avg
                    
    return wba_data

def get_data_from_batch_v2(video_tensor, wba_tensor, batch_set, frame_per_window=1, fps=30):
    video_data = []
    wba_data = []
    # batch_set = (fly#, video#, start_frame)
    for set in batch_set:
        fly_num, video_num, start_frame = set
        video_data.append(video_tensor[video_num, start_frame:start_frame + frame_per_window])
        wba_data.append(wba_tensor[fly_num][video_num][start_frame + 1: start_frame + frame_per_window + 1])
        
    return np.array(video_data, dtype=np.float32), np.array(wba_data, dtype=np.float32)

def load_filtered_diff_data(folder_path, mat_file_name, downsampling_factor, fc = 0.4):
    video_data = LoadVideo(folder_path, downsampling_factor)
    wba_data = convert_mat_to_array(f"{folder_path}/{mat_file_name}")
    filtered_wba_data = apply_low_pass_filter_to_wba_data(wba_data, fc)
    _, interpolated_diff_wba_data = interpolate_and_diff_wba_data(filtered_wba_data, original_freq=1000, target_freq=30)
    total_frame = np.shape(video_data)[1]
    
    return video_data, interpolated_diff_wba_data, total_frame

def generate_tuples_optic(frame_num, frame_per_sliding, fps=30, video_num = 3):
    training_tuples_list = []
    test_tuples_list = []
    for video_n in range(video_num):  # n = 0, 1, 2, video#
        test_period = list(random.sample(range(10), 3))
        for start_frame in range(0, frame_num - fps, frame_per_sliding): # start_frame
            if (start_frame / (frame_num - fps)) // 0.1 in test_period:
                test_tuples_list.append((video_n, start_frame))
            else:
                training_tuples_list.append((video_n, start_frame))
                    
    return training_tuples_list, test_tuples_list


def cal_video_to_optic_power(video_tensor):
    # (3, frame#, h, w, c)
    video_count, frame_count, height, width, channels = video_tensor.shape
    
    results = []
    
    for video_idx in range(video_count):
        video_results = []
        for frame_idx in range(frame_count):
            # optic flow
            frame = video_tensor[video_idx, frame_idx]
            left_optic_flow = frame[:, :, 3]  # ch3: leftward optic flow
            right_optic_flow = frame[:, :, 4]  # ch4: rightward optic flow
            
            # 각 프레임마다 채널 3과 4의 평균 값 계산
            left_mean = np.mean(left_optic_flow)
            right_mean = np.mean(right_optic_flow)
            
            # 왼쪽 평균에서 오른쪽 평균을 뺀 값 계산
            difference = left_mean - right_mean
            video_results.append(difference)
            
            ## intensity
            # frame = video_tensor[video_idx, frame_idx]
            # power = frame[:,:,0]
            # power_mean = np.mean(power)
            # video_results.append(power_mean)
            
            # if frame_idx == 0:
            #     intensity_diff = np.zeors(np.shape(video_tensor[video_idx, frame_idx, :, :, 0]))
            # else:
            #     intensity_diff = video_tensor[video_idx, frame_idx, :, :, 0] - video_tensor[video_idx, frame_idx-1, :, :, 0]
            
            # intensity_mean = np.mean(intensity_diff)
            # video_results.append(intensity_mean)

        results.append(video_results)
    
    return results

def get_data_from_batch_v2(video_tensor, optic_power_tensor, batch_set, frame_per_window=1, fps=30):
    video_data = []
    optic_data = []
    for set in batch_set:
        video_num, start_frame = set
        video_data.append(video_tensor[video_num][start_frame:start_frame + frame_per_window])
        optic_data.append(optic_power_tensor[video_num][start_frame:start_frame + frame_per_window])
        
    return np.array(video_data, dtype=np.float32), np.array(optic_data, dtype=np.float32)

def seq_for_optic_cal(folder_path, downsampling_factor):
    video_data = LoadVideo(folder_path, downsampling_factor)
    optic_power_tensor = cal_video_to_optic_power(video_data)           # (video#, frame#, 1)
    original_video = np.expand_dims(video_data[:,:,:,:,0] , axis=-1)    # (video#, frame#, h, w, 1)
    total_frame = np.shape(video_data)[1]
    return original_video, optic_power_tensor, total_frame
    
    
    

#%%
import h5py
import matplotlib.pyplot as plt
if __name__ == "__main__":
    mat_file_path = "./naturalistic/experimental_data.mat"
    optic_power = convert_mat_to_array(mat_file_path)
    
#%%
    wba_data_filtered = apply_low_pass_filter_to_wba_data(optic_power, 0.7, 1000)
    interpolated_filtered_wba_data, interpolated_filtered_wba_diff_data = interpolate_and_diff_wba_data(wba_data_filtered)
    
    interpolated_orignal_wba_data, interpolated_orignal_wba_diff_data = interpolate_and_diff_wba_data(optic_power)
    fps = 30
    start = 0 *fps
    end = 35 *fps
    interpolated_filtered_wba_data1 = interpolated_filtered_wba_data[0][2][start:end]
    interpolated_filtered_wba_data2 = interpolated_filtered_wba_data[1][2][start:end]
    
    interpolated_filtered_wba_diff_data1 = interpolated_filtered_wba_diff_data[0][2][start:end]
    interpolated_filtered_wba_diff_data2 = interpolated_filtered_wba_diff_data[1][2][start:end]
    
    interpolated_orignal_wba_data = interpolated_orignal_wba_data[0][2][start:end]
    interpolated_orignal_wba_diff_data = interpolated_orignal_wba_diff_data[0][2][start:end]

    
    plt.subplot(2, 1, 1)
    
    plt.plot(interpolated_filtered_wba_data1)
    plt.plot(interpolated_filtered_wba_data2)
    
    plt.title('Interpolated WBA Data')
    plt.xlabel('Time')
    plt.ylabel('WBA')
    plt.grid(True)

    # 두 번째 서브플롯
    plt.subplot(2, 1, 2)
    
    plt.plot(interpolated_filtered_wba_diff_data1)
    plt.plot(interpolated_filtered_wba_diff_data2)
    
    plt.title('Interpolated and Differenced WBA Data')
    plt.xlabel('Time')
    plt.ylabel('Differenced WBA')
    plt.grid(True)

    # 레이아웃 조정 및 표시
    plt.tight_layout()
    plt.show()
#%%
    
    


