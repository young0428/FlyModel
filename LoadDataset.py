import cv2
import numpy as np
import os
import random
import numpy as np
import pickle
import torch
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
            frame = cv2.resize(frame, (int(frame.shape[1] // downsampling_factor), int(frame.shape[0] // downsampling_factor)))
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
        yield tuples_list[i:min(i + batch_size, len(tuples_list))]
        
def interpolate_wba_data(wba_data, original_freq=1000, target_freq=30):
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
    

    return wba_data_interpolated

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

def predict_with_model_sac(trainer, video_data, val_start_frame, val_end_frame, frame_per_window, fps):
    total_frame = video_data.shape[1]
    step = int(10 * fps)  # 10 seconds window
    predictions = []

    print("Starting predictions...")

    video_data = torch.tensor(video_data, dtype=torch.float32).to(trainer.device)
    
    batch_predictions = []

    with torch.no_grad():
        for frame_num in range(val_start_frame, val_end_frame):
            batch_input = video_data[0:1,frame_num - frame_per_window : frame_num]
            
            pred = trainer.model(batch_input)
            batch_predictions.append(pred.squeeze(0).cpu().numpy())

    predictions.extend(batch_predictions)

    print("Predictions complete.")
    return np.array(predictions)

def predict_with_model_wba(trainer, video_data, val_start_frame, val_end_frame, frame_per_window, fps):
    total_frame = video_data.shape[1]
    step = int(10 * fps)  # 10 seconds window
    predictions = []

    print("Starting predictions...")

    video_data = torch.tensor(video_data, dtype=torch.float32).to(trainer.device)
    
    batch_predictions = []

    with torch.no_grad():
        for frame_num in range(val_start_frame, val_end_frame):
            batch_input = video_data[0:1,frame_num - frame_per_window : frame_num]
            
            pred = trainer.model(batch_input)
            predict_label = np.argmax(pred.cpu(), axis=1)
            batch_predictions.append(predict_label.squeeze(0).cpu().numpy())


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
def save_results(filename, target, predictions):

    with h5py.File(filename, 'w') as f:
        f.create_dataset('target', data=np.array(target))
        f.create_dataset('predictions', data=np.array(predictions))

def load_results(filename):

    with h5py.File(filename, 'r') as f:
        target = list(f['target'])
        predictions = list(f['predictions'])
    return target, predictions
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

def vertical_flip(video):
    
    return np.flip(video, axis=1)  # Vertical flip (상하 반전)

def horizontal_flip(video):
    
    return np.flip(video, axis=2)  # Horizontal flip (좌우 반전)

def swap_channels(video, index1=1, index2=2):
    frames = video.copy()
    frames[:, :, :, [index1, index2]] = frames[:, :, :, [index2, index1]]
    return frames

def zero_quadrant(video, quadrant):
    frames = video.copy()
    h, w = frames.shape[1:3]
    if quadrant == 'LU':
        frames[:, :h//2, :w//2, :] = 0  # Left Upper
    elif quadrant == 'RU':
        frames[:, :h//2, w//2:, :] = 0  # Right Upper
    elif quadrant == 'LD':
        frames[:, h//2:, :w//2, :] = 0  # Left Down
    elif quadrant == 'RD':
        frames[:, h//2:, w//2:, :] = 0  # Right Down
    return frames

def crop_and_resize(video, quadrant):
    frames = video.copy()
    h, w = frames.shape[1:3]
    if quadrant == 'LU':
        frames = frames[:, :h//2, :w//2, :]  # Left Upper
    elif quadrant == 'RU':
        frames = frames[:, :h//2, w//2:, :]  # Right Upper
    elif quadrant == 'LD':
        frames = frames[:, h//2:, :w//2, :]  # Left Down
    elif quadrant == 'RD':
        frames = frames[:, h//2:, w//2:, :]  # Right Down
    frames = np.array([cv2.resize(frame, (w, h)) for frame in frames])
    return frames

def central_crop_and_resize(video, crop_size_ratio=0.5):
    frames = video.copy()
    h, w = frames.shape[1:3]
    crop_h, crop_w = int(h * crop_size_ratio), int(w * crop_size_ratio)
    start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
    frames = frames[:, start_h:start_h+crop_h, start_w:start_w+crop_w, :]
    frames = np.array([cv2.resize(frame, (w, h)) for frame in frames])
    return frames

def apply_gaussian_blur(video, kernel_size=(5, 5), sigma=0.1):
    frames = video.copy()
    blurred_frames = []
    
    for frame in frames:
        if frame.ndim == 2:  # 흑백 이미지일 경우
            blurred_frame = cv2.GaussianBlur(frame, kernel_size, sigma)
            blurred_frames.append(np.expand_dims(blurred_frame, axis=-1))  # 채널 차원 유지
        else:
            # 각 채널에 대해 Gaussian Blur 적용
            blurred_frame = np.stack([cv2.GaussianBlur(frame[:, :, ch], kernel_size, sigma) for ch in range(frame.shape[2])], axis=-1)
            blurred_frames.append(blurred_frame)
    
    return np.array(blurred_frames)

def apply_salt_and_pepper(video, salt_prob=0.001, pepper_prob=0.001):
    frames = video.copy()
    h, w, c = frames.shape[1:4]
    
    salt_threshold = salt_prob * (h * w)
    pepper_threshold = pepper_prob * (h * w)

    for i in range(frames.shape[0]):
        # Salt noise: 모든 채널에 동일한 위치에 적용
        num_salt = np.ceil(salt_threshold)
        coords = [np.random.randint(0, dim - 1, int(num_salt)) for dim in frames[i, :, :, 0].shape]
        for ch in range(c):
            frames[i, coords[0], coords[1], ch] = 1  # 각 채널에 같은 위치에 salt 노이즈 적용

        # Pepper noise: 모든 채널에 동일한 위치에 적용
        num_pepper = np.ceil(pepper_threshold)
        coords = [np.random.randint(0, dim - 1, int(num_pepper)) for dim in frames[i, :, :, 0].shape]
        for ch in range(c):
            frames[i, coords[0], coords[1], ch] = 0  # 각 채널에 같은 위치에 pepper 노이즈 적용

    return frames







# ############################    flow estimation part end    ##############################################



###################### for flow estimation #########################


def calculate_manual_wba(video_data):
    """
    각 프레임별로 leftward와 rightward flow vector의 평균을 구하고, 이를 빼서 최종적으로 manual_wba를 반환하는 함수.
    
    Args:
    video_data (numpy.ndarray): (3, frame#, h, w, c) 형태의 비디오 데이터. c=3, c=4가 leftward, rightward flow vector의 정보를 담고 있음.
    
    Returns:
    numpy.ndarray: (3, frame#) 형태의 manual_wba.
    """

    # leftward와 rightward flow vector 추출
    leftward_flow = video_data[..., 3]
    rightward_flow = video_data[..., 4]

    # 각 프레임별로 leftward와 rightward flow vector의 평균 계산
    leftward_mean = np.mean(leftward_flow, axis=(2, 3))  # (3, frame#)
    rightward_mean = np.mean(rightward_flow, axis=(2, 3))  # (3, frame#)

    # leftward와 rightward의 평균을 빼서 manual_wba 계산
    manual_wba = leftward_mean - rightward_mean  # (3, frame#)

    return manual_wba

def aug_videos(videos, wba_data):
    augmented_videos = []
    augmented_wba = []
    before_aug_num = len(videos)
    for i, video in enumerate(videos):
        # 원본 영상 추가
        augmented_videos.append(video)
        augmented_wba.append(wba_data[i])

        augmented_videos.append(apply_gaussian_blur(video))
        augmented_wba.append(wba_data[i])

        augmented_videos.append(apply_salt_and_pepper(video))
        augmented_wba.append(wba_data[i])

        augmented_videos.append(vertical_flip(video))
        augmented_wba.append(wba_data[i])
        
        horizontal_flipped_video = horizontal_flip(video)
        augmented_videos.append(horizontal_flipped_video)
        augmented_wba.append(-wba_data[i])
        
        augmented_videos.append(apply_gaussian_blur(horizontal_flipped_video))
        augmented_wba.append(-wba_data[i])
        
        augmented_videos.append(apply_salt_and_pepper(horizontal_flipped_video))
        augmented_wba.append(-wba_data[i])
        augmented_videos.append(vertical_flip(horizontal_flipped_video))
        augmented_wba.append(-wba_data[i])
        
    after_aug_num = len(augmented_videos)
    
    aug_factor = after_aug_num // before_aug_num

        
    return np.array(augmented_videos), np.array(augmented_wba), aug_factor

def direction_pred_training_data_preparing_seq(folder_path, mat_file_path, downsampling_factor):
    
    video_data = LoadVideo(folder_path, downsampling_factor)
    #manual_wba_data = calculate_manual_wba(video_data)
    wba_data = convert_mat_to_array(f"{folder_path}/{mat_file_path}")
    wba_data_filtered = apply_low_pass_filter_to_wba_data(wba_data, 0.4)
    wba_data_interpolated = np.mean(interpolate_wba_data(wba_data_filtered, original_freq=1000, target_freq=30),axis=0)
    diff_wba_data = np.diff(wba_data_interpolated,axis=-1)
    total_frame = np.shape(video_data)[1]

    return video_data, wba_data, total_frame

def generate_tuples_direction_pred(frame_num, frame_per_window, frame_per_sliding, video_num = 3):
    tuples = []
    
    # 0 = Bird
    # 1 = City
    # 2 = forest
    selected_video_num = 2
    for start_frame in range(frame_per_window, frame_num, frame_per_sliding): # start_frame
        for video_n in range((video_num//3)*selected_video_num, (video_num//3)*(selected_video_num+1)):
            tuples.append((video_n, start_frame))
            
    # for start_frame in range(frame_per_window, frame_num, frame_per_sliding): # start_frame
    #     for video_n in range(1, 6, 3):
    #         tuples.append((video_n, start_frame))
                    
    return tuples

def get_data_from_batch_direction_pred(video_tensor, wba_tensor, batch_set, frame_per_window=1):
    video_data = []
    direction_data = []
    for set in batch_set:
        video_num, start_frame = set
        video_data.append(video_tensor[video_num, start_frame-frame_per_window:start_frame,:,:,0:1])
        direction_data.append([1] if np.mean(wba_tensor[video_num, start_frame]) >= 0 else [0])
        #direction_data.append([1] if np.mean(wba_tensor[video_num, start_frame-frame_per_window:start_frame]) >= 0 else [0])
        #direction_data.append([1] if wba_tensor[video_num, start_frame] >= wba_tensor[video_num, start_frame-frame_per_window] else [0])
        
    return np.array(video_data), np.array(direction_data)

def prepare_data(folder_path, mat_file_name, downsampling_factor):
    video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
    video_data, wba_data = aug_videos(video_data, wba_data)
    print(f"Augmented video data shape: {video_data.shape}")
    print(f"Augmented WBA data shape: {wba_data.shape}")
    return video_data, wba_data, total_frame

############################    flow estimation part end    ##############################################



#%%
import h5py
import matplotlib.pyplot as plt
import numpy as np
if __name__ == "__main__":
    # mat_file_path = "./saccade_prediction_data.mat"
    # sac_data = interpolate_data(get_sac_data(mat_file_path))
    # fig, axes = plt.subplots(2, 1, figsize=(16, 16))
    # sac_binary = compare_and_create_binary_array(sac_data)
    # print(np.shape(sac_binary))
    # axes[0].plot(sac_data[0,:300,0], label='left', color='red')
    # axes[0].plot(sac_data[0,:300,1], label='right', color='blue')
    # axes[0].plot(sac_binary[0,:300]/2, label='sac_dir',color='gray')
    # axes[0].legend()

    # print(np.shape(sac_binary))
    
    # plt.show()
    folder_path = "./naturalistic"
    mat_file_name = "experimental_data.mat"
    ds = 5.625
    video_data, wba_data_interpolated, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, ds)
    #%%
    #%%
    #plt.plot(wba_data_filtered[i, 2,::1000//30], color='red')
    

    plt.plot(wba_data_interpolated[2,:], color='blue')
    
    #plt.plot(mean_wba[2,:3000],color='blue')
    plt.show()
    plt.close()
    
#%%

    



    
    
    
    
    
    
    
    
    
#%%
    
#%%
    
    


