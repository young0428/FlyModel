



import torch
import numpy as np
import matplotlib.pyplot as plt
from FlowNet import *
from trainer_func import *
from LoadDataset import *
from tqdm import tqdm
from collections import deque
import pickle
import os
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, confusion_matrix
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="dropout2d: Received a 5-D input")

torch.autograd.set_detect_anomaly(True)

# 데이터 준비 함수
def prepare_data(folder_path, mat_file_name, downsampling_factor):
    video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
    video_data, wba_data = aug_videos(video_data, wba_data)
    print(f"Augmented video data shape: {video_data.shape}")
    print(f"Augmented WBA data shape: {wba_data.shape}")
    return video_data, wba_data, total_frame



# 메인 코드 시작
if __name__ == "__main__":
    h = 360
    w = 720
    c = 1
    fps = 30
    downsampling_factor = 5.625
    frame_per_window = 16
    frame_per_sliding = 4
    input_ch = 1

    model_string = "only_forest_ud_f1_score"
    model_string += f"_{frame_per_window}frames"

    folder_path = "./naturalistic"
    mat_file_name = f"experimental_data.mat"
    checkpoint_name = "fly_model"

    model_name = f"./model/{model_string}"
    os.makedirs(model_name, exist_ok=True)

    pretrained_model_path = "./pretrained_model/64x128_opticflow_64t51216frames.ckpt"
    batch_size = 20
    lr = 1e-3
    epochs = 100
    fold_factor = 8
    layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]

    video_data, wba_data, total_frame = prepare_data(folder_path, mat_file_name, downsampling_factor)

    batch_tuples = np.array(generate_tuples_direction_pred(total_frame, frame_per_window, frame_per_sliding, video_data.shape[0]))
    kf = KFold(n_splits=fold_factor)

    all_fold_losses = []

    # 각 fold에 대해 학습
    for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):
        print(f"Fold {fold+1}")
        fold_path = f"{model_name}/fold_{fold+1}"
        os.makedirs(fold_path, exist_ok=True)

        # 모델과 트레이너 초기화
        model = initialize_model(layer_configs, pretrained_model_path)
        trainer, current_epoch = initialize_trainer(model, lr, fold_path, checkpoint_name)

        # 모델 학습
        fold_loss = train_model(fold, train_index, val_index, batch_tuples, video_data, wba_data, fold_path, epochs, trainer, frame_per_window, batch_size, fold_factor)
        all_fold_losses.append(fold_loss)

    # 전체 fold 요약
    summarize_folds(all_fold_losses, model_name)
    
