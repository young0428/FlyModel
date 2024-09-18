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
import time
from datetime import datetime, timedelta
import pytz
from sklearn.metrics import f1_score, confusion_matrix  # confusion_matrix import
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="dropout2d: Received a 5-D input")

torch.autograd.set_detect_anomaly(True)

h = 360
w = 720
c = 1
fps = 30
downsampling_factor = 5.625

frame_per_window = 32
frame_per_sliding = 8
input_ch = 1

model_string = "only_forest_ud_nf_SA"
model_string += f"_{frame_per_window}frames"

folder_path = "./naturalistic"
mat_file_name = f"experimental_data.mat"
checkpoint_name = "fly_model"

model_name = f"./model/{model_string}"
os.makedirs(model_name, exist_ok=True)
result_save_path = f"./model/{model_string}/result_data.h5"

pretrained_model_path = "./pretrained_model/64x128_opticflow_64t51216frames.ckpt"

# hyperparameter 
batch_size = 20
lr = 1e-3
epochs = 100
fold_factor = 8

layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]

video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
video_data,wba_data = aug_videos(video_data, wba_data)
print(f"augmented shape : {video_data.shape}")
print(f"augmented shape : {wba_data.shape}")

# Split period and split for training / test data set
recent_losses = deque(maxlen=100)
recent_f1_scores = deque(maxlen=100)
val_losses_per_epoch = []

batch_tuples = np.array(generate_tuples_direction_pred(total_frame, frame_per_window, frame_per_sliding, video_data.shape[0]))
print(np.shape(batch_tuples))
kf = KFold(n_splits=fold_factor)

all_fold_losses = []

# Confusion Matrix 저장 경로
conf_matrix_train_path = f"{model_name}/train_confusion_matrix.pkl"
conf_matrix_val_path = f"{model_name}/val_confusion_matrix.pkl"

# 파일 초기화 (이전 내용 덮어쓰기)
with open(conf_matrix_train_path, "wb") as f:
    pickle.dump([], f)
with open(conf_matrix_val_path, "wb") as f:
    pickle.dump([], f)

KST = pytz.timezone('Asia/Seoul')
for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):
    print(f"Fold {fold+1}")
    fold_path = f"{model_name}/fold_{fold+1}"
    
    

    # create model
    flownet_model = flownet3d(layer_configs, num_classes=1)
    flownet_model = load_model(flownet_model, pretrained_model_path)
    model = FlowNet3DWithFeatureExtraction(flownet_model, feature_dim=128)
    trainer = Trainer(model, loss_function_bce, lr)
    current_epoch = trainer.load(f"{fold_path}/{checkpoint_name}.ckpt")
    os.makedirs(fold_path, exist_ok=True)

    with open(f"{fold_path}/training_tuples.pkl", "wb") as f:
        pickle.dump(batch_tuples[train_index], f)
    with open(f"{fold_path}/validation_tuples.pkl", "wb") as f:
        pickle.dump(batch_tuples[val_index], f)

    # Load epoch start point if exists
    epoch_start_file = f"{fold_path}/epoch_start.pkl"
    if os.path.exists(epoch_start_file):
        with open(epoch_start_file, "rb") as f:
            start_epoch = pickle.load(f)
    else:
        start_epoch = current_epoch

    # Initialize minimum loss to a large value
    min_val_loss = float('inf')
    best_f1_score = 0
    best_epoch = 0

    # Initialize lists to store metrics
    train_losses = []
    train_f1_scores = []
    train_matrices = []
    val_losses = []
    val_f1_scores = []
    val_matrices = []
    
    

    # 타이머 시작
    start_time = time.time()
    cumulative_tp_train, cumulative_tn_train, cumulative_fp_train, cumulative_fn_train = 0, 0, 0, 0
    cumulative_tp_val, cumulative_tn_val, cumulative_fp_val, cumulative_fn_val = 0, 0, 0, 0
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()  # 각 epoch의 시작 시간을 기록합니다.
    
        training_tuples = batch_tuples[train_index]
        training_tuples = training_tuples[4:-4]
        val_tuples = batch_tuples[val_index]
        
        batches = list(get_batches(training_tuples, batch_size))
        print(f"Epoch {epoch + 1}/{epochs}")

        progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
        
        total_train_loss = 0.0
        tp_train, tn_train, fp_train, fn_train = 0, 0, 0, 0  # 누적할 변수 초기화
        
        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                video_data, wba_data, batch, frame_per_window
            )
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)

            loss, pred = trainer.step(batch_input_data, batch_target_data)

            recent_losses.append(loss.item())
            avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0

            batch_target_data_cpu = batch_target_data.cpu()
            predictions_cpu = pred.cpu()

            predicted_labels = (predictions_cpu >= 0).int().numpy()
            true_labels = batch_target_data_cpu.int().numpy()

            # Confusion matrix에서 각 값을 추출하여 누적
            conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
            
            tn, fp, fn, tp = conf_matrix.ravel()
            tp_train += tp
            tn_train += tn
            fp_train += fp
            fn_train += fn
            sensitivity_train = tp_train / (tp_train + fn_train)
            specificity_train = tn_train / (tn_train + fp_train)
            f1_train = 2 * tp_train / (2 * tp_train + fp_train + fn_train)
            progress_bar.set_postfix(
                loss=f"{loss.item():.5f}",
                avg_recent_loss=f"{avg_recent_loss:.5f}",
                lr=f"{trainer.lr:.7f}",
                f1_train=f"{f1_train:.3f}"
            )

            total_train_loss += loss.item()

            del batch_input_data, batch_target_data, loss, pred

        avg_train_loss = total_train_loss / len(batches)
        train_losses.append(avg_train_loss)

        # 누적된 값을 사용하여 최종적으로 F1 점수, 민감도, 특이도를 계산
        sensitivity_train = tp_train / (tp_train + fn_train)
        specificity_train = tn_train / (tn_train + fp_train)
        f1_train = 2 * (sensitivity_train * specificity_train) / (sensitivity_train + specificity_train)

        print(f"Training Sensitivity: {sensitivity_train:.5f}, Specificity: {specificity_train:.5f}, F1 Score: {f1_train:.5f}")

        # Training 성능 지표를 pickle 파일로 저장
        train_metrics = {
            'sensitivity': sensitivity_train,
            'specificity': specificity_train,
            'f1_score': f1_train,
            'loss': avg_train_loss
        }
        train_f1_scores.append(f1_train)
        train_matrices.append(conf_matrix)
        with open(f"{fold_path}/train_metrics.pkl", "wb") as f:
            pickle.dump(train_matrices, f)

        # Validation Phase
        val_batches = list(get_batches(val_tuples, batch_size))
        total_val_loss = 0.0
        
        tp_val, tn_val, fp_val, fn_val = 0, 0, 0, 0  # 누적할 변수 초기화
        
        progress_bar = tqdm(val_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=150)

        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                video_data, wba_data, batch, frame_per_window
            )
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)

            # Calculate test loss
            loss, pred = trainer.evaluate(batch_input_data, batch_target_data)

            progress_bar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{trainer.lr:.7f}")

            batch_target_data_cpu = batch_target_data.cpu()
            predictions_cpu = pred.cpu()

            predicted_labels = (predictions_cpu >= 0).int().numpy()
            true_labels = batch_target_data_cpu.int().numpy()

            # Confusion matrix에서 각 값을 추출하여 누적
            conf_matrix = confusion_matrix(true_labels, predicted_labels, labels=[0, 1])
            tn, fp, fn, tp = conf_matrix.ravel()
            tp_val += tp
            tn_val += tn
            fp_val += fp
            fn_val += fn

            total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_batches)
        val_losses.append(avg_val_loss)
        

        # 누적된 값을 사용하여 최종적으로 F1 점수, 민감도, 특이도를 계산
        sensitivity_val = tp_val / (tp_val + fn_val)
        specificity_val = tn_val / (tn_val + fp_val)
        f1_val = 2 * (sensitivity_val * specificity_val)/ (sensitivity_val + specificity_val)

        print(f"Validation Sensitivity: {sensitivity_val:.5f}, Specificity: {specificity_val:.5f}, F1 Score: {f1_val:.5f}")

        # Validation 성능 지표를 pickle 파일로 저장
        val_metrics = {
            'sensitivity': sensitivity_val,
            'specificity': specificity_val,
            'f1_score': f1_val,
            'loss': avg_val_loss
        }
        val_f1_scores.append(f1_val)
        val_matrices.append([tp_val, tn_val, fp_val, fn_val])
        with open(f"{fold_path}/val_metrics_epoch.pkl", "wb") as f:
            pickle.dump(val_metrics, f)

        # print(f"Average validation loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")
        # print(f"Validation F1 Score: {f1_test:.5f}")

        # Update metrics plot
        update_metrics_plot(fold_path, epoch, train_losses, train_f1_scores, val_losses, val_f1_scores)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save model if this epoch has the lowest test loss
        if f1_val > best_f1_score:
            best_f1_score = f1_val
            best_epoch = epoch + 1
            best_model_path = f"{fold_path}/best_model.ckpt"
            trainer.save(best_model_path, epoch)
            print(f"New best model saved at epoch {best_epoch} with f1 {f1_val:.5f}")

            
        if epoch == start_epoch:
            first_epoch_duration = time.time() - start_time
            print(f"First epoch took {first_epoch_duration:.2f} seconds.")

            # 전체 프로그램의 예상 종료 시간 계산
            total_duration = first_epoch_duration * epochs * fold_factor
            estimated_end_time = datetime.now(KST) + timedelta(seconds=total_duration)
            print(f"Estimated total program duration: {total_duration / 3600:.2f} hours")
            print(f"Estimated program end time (KST): {estimated_end_time.strftime('%Y/%m/%d %H:%M:%S')}")
            
        # Save intermediate results every 5 epochs
        if (epoch + 1) % 5 == 0:
            fig = plt.figure(figsize=(16, 12))
            intermediate_path = f"{fold_path}/intermediate_epoch"
            os.makedirs(intermediate_path, exist_ok=True)
            val_tuples = batch_tuples[val_index]
            selected_indices = np.random.choice(val_tuples.shape[0], size=9, replace=False)

            selected_val_tuples = val_tuples[selected_indices]
            batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                video_data, wba_data, selected_val_tuples, frame_per_window
            )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)

            _, predictions = trainer.evaluate(batch_input_data, batch_target_data)

            predictions_cpu = predictions.cpu().numpy()
            true_labels_cpu = batch_target_data.cpu().numpy()

            f1 = f1_score(true_labels_cpu, predictions_cpu >= 0, average='binary')

            save_test_result(batch_input_data, batch_target_data, predictions, f1_val, epoch, fold_path)

    print(f"Best model for fold {fold + 1} saved from epoch {best_epoch} with loss {min_val_loss:.5f}")
    all_fold_losses.append(min_val_loss)

    print(f"Final training loss: {train_losses[-1]:.5f}")
    print(f"Final test loss: {val_losses[-1]:.5f}")

# Save and print overall results
overall_result_path = f"{model_name}/overall_results"
os.makedirs(overall_result_path, exist_ok=True)

with open(f"{overall_result_path}/fold_losses.pkl", "wb") as f:
    pickle.dump(all_fold_losses, f)

average_loss = np.mean(all_fold_losses)
print(f"All fold val losses: {all_fold_losses}")
print(f"Average val loss: {average_loss:.5f}")

with open(f"{overall_result_path}/average_loss.txt", "w") as f:
    f.write(f"All fold val losses: {all_fold_losses}\n")
    f.write(f"Average val loss: {average_loss:.5f}\n")