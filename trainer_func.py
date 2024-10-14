import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from FlowNet import *
from LoadDataset import *
from sklearn.metrics import confusion_matrix
import h5py
from tqdm import tqdm
import time
from datetime import datetime, timedelta

class Trainer :
    def __init__(self, model, loss_func, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device : {self.device}")
        self.step_counter = 0
        self.loss_sum = 0
        self.loss_func = loss_func
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, threshold=0.005, min_lr=1e-6)
    
    def save(self, path, epoch):
        save_state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(save_state, path)

    def load(self, path):
        epoch = 0
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            epoch = checkpoint['epoch']
            print(f"load model")
            print(f"epoch : {epoch}")
            print(f"learning rate : {self.optimizer.param_groups[0]['lr']}")
        else:
            print(f"Model not found at {path}")
        
        return epoch
        
    def step(self, input, target):
        self.step_counter += 1
        input = input.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad()
        
        pred = self.model(input)
        
        loss = self.loss_func(pred, target)
        self.loss_sum += loss.item()
        if self.step_counter % 500 == 0:
            avg_loss = self.loss_sum / 100
            self.loss_sum = 0
            self.scheduler.step(avg_loss)
            self.lr = self.optimizer.param_groups[0]['lr']
        
        loss.backward()
        self.optimizer.step()
        return loss, pred
    
    
    def evaluate(self, input, target):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            target = target.to(self.device)
            pred = self.model(input)
            loss = self.loss_func(pred, target)
        self.model.train()
        return loss, pred
    
    def pred(self, input):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            pred = self.model(input)
        self.model.train()
        return pred

# 모델 로드 함수 정의
def load_model(model, path):
    if os.path.exists(path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded successfully from {path}")
        return model
    else:
        print(f"Model not found at {path}")
        return None

    
def save_test_result(batch_input_data, batch_target_data, predictions, f1_score, epoch, fold_path ):
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))
    
    fig.suptitle(f'f1_score: {f1_score:.2f}', fontsize=16)
    def update(frame_idx):
        for i in range(9):
            axes[i//3,i%3].clear()
            # Extract the frame for the current batch
            frame = batch_input_data[i, frame_idx, :, :, 0].cpu().numpy()
            
            # Plot the frame
            axes[i//3,i%3].imshow(frame, cmap='gray')
            axes[i//3,i%3].axis('off')
            
            # Add the prediction label
            prediction_label = 'left' if predictions[i].item() > 0 else 'right'
            target_label = 'left' if batch_target_data[i].item() > 0 else 'right'
            axes[i//3,i%3].set_title(f"Batch {i}: \npred : {prediction_label}\ntarget : {target_label}")
        
    ani = animation.FuncAnimation(fig, update, frames=16, interval=200)    
    intermediate_path = f"{fold_path}/intermediate_epoch"
    ani.save(f'{intermediate_path}/{epoch+1}.gif', writer='PillowWriter')
    plt.close()
    
def update_confusion_matrix_file(fold_path, epoch, confusion_matrices, mode='train'):
    """
    confusion matrix를 HDF5 파일로 저장합니다. 
    'mode'는 train 또는 val을 나타냅니다.
    """
    filename = f'{fold_path}/{mode}_confusion_matrices.h5'
    with h5py.File(filename, 'a') as f:
        grp = f.create_group(f"epoch_{epoch+1}")
        grp.create_dataset('confusion_matrix', data=confusion_matrices)
    
def update_metrics_plot(fold_path, epoch, train_losses, test_losses):
    
    plt.figure(figsize=(12, 8))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


    plt.tight_layout()
    plt.savefig(f'{fold_path}/metrics_plot.png')
    plt.close()
    
# # 모델 및 트레이너 초기화 함수
# def initialize_model(layer_configs, pretrained_model_path):
#     flownet_model = flownet3d(layer_configs, num_classes=1)
#     flownet_model = load_model(flownet_model, pretrained_model_path)
#     model = FlowNet3DWithFeatureExtraction(flownet_model, feature_dim=128)
#     return model

# def initialize_trainer(model, lr, fold_path, checkpoint_name):
#     trainer = Trainer(model, loss_function_bce, lr)
#     current_epoch = trainer.load(f"{fold_path}/{checkpoint_name}.ckpt")
#     return trainer, current_epoch

# # 시간 출력 함수
# def print_estimated_time(start_time, epoch, fold_factor, epochs):
#     elapsed_time = time.time() - start_time
#     first_epoch_duration = elapsed_time / (epoch + 1) if epoch > 0 else elapsed_time
#     total_duration = first_epoch_duration * epochs * fold_factor
#     estimated_end_time = datetime.now() + timedelta(seconds=total_duration)
    
#     print(f"Estimated total program duration: {total_duration / 3600:.2f} hours")
#     print(f"Estimated program end time: {estimated_end_time.strftime('%Y/%m/%d %H:%M:%S')}")

# # Epoch 진행 함수
# def train_epoch(trainer, batches, video_data, wba_data, frame_per_window, progress_bar):
#     total_train_loss = 0.0
#     tp_train, tn_train, fp_train, fn_train = 0, 0, 0, 0
#     for batch in progress_bar:
#         batch_input_data, batch_target_data = get_data_from_batch_direction_pred(video_data, wba_data, batch, frame_per_window)
#         batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
#         batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)

#         loss, pred = trainer.step(batch_input_data, batch_target_data)
#         total_train_loss += loss.item()

#         predicted_labels = (pred.cpu() >= 0).int().numpy()
#         true_labels = batch_target_data.cpu().int().numpy()

#         tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
#         tp_train += tp
#         tn_train += tn
#         fp_train += fp
#         fn_train += fn

#     avg_train_loss = total_train_loss / len(batches)
#     return avg_train_loss, tp_train, tn_train, fp_train, fn_train

# # Validation 진행 함수
# def validate_epoch(trainer, val_batches, video_data, wba_data, frame_per_window):
#     total_val_loss = 0.0
#     tp_val, tn_val, fp_val, fn_val = 0, 0, 0, 0
#     for batch in val_batches:
#         batch_input_data, batch_target_data = get_data_from_batch_direction_pred(video_data, wba_data, batch, frame_per_window)
#         batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
#         batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)

#         loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
#         total_val_loss += loss.item()

#         predicted_labels = (pred.cpu() >= 0).int().numpy()
#         true_labels = batch_target_data.cpu().int().numpy()

#         tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
#         tp_val += tp
#         tn_val += tn
#         fp_val += fp
#         fn_val += fn

#     avg_val_loss = total_val_loss / len(val_batches)
#     return avg_val_loss, tp_val, tn_val, fp_val, fn_val

# # 지표 계산 함수
# def calculate_metrics(tp, tn, fp, fn):
#     sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
#     specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
#     f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
#     return sensitivity, specificity, f1

# # 지표 저장 함수
# def save_metrics(fold_path, epoch, metrics, is_train=True):
#     metrics_type = 'train' if is_train else 'val'
#     metrics_filename = f"{fold_path}/{metrics_type}_metrics_epoch_{epoch + 1}.pkl"
#     with open(metrics_filename, "wb") as f:
#         pickle.dump(metrics, f)

# # Plot 업데이트 함수 (5번마다 Plot 생성)
# def update_metrics_plot(fold_path, epoch, train_losses, train_f1_scores, val_losses, val_f1_scores):

#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label='Train Loss')
#     plt.plot(val_losses, label='Validation Loss')
#     plt.plot(train_f1_scores, label='Train F1 Score')
#     plt.plot(val_f1_scores, label='Validation F1 Score')
#     plt.legend()
#     plt.title(f"Metrics after {epoch + 1} Epochs")
#     plt.savefig(f"{fold_path}/metrics_epoch_{epoch + 1}.png")
#     plt.close()

# # 모델 저장 함수 (최소 Validation Loss일 때)
# def save_best_model(trainer, epoch, fold_path, avg_val_loss, min_val_loss):
#     if avg_val_loss < min_val_loss:
#         best_model_path = f"{fold_path}/best_model.ckpt"
#         trainer.save(best_model_path, epoch)
#         print(f"New best model saved at epoch {epoch + 1} with loss {avg_val_loss:.5f}")
#         return avg_val_loss
#     return min_val_loss

# # 모델 학습 함수
# def train_model(fold, train_index, val_index, batch_tuples, video_data, wba_data, fold_path, epochs, trainer, frame_per_window, batch_size, fold_factor):
#     start_time = time.time()
#     min_val_loss = float('inf')

#     train_losses, val_losses = [], []
#     train_f1_scores, val_f1_scores = [], []

#     for epoch in range(epochs):
#         print(f"Epoch {epoch + 1}/{epochs}")
        
#         train_batches = list(get_batches(batch_tuples[train_index], batch_size))
#         val_batches = list(get_batches(batch_tuples[val_index], batch_size))

#         progress_bar = tqdm(train_batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
#         avg_train_loss, tp_train, tn_train, fp_train, fn_train = train_epoch(
#             trainer, train_batches, video_data, wba_data, frame_per_window, progress_bar
#         )

#         # Train metrics 계산 및 저장
#         sensitivity_train, specificity_train, f1_train = calculate_metrics(tp_train, tn_train, fp_train, fn_train)
#         train_losses.append(avg_train_loss)
#         train_f1_scores.append(f1_train)
#         save_metrics(fold_path, epoch, {'sensitivity': sensitivity_train, 'specificity': specificity_train, 'f1_score': f1_train}, is_train=True)

#         # Validation 진행
#         avg_val_loss, tp_val, tn_val, fp_val, fn_val = validate_epoch(
#             trainer, val_batches, video_data, wba_data, frame_per_window
#         )
#         sensitivity_val, specificity_val, f1_val = calculate_metrics(tp_val, tn_val, fp_val, fn_val)
#         val_losses.append(avg_val_loss)
#         val_f1_scores.append(f1_val)
#         save_metrics(fold_path, epoch, {'sensitivity': sensitivity_val, 'specificity': specificity_val, 'f1_score': f1_val}, is_train=False)

#         # 모델 저장 (Validation loss가 최소일 때)
#         min_val_loss = save_best_model(trainer, epoch, fold_path, avg_val_loss, min_val_loss)

#         if epoch == 0:
#             print_estimated_time(start_time, epoch, fold_factor, epochs)
        
#         # 5 epoch 마다 plot 업데이트
#         if (epoch + 1) % 5 == 0:
#             update_metrics_plot(fold_path, epoch, train_losses, train_f1_scores, val_losses, val_f1_scores)

#         # 시간 출력 (첫 epoch 완료 후)
        

#     return min_val_loss

# # 전체 fold 결과 요약 함수
# def summarize_folds(all_fold_losses, model_name):
#     average_loss = np.mean(all_fold_losses)
#     print(f"All fold val losses: {all_fold_losses}")
#     print(f"Average val loss: {average_loss:.5f}")
    
#     overall_result_path = f"{model_name}/overall_results"
#     os.makedirs(overall_result_path, exist_ok=True)

#     with open(f"{overall_result_path}/fold_losses.pkl", "wb") as f:
#         pickle.dump(all_fold_losses, f)

#     with open(f"{overall_result_path}/average_loss.txt", "w") as f:
#         f.write(f"All fold val losses: {all_fold_losses}\n")
#         f.write(f"Average val loss: {average_loss:.5f}\n")


