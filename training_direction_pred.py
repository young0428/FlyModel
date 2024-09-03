#%%
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

torch.autograd.set_detect_anomaly(True)

h = 360
w = 720
c = 1
fps = 30
downsampling_factor = 5.625

frame_per_window = 16
frame_per_sliding = 4
input_ch = 1 

model_string = "direction_pred"
model_string += f"_{frame_per_window}frames_"

folder_path = "./naturalistic"
mat_file_name = f"experimental_data.mat"
checkpoint_name = "fly_model"

model_name = f"./model/{model_string}"
os.makedirs(model_name, exist_ok=True)
result_save_path = f"./model/{model_string}/result_data.h5"

pretrained_model_path = "./pretrained_model/64x128_opticflow_64t51216frames.ckpt"

# hyperparameter 
batch_size = 10
lr = 1e-3
epochs = 100
fold_factor = 8

layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]

video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
video_data = aug_videos(video_data)
print(f"augmented shape : {video_data.shape}")

# Split period and split for training / test data set
recent_losses = deque(maxlen=100)
test_losses_per_epoch = []


flownet_model = flownet3d(layer_configs, num_classes=1)
flownet_model = load_model(flownet_model, pretrained_model_path)

batch_tuples = np.array(generate_tuples_direction_pred(total_frame, frame_per_window, frame_per_sliding, video_data.shape[0]))
kf = KFold(n_splits=fold_factor)

all_fold_losses = []
#%%


#%%
for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):

    print(f"Fold {fold+1}")
    fold_path = f"{model_name}/fold_{fold+1}"

    # create model
    model = FlowNet3DWithFeatureExtraction(flownet_model, feature_dim = 128)
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
    min_test_loss = float('inf')
    best_epoch = 0

    for epoch in range(start_epoch, epochs):
        
        training_tuples = batch_tuples[train_index]
        training_tuples = training_tuples[4:-4]
        val_tuples = batch_tuples[val_index]
        
        batches = list(get_batches(training_tuples, batch_size))
        print(f"Epoch {epoch + 1}/{epochs}")

        progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
        
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0
        
        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                video_data, wba_data, batch, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            

            loss, pred = trainer.step(batch_input_data, batch_target_data)
            
            
            recent_losses.append(loss.item())
            avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            progress_bar.set_postfix(loss=f"{loss.item():.5f}", avg_recent_loss=f"{avg_recent_loss:.5f}", lr=f"{trainer.lr:.7f}")
            
            del batch_input_data, batch_target_data, loss, pred
        
        trainer.save(f"{fold_path}/{checkpoint_name}.ckpt", epoch)
        
        # Save the current epoch to resume later if needed
        with open(epoch_start_file, "wb") as f:
            pickle.dump(epoch + 1, f)
        
        # validation phase after each epoch
        val_batches = list(get_batches(val_tuples, batch_size))
        total_test_loss = 0.0
        
        progress_bar = tqdm(val_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=150)

        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                video_data, wba_data, batch, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            
            # Calculate test loss
            loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
            
            total_test_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{trainer.lr:.7f}")

        avg_test_loss = total_test_loss / len(val_batches)
        test_losses_per_epoch.append(avg_test_loss)
        print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")
        batch_target_data_cpu = batch_target_data.cpu()
        predictions_cpu = pred.cpu()

        # 2. logits 값을 기준으로 0 이상은 1, 0 이하는 0으로 변환
        predicted_labels = (predictions_cpu >= 0).float()
        true_labels = (batch_target_data_cpu >= 0).float()

        # 3. 정확도 계산
        correct_predictions = (predicted_labels == true_labels).float()
        accuracy = correct_predictions.sum() / correct_predictions.numel()

        print(f"Test Accuracy: {accuracy.item() * 100:.2f}%")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save model if this epoch has the lowest test loss
        if avg_test_loss < min_test_loss:
            min_test_loss = avg_test_loss
            best_epoch = epoch + 1
            best_model_path = f"{fold_path}/best_model.ckpt"
            trainer.save(best_model_path, epoch)
            print(f"New best model saved at epoch {best_epoch} with loss {min_test_loss:.5f}")

        # Save intermediate results 
        if (epoch + 1) % 5 == 0:
            fig = plt.figure(figsize=(16,12))
            intermediate_path = f"{fold_path}/intermediate_epoch"
            os.makedirs(intermediate_path, exist_ok=True)
            all_predictions = []
            all_targets = []
            val_tuples = batch_tuples[val_index]
            selected_indices = np.random.choice(val_tuples.shape[0], size=9, replace=False)
            
            selected_val_tuples = val_tuples[selected_indices]
            batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                video_data, wba_data, selected_val_tuples, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            
            _, predictions = trainer.evaluate(batch_input_data, batch_target_data)
            
            # 1. 데이터를 CPU로 이동
            batch_target_data_cpu = batch_target_data.cpu()
            predictions_cpu = predictions.cpu()

            # 2. logits 값을 기준으로 0 이상은 1, 0 이하는 0으로 변환
            predicted_labels = (predictions_cpu >= 0).float()
            true_labels = (batch_target_data_cpu >= 0).float()

            # 3. 정확도 계산
            correct_predictions = (predicted_labels == true_labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()

            print(f"Accuracy: {accuracy.item() * 100:.2f}%")
            
            save_test_result(batch_input_data, batch_target_data, predictions, accuracy, epoch, fold_path)
                
            
            

    print(f"Best model for fold {fold + 1} saved from epoch {best_epoch} with loss {min_test_loss:.5f}")
    all_fold_losses.append(min_test_loss)

# Save and print overall results
overall_result_path = f"{model_name}/overall_results"
os.makedirs(overall_result_path, exist_ok=True)

with open(f"{overall_result_path}/fold_losses.pkl", "wb") as f:
    pickle.dump(all_fold_losses, f)

average_loss = np.mean(all_fold_losses)
print(f"All fold test losses: {all_fold_losses}")
print(f"Average test loss: {average_loss:.5f}")

with open(f"{overall_result_path}/average_loss.txt", "w") as f:
    f.write(f"All fold test losses: {all_fold_losses}\n")
    f.write(f"Average test loss: {average_loss:.5f}\n")