#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from p3d_resnet import *
from trainer_func import Trainer
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

frame_per_window = 8
frame_per_sliding = 4
input_ch = 1 

model_string = "8fold_3dresnet50_categorical"
model_string += f"{frame_per_window}frames_"

folder_path = "./naturalistic"
mat_file_name = "experimental_data.mat"
checkpoint_name = "fly_model"

model_name = f"./model/{model_string}"
os.makedirs(model_name, exist_ok=True)
result_save_path = f"./model/{model_string}/result_data.h5"

# hyperparameter 
batch_size = 10
lr = 1e-3
epochs = 300
fold_factor = 8

# p3d renet18 spec
# block_list = [[64, 2], [128, 2], [256, 2], [512, 2]]
# feature_output_dims = 400

video_data, _, wba_data, total_frame = load_filtered_diff_data(folder_path, mat_file_name, downsampling_factor)

# Split period and split for training / test data set
recent_losses = deque(maxlen=100)
test_losses_per_epoch = []

batch_tuples = np.array(generate_tuples_sac(total_frame, frame_per_window, frame_per_sliding, 3))
kf = KFold(n_splits=fold_factor)

all_fold_losses = []
#%%


#%%
train_losses_per_epoch_file = f"{model_name}/train_losses_per_epoch.pkl"
train_accuracies_per_epoch_file = f"{model_name}/train_accuracies_per_epoch.pkl"

test_losses_per_epoch_file = f"{model_name}/test_losses_per_epoch.pkl"
test_accuracies_per_epoch_file = f"{model_name}/test_accuracies_per_epoch.pkl"

if os.path.exists(train_losses_per_epoch_file):
    with open(train_losses_per_epoch_file, "rb") as f:
        train_losses_per_epoch = pickle.load(f)
else:
    train_losses_per_epoch = []

if os.path.exists(train_accuracies_per_epoch_file):
    with open(train_accuracies_per_epoch_file, "rb") as f:
        train_accuracies_per_epoch = pickle.load(f)
else:
    train_accuracies_per_epoch = []

if os.path.exists(test_losses_per_epoch_file):
    with open(test_losses_per_epoch_file, "rb") as f:
        test_losses_per_epoch = pickle.load(f)
else:
    test_losses_per_epoch = []

if os.path.exists(test_accuracies_per_epoch_file):
    with open(test_accuracies_per_epoch_file, "rb") as f:
        test_accuracies_per_epoch = pickle.load(f)
else:
    test_accuracies_per_epoch = []

#%%

for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):
    if not fold < 4:
        continue

    print(f"Fold {fold+1}")
    fold_path = f"{model_name}/fold_{fold+1}"

    # create model
    #model = p3d_resnet(input_ch, block_list, feature_output_dims)
    model = resnet3d50(num_classes=3)
    trainer = Trainer(model, loss_function, lr)
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
            batch_input_data, batch_target_data = get_data_from_batch_diff_cat(
                video_data, wba_data, batch, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            
            loss, pred = trainer.step(batch_input_data, batch_target_data)
            batch_target_indices = torch.argmax(batch_target_data, dim=1)
            # Calculate training accuracy
            _, predicted = torch.max(pred, 1)
            predicted = predicted.to(trainer.device)
            correct = (predicted == batch_target_indices).sum().item()
            total_train_correct += correct
            total_train_samples += batch_target_data.size(0)
            train_accuracy = total_train_correct / total_train_samples
            
            total_train_loss += loss.item()
            recent_losses.append(loss.item())
            avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0
            
            progress_bar.set_postfix(loss=f"{loss.item():.5f}", avg_recent_loss=f"{avg_recent_loss:.5f}", lr=f"{trainer.lr:.7f}", accuracy=f"{train_accuracy:.3f}")
            
            del batch_input_data, batch_target_data, loss, pred
        
        # Epoch 끝난 후, 평균 loss와 accuracy 계산
        avg_train_loss = total_train_loss / len(batches)
        train_losses_per_epoch.append(avg_train_loss)
        
        avg_train_accuracy = total_train_correct / total_train_samples
        train_accuracies_per_epoch.append(avg_train_accuracy)

        # validation phase after each epoch
        val_batches = list(get_batches(val_tuples, batch_size))
        total_test_loss = 0.0
        total_test_correct = 0
        total_test_samples = 0
        progress_bar = tqdm(val_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=150)

        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_diff_cat(
                video_data, wba_data, batch, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            
            # Calculate test accuracy
            loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
            batch_target_indices = torch.argmax(batch_target_data, dim=1)
            # 테스트 정확도 계산
            _, predicted = torch.max(pred, 1)
            predicted = predicted.to(trainer.device)
            correct = (predicted == batch_target_indices).sum().item()
            total_test_correct += correct
            total_test_samples += batch_target_data.size(0)
            test_accuracy = total_test_correct / total_test_samples
            
            total_test_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{trainer.lr:.7f}", accuracy=f"{test_accuracy:.3f}")
        
        # Epoch 끝난 후, test loss와 accuracy 계산
        avg_test_loss = total_test_loss / len(val_batches)
        test_losses_per_epoch.append(avg_test_loss)
        print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")
        print(f"Test accuracy after Epoch {epoch + 1}: {test_accuracy:.3f}")

        avg_test_loss = total_test_loss / len(val_batches)
        test_losses_per_epoch.append(avg_test_loss)
        
        avg_test_accuracy = total_test_correct / total_test_samples
        test_accuracies_per_epoch.append(avg_test_accuracy)

        # Training 및 Test 결과 플롯 갱신 및 저장
        plt.figure(figsize=(10,10))
        
        # Training Loss
        plt.subplot(2, 2, 1)
        plt.plot(train_losses_per_epoch, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Training Accuracy
        plt.subplot(2, 2, 2)
        plt.plot(train_accuracies_per_epoch, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Test Loss
        plt.subplot(2, 2, 3)
        plt.plot(test_losses_per_epoch, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Test Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(test_accuracies_per_epoch, label='Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{fold_path}/training_testing_progress.png")
        plt.close()

        # Loss와 accuracy 리스트를 파일로 저장
        with open(train_losses_per_epoch_file, "wb") as f:
            pickle.dump(train_losses_per_epoch, f)

        with open(train_accuracies_per_epoch_file, "wb") as f:
            pickle.dump(train_accuracies_per_epoch, f)

        with open(test_losses_per_epoch_file, "wb") as f:
            pickle.dump(test_losses_per_epoch, f)

        with open(test_accuracies_per_epoch_file, "wb") as f:
            pickle.dump(test_accuracies_per_epoch, f)

        trainer.save(f"{fold_path}/{checkpoint_name}.ckpt", epoch)
        
        # Save the current epoch to resume later if needed
        with open(epoch_start_file, "wb") as f:
            pickle.dump(epoch + 1, f)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Save intermediate results every 10 epochs
        if (epoch + 1) % 10 == 0:
            fig = plt.figure(figsize=(16,12))
            ax = fig.add_subplot(1,1,1)
            intermediate_path = f"{fold_path}/intermediate_epoch"
            os.makedirs(intermediate_path, exist_ok=True)
            all_predictions = []
            all_targets = []
            val_tuples = batch_tuples[val_index]
            val_start_frame = val_tuples[0][1]
            val_end_frame = val_tuples[-1][1]
            predictions = predict_with_model_wba(trainer, video_data, val_start_frame, val_end_frame, frame_per_window, fps)
            target = wba_data[val_tuples[0][0],val_start_frame:val_end_frame]
            
            predicted = torch.tensor(predictions).argmax(dim=-1)
            predicted = predicted.to(trainer.device)
            batch_target_indices = torch.argmax(batch_target_data, dim=1)
            correct = (predicted == batch_target_indices).sum().item()
            total_test_correct += correct
            total_test_samples += batch_target_data.size(0)
            test_accuracy = total_test_correct / total_test_samples

            ax.plot(range(val_start_frame, val_end_frame), target[:], label='Target', color='red')
            ax.plot(range(val_start_frame, val_end_frame), predictions[:], label='Prediction', color='blue')
            ax.legend()
            plt.title(f"Accuracy: {test_accuracy:.3f}")
            plt.tight_layout()
            plt.savefig(f"{intermediate_path}/results_epoch_{epoch + 1}.png")
            plt.close()
            
    # End of fold operations
    all_fold_losses.append(avg_test_loss)

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