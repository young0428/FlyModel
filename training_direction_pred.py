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

torch.autograd.set_detect_anomaly(True)

h = 360
w = 720
c = 1
fps = 30
downsampling_factor = 5.625

frame_per_window = 16
frame_per_sliding = 4
input_ch = 1 

model_string = "only_forest_ud_preX"
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
video_data = aug_videos(video_data)
print(f"augmented shape : {video_data.shape}")

# Split period and split for training / test data set
recent_losses = deque(maxlen=100)
recent_accuracies = deque(maxlen=100)
test_losses_per_epoch = []

flownet_model = flownet3d(layer_configs, num_classes=1)
#flownet_model = load_model(flownet_model, pretrained_model_path)

batch_tuples = np.array(generate_tuples_direction_pred(total_frame, frame_per_window, frame_per_sliding, video_data.shape[0]))
print(np.shape(batch_tuples))
kf = KFold(n_splits=fold_factor)

all_fold_losses = []

def update_metrics_plot(fold_path, epoch, train_losses, train_accuracies, test_losses, test_accuracies):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{fold_path}/metrics_plot.png')
    plt.close()

for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):
    print(f"Fold {fold+1}")
    fold_path = f"{model_name}/fold_{fold+1}"

    # create model
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
    min_test_loss = float('inf')
    best_epoch = 0

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    # 타이머 시작
    start_time = time.time()

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

            batch_target_data_cpu = batch_target_data.cpu()
            predictions_cpu = pred.cpu()

            predicted_labels = (predictions_cpu >= 0).int()
            true_labels = batch_target_data_cpu.int()

            correct_predictions = (predicted_labels == true_labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()

            recent_accuracies.append(accuracy.item())

            avg_recent_accuracy = sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0

            progress_bar.set_postfix(
                loss=f"{loss.item():.5f}",
                avg_recent_loss=f"{avg_recent_loss:.5f}",
                lr=f"{trainer.lr:.7f}",
                avg_accuracy=f"{avg_recent_accuracy:.5f}"
            )

            total_train_loss += loss.item()
            total_train_correct += correct_predictions.sum().item()
            total_train_samples += correct_predictions.numel()

            del batch_input_data, batch_target_data, loss, pred

        avg_train_loss = total_train_loss / len(batches)
        avg_train_accuracy = total_train_correct / total_train_samples
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_accuracy)

        trainer.save(f"{fold_path}/{checkpoint_name}.ckpt", epoch)

        # Save the current epoch to resume later if needed
        with open(epoch_start_file, "wb") as f:
            pickle.dump(epoch + 1, f)

        # validation phase after each epoch
        val_batches = list(get_batches(val_tuples, batch_size))
        total_test_loss = 0.0
        total_test_acc = 0.0
        it = 0

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

            predicted_labels = (predictions_cpu >= 0).int()
            true_labels = batch_target_data_cpu.int()

            correct_predictions = (predicted_labels == true_labels).float()
            accuracy = correct_predictions.sum() / correct_predictions.numel()
            total_test_loss += loss.item()
            total_test_acc += accuracy.item()

            it += 1

        avg_test_loss = total_test_loss / it
        avg_total_acc = total_test_acc / it

        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_total_acc)

        print(f"Average validation loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")
        print(f"Validation Accuracy: {avg_total_acc * 100:.2f}%")

        # Update metrics plot
        update_metrics_plot(fold_path, epoch, train_losses, train_accuracies, test_losses, test_accuracies)

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
            fig = plt.figure(figsize=(16, 12))
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

            save_test_result(batch_input_data, batch_target_data, predictions, avg_total_acc, epoch, fold_path)

        # 첫 번째 epoch이 끝났을 때 소요 시간 계산
        if epoch == start_epoch:
            first_epoch_duration = time.time() - start_time
            print(f"First epoch took {first_epoch_duration:.2f} seconds.")

            # 전체 epoch와 fold의 예상 종료 시간 계산
            total_duration = first_epoch_duration * (epochs * fold_factor)
            estimated_end_time = datetime.now() + timedelta(seconds=total_duration)
            print(f"Estimated end time: {estimated_end_time.strftime('%Y/%m/%d %H:%M')}")

    print(f"Best model for fold {fold + 1} saved from epoch {best_epoch} with loss {min_test_loss:.5f}")
    all_fold_losses.append(min_test_loss)

    print(f"Final training loss: {train_losses[-1]:.5f}")
    print(f"Final training accuracy: {train_accuracies[-1]:.5f}")
    print(f"Final test loss: {test_losses[-1]:.5f}")
    print(f"Final test accuracy: {test_accuracies[-1]:.5f}")

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