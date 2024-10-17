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

for piece_size in [1, 5, 10, 20, 30, 40]:
    h = 360
    w = 720
    c = 1
    fps = 30
    downsampling_factor = 5.625

    frame_per_window = 8
    frame_per_sliding = 8
    input_ch = 1

    model_string = f"only_forest_predict_wba_diff_random_val_optic_3layers_non_fixed_apply_piece_size_{piece_size}"
    model_string += f"_{frame_per_window}frames"

    folder_path = "./naturalistic"
    mat_file_name = f"experimental_data.mat"
    checkpoint_name = "fly_model"

    model_name = f"./model/{model_string}"
    if os.path.exists(model_name):
        for i in range(1,100):
            model_name = f"./model/{model_string}_{i}"
            if not os.path.exists(model_name):
                os.makedirs(model_name)
                break
            else:
                continue
    else:
        os.makedirs(model_name)

    result_save_path = f"./model/{model_string}/result_data.h5"

    pretrained_model_path = "./pretrained_model/64_to_256_3layers.ckpt"

    # hyperparameter 
    batch_size = 20
    lr = 1e-4
    epochs = 100
    fold_factor = 5

    layer_configs = [[64, 2], [128, 2], [256, 2]]#, [512, 2]]

    video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
    video_data, wba_data, aug_factor = aug_videos(video_data, wba_data)

    print(f"augmented shape : {video_data.shape}")
    print(f"augmented shape : {wba_data.shape}")
    wba_data = wba_data

    # Split period and split for training / test data set
    recent_losses = deque(maxlen=100)
    recent_f1_scores = deque(maxlen=100)
    val_losses_per_epoch = []

    batch_tuples = np.array(generate_tuples_direction_pred(total_frame, frame_per_window, frame_per_sliding, video_data.shape[0]))
    #kf = KFold(n_splits=fold_factor, random_state=42, shuffle=True)
    fold_set_list = []
    for i in range(fold_factor):
        train_idx, val_idx = split_train_val_index(batch_tuples, aug_factor, piece_size=piece_size, val_ratio=0.2)
        fold_set_list.append((train_idx, val_idx))
        

    all_fold_losses = []


    KST = pytz.timezone('Asia/Seoul')
    for fold, (train_index, val_index) in enumerate(fold_set_list):
        print(f"Fold {fold+1}")
        
        fold_path = f"{model_name}/fold_{fold+1}"
        
        

        # create model
        flownet_model = flownet3d(layer_configs, num_classes=2)
        flownet_model = load_model(flownet_model, pretrained_model_path)
        model = FlowNet3DWithFeatureExtraction(flownet_model, feature_dim=128, 
                                            input_size=(frame_per_window, 
                                                        int(h//downsampling_factor), 
                                                        int(w//downsampling_factor), 
                                                        1))
        trainer = Trainer(model, loss_function_mse, lr)
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
        
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()  # 각 epoch의 시작 시간을 기록합니다.

            training_tuples = batch_tuples[train_index]
            
            val_tuples = batch_tuples[val_index]
            val_tuples = [tup for tup in val_tuples if tup[0] == 2 * aug_factor]

            batches = list(get_batches(training_tuples, batch_size))
            print(f"Epoch {epoch + 1}/{epochs}")

            progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=150)
            
            total_train_loss = 0.0
            
            for batch in progress_bar:
                batch_input_data, batch_target_data = get_data_from_batch_direction_pred(
                    video_data, wba_data, batch, frame_per_window
                )
                batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
                batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)

                loss, pred = trainer.step(batch_input_data, batch_target_data)

                recent_losses.append(loss.item())
                avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0

                progress_bar.set_postfix(
                    loss=f"{loss.item():.5f}",
                    avg_recent_loss=f"{avg_recent_loss:.5f}",
                    lr=f"{trainer.lr:.7f}"
                )

                total_train_loss += loss.item()

                del batch_input_data, batch_target_data, loss, pred

            avg_train_loss = total_train_loss / len(batches)
            train_losses.append(avg_train_loss)

            # Validation Phase
            val_batches = list(get_batches(val_tuples, batch_size))
            total_val_loss = 0.0
            val_predictions = []
            
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

                # (frame, prediction) 형태로 저장
                for i, (video_num, start_frame) in enumerate(batch):
                    val_predictions.append((start_frame, predictions_cpu[i].item()))

                total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_batches)
            val_losses.append(avg_val_loss)
            
            print(f"Training loss: {avg_train_loss:.5f}")
            print(f"Validation loss: {avg_val_loss:.5f}")
            
            update_metrics_plot(fold_path, epoch, train_losses, val_losses)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Save model if this epoch has the lowest test loss
            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_path = f"{fold_path}/best_model.ckpt"
                trainer.save(best_model_path, epoch)
                print(f"New best model saved at epoch {best_epoch} with loss {avg_val_loss:.5f}")

            if epoch == start_epoch:
                first_epoch_duration = time.time() - start_time
                print(f"First epoch took {first_epoch_duration:.2f} seconds.")

                # 전체 프로그램의 예상 종료 시간 계산
                total_duration = first_epoch_duration * epochs * fold_factor
                estimated_end_time = datetime.now(KST) + timedelta(seconds=total_duration)
                print(f"Estimated total program duration: {total_duration / 3600:.2f} hours")
                print(f"Estimated program end time (KST): {estimated_end_time.strftime('%Y/%m/%d %H:%M:%S')}")

            # 5 epoch마다 그래프 플로팅 및 저장
            if (epoch + 1) % 5 == 0:
                plt.figure(figsize=(10, 6))
                
                # wba_data를 window size만큼 생략하고 플로팅
                #diff_wba_for_plotting = [ wba_data[ 2*aug_factor, frame_per_window * (i+1) ] - wba_data[2*aug_factor, frame_per_window * i ] for i in range(len(wba_data[2*aug_factor]) // frame_per_window - 1) ]
                #plt.plot(np.array(range(len(diff_wba_for_plotting)))+1, diff_wba_for_plotting, label='WBA Data', color='blue')
                plt.plot(np.array(range(0,len(wba_data[2*aug_factor])))+1, wba_data[2*aug_factor], label='WBA Data', color='blue')
                
                # Validation prediction 결과를 다른 색으로 점으로 플로팅
                val_frames, val_preds = zip(*val_predictions)
                plt.scatter(np.array(val_frames), val_preds, color='red', s=6, label='Validation Predictions')
                
                plt.xlabel('Frame')
                plt.ylabel('WBA Value')
                plt.title(f'Validation Predictions vs WBA Data at Epoch {epoch + 1}')
                plt.legend()
                
                intermediate_path = f"{fold_path}/intermediate_epoch"
                os.makedirs(intermediate_path, exist_ok=True)
                plt.savefig(f"{intermediate_path}/{epoch + 1}.png")
                plt.close()

        print(f"Best model for fold {fold + 1} saved from epoch {best_epoch} with f1 {min_val_loss:.5f}")
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