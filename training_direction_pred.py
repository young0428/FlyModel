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
import time
import pytz
import warnings

from make_visualization_video import *
from sklearn.model_selection import KFold
from datetime import datetime, timedelta
from sklearn.metrics import f1_score, confusion_matrix  # confusion_matrix import

warnings.filterwarnings("ignore", category=UserWarning, message="dropout2d: Received a 5-D input")

torch.autograd.set_detect_anomaly(True)

def training_direction_pred(model_folder_name,
                            model_class, 
                            video_indices = [2], 
                            piece_sizes = [1, 5, 10, 20, 40], 
                            frame_size = 8,
                            validation_ratio = 0.3,
                            use_pretrained_model = True, 
                            fix_pre_trained_model = True,
                            fold_factor = 3,
                            making_video_type = [2],
                            share_tuples = True):
    frame_per_window = frame_size
    frame_per_sliding = frame_size
    
    # Create base model directory
    base_model_path = f"./model/{model_folder_name}_{frame_per_window}frames"
    os.makedirs(base_model_path, exist_ok=True)
    
    
    
    h = 360
    w = 720
    c = 1
    fps = 30
    downsampling_factor = 5.625
    
    input_ch = 1
    
    # hyperparameter 
    batch_size = 10
    lr = 1e-4
    epochs = 50
    
    
    folder_path = "./naturalistic"
    mat_file_name = f"experimental_data.mat"
    checkpoint_name = "fly_model"
    pretrained_model_path = "./pretrained_model/64x128_max_poolled.ckpt"
    
    video_name = {
        0 : 'bird',
        1 : 'city',
        2 : 'forest'
    }
    

    #layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]
    layer_configs = [[64, 2], [128, 2], [256, 2]]#, [512, 2]]
    video_data, wba_data, total_frame = direction_pred_training_data_preparing_seq(folder_path, mat_file_name, downsampling_factor)
    video_data, wba_data, aug_factor = aug_videos(video_data, wba_data)
    
    print(f"augmented shape : {video_data.shape}")
    print(f"augmented shape : {wba_data.shape}")
    
    model_name = base_model_path
    for pre_trained in [True, False]:
        use_pretrained_model = pre_trained
        for piece_index, piece_size in enumerate(piece_sizes):
            # Create subdirectory for each configuration
            config_string = f"piece_size_{piece_size}_fix_{str(fix_pre_trained_model)}_pretrained_{str(use_pretrained_model)}"
            config = {
                "fix" : fix_pre_trained_model,
                "pretrained" : use_pretrained_model
            }
            model_path = os.path.join(base_model_path, config_string)
            os.makedirs(model_path, exist_ok=True)
            
            # Update model_name to use new path

            model_string = model_folder_name
            model_string += f"_piece_size_{piece_size}_fix_{str(fix_pre_trained_model)}"
            model_string += f"_{frame_per_window}frames"

            result_save_path = os.path.join(model_path, "result_data.h5")

            # Split period and split for training / test data set
            recent_losses = deque(maxlen=100)
            recent_f1_scores = deque(maxlen=100)
            val_losses_per_epoch = []

            batch_tuples = np.array(generate_tuples_direction_pred(
                total_frame, 
                frame_per_window,
                frame_per_sliding, 
                video_indices,
                video_data.shape[0] 
            ))
            #kf = KFold(n_splits=fold_factor, random_state=42, shuffle=True)
            fold_set_list = []
                
            if validation_ratio > 0:
                fold_set_list = split_train_val_index(batch_tuples, 
                                                    aug_factor, 
                                                    fold_factor=fold_factor, 
                                                    piece_size=piece_size, 
                                                    val_ratio=validation_ratio, 
                                                    video_index_num = len(video_indices),
                                                    )
            else:
                # validation_ratio가 0일 때는 모든 데이터를 training set으로
                fold_set_list = [(np.arange(len(batch_tuples)), np.array([])) for _ in range(fold_factor)]
            

            all_fold_losses = []


            KST = pytz.timezone('Asia/Seoul')
            for fold, (train_index, val_index) in enumerate(fold_set_list):
                print(f"Fold {fold+1}")
                
                fold_path = f"{model_path}/fold_{fold+1}"

                # create model
                flownet_model = flownet3d(layer_configs, num_classes=2)
                if use_pretrained_model:
                    flownet_model = load_model(flownet_model, pretrained_model_path)
                model = model_class(flownet_model, 
                                    feature_dim=128, 
                                    input_size=(frame_per_window, int(h//downsampling_factor), int(w//downsampling_factor), c),
                                    freeze=fix_pre_trained_model)
                trainer = Trainer(model, loss_function_mse, lr)
                
                
                current_epoch = trainer.load(f"{fold_path}/{checkpoint_name}.ckpt")
                os.makedirs(fold_path, exist_ok=True)
                
                # split training and validation tuples
                if share_tuples:
                    training_tuples, val_tuples = search_related_tuples(base_model_path, config_string, fold+1)
                    if training_tuples is None:
                        training_tuples = batch_tuples[train_index]
                        val_tuples = batch_tuples[val_index]
                else:
                    training_tuples = batch_tuples[train_index]
                    val_tuples = batch_tuples[val_index]
                    
                with open(f"{fold_path}/training_tuples.pkl", "wb") as f:
                    pickle.dump(training_tuples, f)
                with open(f"{fold_path}/validation_tuples.pkl", "wb") as f:
                    pickle.dump(val_tuples, f)
                
                val_tuples = [tup for tup in val_tuples if tup[0] in [video_n * aug_factor for video_n in video_indices]]
                

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

                    # training phase
                    batches = list(get_batches(training_tuples, batch_size))
                    print(f"Epoch {epoch + 1}/{epochs}")
                    progress_bar = tqdm(batches, desc=f'Epoch {epoch + 1}', leave=False, ncols=120, disable=True)
                    total_train_loss = 0.0

                    for batch in progress_bar:
                        batch_input_data, batch_target_data, batch_wba_data = get_data_from_batch_direction_pred(
                            video_data, 
                            wba_data, 
                            batch, 
                            frame_per_window
                        )
                        batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
                        batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
                        batch_wba_data = torch.tensor(batch_wba_data, dtype=torch.float32).to(trainer.device)

                        loss, pred = trainer.step(batch_input_data, batch_target_data, batch_wba_data)

                        recent_losses.append(loss.item())
                        avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else 0

                        #progress_bar.set_postfix(
                        #    loss=f"{loss.item():.5f}",
                        #    avg_recent_loss=f"{avg_recent_loss:.5f}",
                        #    lr=f"{trainer.lr:.7f}"
                        #)

                        total_train_loss += loss.item()

                        del batch_input_data, batch_target_data, batch_wba_data, loss, pred

                    avg_train_loss = total_train_loss / len(batches)
                    train_losses.append(avg_train_loss)

                    # Validation Phase
                    if validation_ratio > 0:
                        val_batches = list(get_batches(val_tuples, batch_size))
                        total_val_loss = 0.0
                        val_predictions = []
                        
                        progress_bar = tqdm(val_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=120, disable=True)

                        for batch in progress_bar:
                            batch_input_data, batch_target_data, batch_wba_data = get_data_from_batch_direction_pred(
                                video_data,
                                wba_data, 
                                batch, 
                                frame_per_window
                            )
                            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
                            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
                            batch_wba_data = torch.tensor(batch_wba_data, dtype=torch.float32).to(trainer.device)

                            loss, pred = trainer.evaluate(batch_input_data, batch_target_data, batch_wba_data)

                            batch_target_data_cpu = batch_target_data.cpu()
                            predictions_cpu = pred.cpu()

                            for i, (video_num, start_frame) in enumerate(batch):
                                val_predictions.append((video_num, start_frame, predictions_cpu[i].item()))

                            total_val_loss += loss.item()
                        
                        avg_val_loss = total_val_loss / len(val_batches) if val_batches else float('inf')
                        val_losses.append(avg_val_loss)
                        print(f"Training loss: {avg_train_loss:.5f} || Validation loss: {avg_val_loss:.5f}")
                    else:
                        avg_val_loss = avg_train_loss
                        val_losses.append(avg_val_loss)
                        val_predictions = []  # 빈 리스트로 초기화
                        print(f"Training loss: {avg_train_loss:.5f} (No validation)")

                    update_metrics_plot(fold_path, epoch, train_losses, val_losses)

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                    
                    # Save model if this epoch has the lowest test loss
                    if avg_val_loss < min_val_loss:
                        min_val_loss = avg_val_loss
                        best_epoch = epoch + 1
                        best_model_path = f"{fold_path}/best_model.ckpt"
                        trainer.save(best_model_path, epoch)
                        if validation_ratio > 0:
                            print(f"New best model saved at epoch {best_epoch} with validation loss {avg_val_loss:.5f}")
                        else:
                            print(f"New best model saved at epoch {best_epoch} with training loss {avg_val_loss:.5f}")

                    if epoch == start_epoch and fold == 0 and piece_index == 0:
                        first_epoch_duration = time.time() - start_time
                        print(f"First epoch took {first_epoch_duration:.2f} seconds.")

                        # 전체 프로그램의 예상 종료 시간 계산
                        total_duration = first_epoch_duration * epochs * fold_factor * len(piece_sizes)
                        estimated_end_time = datetime.now(KST) + timedelta(seconds=total_duration)
                        print(f"Estimated total program duration: {total_duration / 3600:.2f} hours")
                        print(f"Estimated program end time (KST): {estimated_end_time.strftime('%Y/%m/%d %H:%M:%S')}")

                    # 5 epoch마다 그래프 플로팅 및 저장
                    if (epoch + 1) % 5 == 0:
                        
                        for video_index in video_indices:
                            selected_val_predictions = [tup for tup in val_predictions if tup[0] == video_index*aug_factor]
                            plt.figure(figsize=(10, 6))
                            
                            # wba_data를 window size만큼 생략하고 플로팅
                            #diff_wba_for_plotting = [ wba_data[ 2*aug_factor, frame_per_window * (i+1) ] - wba_data[2*aug_factor, frame_per_window * i ] for i in range(len(wba_data[2*aug_factor]) // frame_per_window - 1) ]
                            #plt.plot(np.array(range(len(diff_wba_for_plotting)))+1, diff_wba_for_plotting, label='WBA Data', color='blue')
                            plt.plot(np.array(range(0,len(wba_data[video_index*aug_factor])))+1, wba_data[video_index*aug_factor], label='WBA Data', color='blue')
                            
                            
                            video_num, val_frames, val_preds = zip(*selected_val_predictions)
                            for i in range(len(val_frames)):
                                start_frame = val_frames[i] - frame_per_window
                                plt.plot([start_frame+3, val_frames[i]+3], [wba_data[video_index*aug_factor][start_frame+3], val_preds[i]], color='red')
                                plt.scatter(start_frame+3, wba_data[video_index*aug_factor][start_frame+3], color='black', s=7)  # 시작점에 작은 초록색 원 추가
                                plt.scatter(val_frames[i]+3, val_preds[i], color='black', s=7)  # 끝점에 작은 주황색 원 추가
                            #plt.scatter(np.array(val_frames), val_preds, color='red', s=6, label='Validation Predictions')
                            
                            plt.xlabel('Frame')
                            plt.ylabel('WBA Value')
                            plt.title(f'Validation Predictions vs WBA Data at Epoch {epoch + 1}')
                            plt.legend()
                            
                            intermediate_path = f"{fold_path}/intermediate_epoch"
                            os.makedirs(intermediate_path, exist_ok=True)
                            plt.savefig(f"{intermediate_path}/{epoch + 1}_{video_name[video_index]}.png")
                            plt.close()

                print(f"Best model for fold {fold + 1} saved from epoch {best_epoch} with f1 {min_val_loss:.5f}")
                all_fold_losses.append(min_val_loss)

                #print(f"Final training loss: {train_losses[-1]:.5f} || Final test loss: {val_losses[-1]:.5f}")

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
        
        model_name_for_visualization = f"{model_folder_name}_{frame_per_window}frames"
        for video_type in making_video_type:
            create_visualization_video(model_name_for_visualization, video_type=video_type, config=config)
    
if __name__ == "__main__":
    model_string = "forest_wba_value_compare_pretrained_and_non_decoder_output"
    video_name = {
        0 : 'bird',
        1 : 'city',
        2 : 'forest'
    }
    
    piece_sizes = [1, 5, 10, 20, 40]
    frame_sizes = [8]
    video_indices = [2]
    making_video_type = [2]
    use_pretrained_model = False
    fix_pre_trained_model = False
    share_tuples = True
    fold_factor = 1
    validation_ratio = 0.3
    
    
    model_class = FlowNet3DWithFeatureExtraction_decoder_output
    
    #training_direction_pred(model_string, piece_sizes, fix_pre_trained_model= True)
    
    
    for frame_size in frame_sizes:
        training_direction_pred(
            model_string,
            model_class = model_class,
            video_indices=video_indices, 
            piece_sizes=piece_sizes, 
            frame_size=frame_size, 
            validation_ratio = 0.3,
            fold_factor = fold_factor,
            use_pretrained_model= use_pretrained_model,
            fix_pre_trained_model= fix_pre_trained_model,
            making_video_type=making_video_type,
            share_tuples = share_tuples
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    