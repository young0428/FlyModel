#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from FlowNet import *
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

frame_per_window = 16
frame_per_sliding = 5
input_ch = 1 

model_string = "8fold_opticflow"
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

layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]

video_data, total_frame = load_video_data(folder_path, downsampling_factor)

# Split period and split for training / test data set
recent_losses = deque(maxlen=100)
test_losses_per_epoch = []

batch_tuples = np.array(generate_tuples_flow(total_frame, frame_per_window, frame_per_sliding, 3))
kf = KFold(n_splits=fold_factor)

all_fold_losses = []
#%%


#%%
for fold, (train_index, val_index) in enumerate(kf.split(batch_tuples)):
    if not fold < 4:
        continue

    print(f"Fold {fold+1}")
    fold_path = f"{model_name}/fold_{fold+1}"

    # create model
    #model = p3d_resnet(input_ch, block_list, feature_output_dims)
    model = flownet3d(layer_configs, num_classes=2)
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
            batch_input_data, batch_target_data = get_data_from_batch_flow_estimate(
                video_data, batch, frame_per_window
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
        total_test_correct = 0
        total_test_samples = 0
        
        progress_bar = tqdm(val_batches, desc=f'Testing after Epoch {epoch + 1}', leave=False, ncols=150)

        for batch in progress_bar:
            batch_input_data, batch_target_data = get_data_from_batch_flow_estimate(
                video_data, batch, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            
            
            # Calculate test accuracy
            loss, pred = trainer.evaluate(batch_input_data, batch_target_data)
            
            total_test_loss = total_test_loss + loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.5f}", lr=f"{trainer.lr:.7f}")
            
            #del batch_input_data, batch_target_data, loss, pred

        avg_test_loss = total_test_loss / len(val_batches)
        test_losses_per_epoch.append(avg_test_loss)
        print(f"Average test loss after Epoch {epoch + 1}: {avg_test_loss:.5f}")

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
            selected_indices = np.random.choice(val_tuples.shape[0], size=3, replace=False)
            selected_val_tuples = val_tuples[selected_indices]
            
            batch_input_data, batch_target_data = get_data_from_batch_flow_estimate(
                video_data, selected_val_tuples, frame_per_window
                )
            
            batch_input_data = torch.tensor(batch_input_data, dtype=torch.float32).to(trainer.device)
            batch_target_data = torch.tensor(batch_target_data, dtype=torch.float32).to(trainer.device)
            
            _, predictions = trainer.evaluate(batch_input_data, batch_target_data)
            batch_input_data[:, ::2, ::2, ::2, 0:1]
            

            fig, axes = plt.subplots(3, 6, figsize=(15, 8))

            # 서브플롯에 이미지를 삽입
            
            # input            
            for j in range(3):
                img = batch_input_data[j,-1].cpu()
                axes[0, j*2].imshow(img)
                axes[0, j*2].axis('off')  # 축 숨기기
                axes[0, j*2+1].axis('off')  
                axes[0, j*2].set_title(f'{str(val_tuples[j][0])}')  # 제목 설정
            
            # target
            for j in range(3):
                img_left = batch_target_data[j,-1,:,:,0].cpu()
                img_right = batch_target_data[j,-1,:,:,1].cpu()
                axes[1, j*2].imshow(img_left)
                axes[1, j*2+1].imshow(img_right)
                axes[1, j*2].axis('off')  # 축 숨기기
                axes[1, j*2+1].axis('off')
                
            # prediction       
            for j in range(3):
                img_left = predictions[j,-1,:,:,0].cpu()
                img_right = predictions[j,-1,:,:,1].cpu()
                axes[2, j*2].imshow(img_left)
                axes[2, j*2+1].imshow(img_right)
                axes[2, j*2].axis('off')  # 축 숨기기
                axes[2, j*2+1].axis('off')
                
            intermediate_path = f"{fold_path}/intermediate_epoch"
            os.makedirs(intermediate_path, exist_ok=True)
            plt.tight_layout()
            plt.savefig(f"{intermediate_path}/results_epoch_{epoch + 1}.png")
            plt.close()
            
            

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