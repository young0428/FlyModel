import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Trainer :
    def __init__(self, model, loss_func, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.step_counter = 0
        self.loss_sum = 0
        self.loss_func = loss_func
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, threshold=0.1, min_lr=1e-6)
    
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
        if self.step_counter % 100 == 0:
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

    
def save_test_result(batch_input_data, predictions, epoch, fold_path ):
    fig, axes = plt.subplots(1, 3, figsize=(15, 9))
    
    def update(frame_idx):
        for i in range(3):
            axes[i].clear()
            # Extract the frame for the current batch
            frame = batch_input_data[i, frame_idx, :, :, 0].cpu().numpy()
            
            # Plot the frame
            axes[i].imshow(frame, cmap='gray')
            axes[i].axis('off')
            
            # Add the prediction label
            prediction_label = '왼쪽' if predictions[i].item() > 0 else '오른쪽'
            axes[i].set_title(f"Batch {i}: {prediction_label}")
        
    ani = animation.FuncAnimation(fig, update, frames=16, interval=200)    
    intermediate_path = f"{fold_path}/intermediate_epoch"
    ani.save(f'{intermediate_path}/{epoch+1}.gif', writer='imagemagick')
    plt.close()
