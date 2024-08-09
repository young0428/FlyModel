import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

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