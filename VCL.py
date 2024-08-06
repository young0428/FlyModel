import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


from convlstm import *
class VCL(nn.Module):
    def __init__(self, input_dims, video_size, result_patience, channel_num_list=[64,64,64]):
        super(VCL, self).__init__()
        
        self.latent_dims = 1024
        
        num_layers = len(channel_num_list)
        self.result_patience = result_patience
        self.conv_lstm = ConvLSTM(input_dims,
                                  hidden_dim=channel_num_list, 
                                  kernel_size=(3, 3), 
                                  num_layers=num_layers, 
                                  #pooling_size = 2, 
                                  batch_first=True, 
                                  return_all_layers=False)
        h, w = video_size
        #flatten_size =  (h//(2**num_layers)) * (w//(2**num_layers)) * channel_num_list[-1]
        flatten_size =  h * w * channel_num_list[-1]
        
        self.fc = nn.Sequential(
            nn.Linear(flatten_size,2048),
            nn.GELU(),
            nn.Linear(2048,1024),
            nn.GELU(),
            nn.Linear(1024,1),

        )
        print("channel list : " + str(channel_num_list))

        
    def swap_axis_for_input(self, t):
        return t.permute(0, 1, 4, 2, 3)
    

    
    def forward(self, x):
        x = self.swap_axis_for_input(x) # B, T, C, H, W
        
        cl_output, _ = self.conv_lstm(x)
        cl_output = cl_output[-1]
        cl_output = torch.flatten(cl_output, start_dim=2)
        
        output = self.fc(cl_output).squeeze(-1)
        
        output = output[:,self.result_patience:]

    
        
        return output
        
def loss_function(pred, target):
    mse_loss =  F.mse_loss(pred, target, reduction='sum')
    
    return mse_loss

class VCL_Trainer :
    def __init__(self, model, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.step_counter = 0
        self.loss_sum = 0
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=100, threshold=0.1, min_lr=1e-6)
    
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
        pred= self.model(input)
        
        loss = loss_function(pred, target)
        self.loss_sum += loss
        
        if self.step_counter % 100 == 0:
            avg_loss = self.loss_sum / 100
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(avg_loss)
            post_lr = self.optimizer.param_groups[0]['lr']
            self.loss_sum = 0

        
        loss.backward()
        self.optimizer.step()
        return loss, pred
    
    def evaluate(self, input, target):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            target = target.to(self.device)
            pred = self.model(input)
            loss = loss_function(pred, target)
        self.model.train()
        return loss, pred
    
    
    
    