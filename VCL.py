import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


from convlstm import *
class VCL(nn.Module):
    def __init__(self, input_dims, video_size, result_patience):
        super(VCL, self).__init__()
        
        self.latent_dims = 1024
        channel_num_list = [16, 32, 16]
        num_layers = len(channel_num_list)
        self.result_delay = result_patience
        self.conv_lstm = ConvLSTM(input_dims,
                                  hidden_dim=channel_num_list, 
                                  kernel_size=(3, 3), 
                                  num_layers=num_layers, 
                                  pooling_size = 2, 
                                  batch_first=True, 
                                  return_all_layers=False)
        h, w = video_size
        flatten_size =  (h//(2**num_layers)) * (w//(2**num_layers)) * channel_num_list[-1]
        self.mu = nn.Linear(flatten_size , self.latent_dims)
        self.log_var = nn.Linear(flatten_size , self.latent_dims)
        
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dims,512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        # self.conv_lstm = self.conv_lstm.half()
        # self.mu = self.mu.half()
        # self.log_var = self.log_var.half()
        # self.fc = self.fc.half()
        
    def swap_axis_for_input(self, t):
        return t.permute(0, 1, 4, 2, 3)
    
    
    def reparameterization(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        epsilon = torch.randn_like(std)
        z = mu + std * epsilon
        return z
    
    def forward(self, x):
        #x = x.half()
        x = self.swap_axis_for_input(x) # B, T, C, H, W
        
        
        cl_output, _ = self.conv_lstm(x)
        #(time_step, channel(16), h//3, w//3)
        
        cl_output = cl_output[-1]
        #cl_output = cl_output.view(cl_output.size(0), cl_output.size(1), -1) # flatten
        cl_output = torch.flatten(cl_output, start_dim=2)
        
        # (time_step, dims )
        
        mu = F.relu(self.mu(cl_output))
        log_var = F.relu(self.log_var(cl_output))   # log
        # (time_step, 64 )
        
        z = self.reparameterization(mu, torch.exp(0.5 * log_var)) # log std -> std
        
        # (time_step, 1), time series graph
        output = self.fc(z).squeeze()
        
        mu = mu[:,self.result_delay:]
        log_var = log_var[:,self.result_delay:]
        output = output[:,self.result_delay:]
    
        
        return output, mu, log_var
        
def loss_function(pred, target, mu, log_var):
    mse_loss =  F.mse_loss(pred, target, reduction='sum')
    kl_div = 0.1 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    
    return mse_loss + kl_div

class VCL_Trainer :
    def __init__(self, model, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.step_counter = 0
        self.loss_sum = 0
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=1, threshold=0.5, min_lr=1e-6)
    
    def predict(self, input):
        input = input.to(self.device)
        pred, *_ = self.model(input)
        return pred
    
    def save(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
        else:
            print(f"Model not found at {path}")
        
    def step(self, input, target):
        self.step_counter += 1
        input = input.to(self.device)
        target = target.to(self.device)
        
        self.optimizer.zero_grad()
        pred, mu, log_var = self.model(input)
        
        loss = loss_function(pred, target, mu, log_var)
        self.loss_sum += loss
        
        if self.step_counter % 100 == 0:
            avg_loss = self.loss_sum / 100
            prev_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(avg_loss)
            post_lr = self.optimizer.param_groups[0]['lr']
            if not prev_lr == post_lr:
                print()
                print(f"lr : {post_lr}")
            self.loss_sum = 0

        
        loss.backward()
        self.optimizer.step()
        return loss, pred
    
    def evaluate(self, input, target):
        self.model.eval()
        with torch.no_grad():
            input = input.to(self.device)
            target = target.to(self.device)
            pred, mu, log_var = self.model(input)
            loss = loss_function(pred, target, mu, log_var)
        self.model.train()
        return loss, pred
    
    
    
    
        
    





# class Simple3DCNN(nn.Module):
#     def __init__(self, direction_filters):
#         super(Simple3DCNN, self).__init__()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.max_pool = nn.MaxPool3d(kernel_size=(2, 1, 1))

#     def forward(self, x):
#         return self.max_pool(x)