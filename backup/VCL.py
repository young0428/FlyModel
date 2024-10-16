import torch
import torch.nn as nn
import torch.nn.functional as F

from convlstm import *
class VCL(nn.Module):
    def __init__(self, input_dims, video_size):
        super(VCL, self).__init__()
        
        self.latent_dims = 64
        self.conv_lstm = ConvLSTM(input_dims, hidden_dim=[16, 32, 64, 32, 16], kernel_size=(3, 3), num_layers=5, pooling_size = 2, batch_first=True, return_all_layers=False)
        h, w = video_size
        flatten_size =  (h//(2**4)) * (w//(2**4)) * 16
        self.mu = nn.Linear(flatten_size , self.latent_dims)
        self.sig = nn.Linear(flatten_size , self.latent_dims)
        
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def swap_axis_for_input(self, t):
        return t.permute(0, 1, 4, 2, 3)
    
    def swap_axis_for_output(self, t):
        return t.permute(0, 1, 3, 4, 2)
    
    def reparameterization(self, mu, sig):
        epsilon = torch.randn_like(sig)
        z = mu + sig * epsilon
        return z
    def forward(self, x):
        x = self.swap_axis_for_input(x) # B, T, C, H, W
        cl_output, _ = self.conv_lstm(x)
        cl_output = cl_output[0]
        cl_output = cl_output.view(cl_output.size(0), cl_output.size(1), -1) # flatten
        
        
        mu = self.mu(cl_output)
        log_std = self.sig(cl_output)   # log
        
        z = self.reparameterization(mu, torch.exp(0.5 * log_std)) # log std -> std
        
        output = self.fc(z).squeeze()
        
        
        return output, mu, log_std
        
def loss_function(pred, target, mu, log_std):
    mse_loss =  F.mse_loss(pred, target, reduction='sum')
    kl_div = 0.5 * torch.sum(mu.pow(2) + log_std.exp() - log_std - 1)
    
    return mse_loss + kl_div

class VCL_Trainer :
    def __init__(self, model, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def step(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        
        pred, mu, log_std = self.model(input)
        loss = loss_function(pred, target, mu, log_std)
        

        self.optimizer.zero_grad()
        
        loss.backward()
        self.optimizer.step()
        
        return loss
    
    def predict(self, input):
        input = input.to(self.device)
        pred, *_ = self.model(input)
        return pred
        
    





# class Simple3DCNN(nn.Module):
#     def __init__(self, direction_filters):
#         super(Simple3DCNN, self).__init__()
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.max_pool = nn.MaxPool3d(kernel_size=(2, 1, 1))

#     def forward(self, x):
#         return self.max_pool(x)