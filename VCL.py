import torch
import torch.nn as nn
import torch.nn.functional as F


from convlstm import *
class VCL(nn.Module):
    def __init__(self, input_dims, video_size):
        super(VCL, self).__init__()
        
        self.latent_dims = 64
        num_layers = 3
        self.conv_lstm = ConvLSTM(input_dims,
                                  hidden_dim=[16, 32, 16], 
                                  kernel_size=(3, 3), 
                                  num_layers=num_layers, 
                                  pooling_size = 2, 
                                  batch_first=True, 
                                  return_all_layers=False)
        h, w = video_size
        flatten_size =  (h//(2**num_layers)) * (w//(2**num_layers)) * 16
        self.mu = nn.Linear(flatten_size , self.latent_dims)
        self.log_var = nn.Linear(flatten_size , self.latent_dims)
        
        self.fc = nn.Sequential(
            nn.Linear(self.latent_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
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
        
        # (time_step, dims(ch * h//3 * w//3) )
        mu = F.relu(self.mu(cl_output))
        log_var = F.relu(self.log_var(cl_output))   # log
        # (time_step, 64 )
        
        z = self.reparameterization(mu, torch.exp(0.5 * log_var)) # log std -> std
        
        # (time_step, 1), time series graph
        output = self.fc(z).squeeze()
        
        
        return output, mu, log_var
        
def loss_function(pred, target, mu, log_var):
    mse_loss =  F.mse_loss(pred, target, reduction='sum')
    kl_div = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    
    return mse_loss + kl_div

class VCL_Trainer :
    def __init__(self, model, lr):
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.model = self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr=lr)
        
    def step(self, input, target):
        input = input.to(self.device)
        target = target.to(self.device)
        
        # for idx, param in enumerate(self.model.parameters()):
        #     if idx < 2 : print(param)
        self.optimizer.zero_grad()
        pred, mu, log_var = self.model(input)
        loss = loss_function(pred, target, mu, log_var)

        
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