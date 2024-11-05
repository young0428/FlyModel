import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.relu(out+identity)
        
        return out

class Encoder3D(nn.Module):
    def __init__(self, block, layer_configs):
        super(Encoder3D, self).__init__()
        self.in_channels = layer_configs[0][0]  # 첫 번째 레이어의 채널 수로 초기화
        
        self.conv1 = nn.Conv3d(1, self.in_channels, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layers = self._make_layers(block, layer_configs)
    
    def _make_layers(self, block, layer_configs):
        layers = []
        for out_channels, blocks in layer_configs:
            stride = 2 if self.in_channels != out_channels else 1
            layers.append(self._make_layer(block, out_channels, blocks, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        outputs = []
        for layer in self.layers:
            x = layer(x)
            outputs.append(x)
            
        
        return outputs

class Decoder3D(nn.Module):
    def __init__(self, layer_configs, num_classes=2):
        super(Decoder3D, self).__init__()
        self.upconvs = nn.ModuleList()
        self.convs = nn.ModuleList()
        
        for i in range(len(layer_configs)-1, 0, -1):
            in_channels = layer_configs[i][0]
            out_channels = layer_configs[i-1][0]
            self.upconvs.append(nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2))
            self.convs.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1))
        
        self.final_conv = nn.Conv3d(layer_configs[0][0], num_classes, kernel_size=1)
    
    def forward(self, encoder_outputs):
        x = encoder_outputs[-1]
        
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)

            x = torch.cat([x, encoder_outputs[-(i+2)]], dim=1)
            x = F.relu(self.convs[i](x))
        
        x = self.final_conv(x)
        return x

class FlowNet3D(nn.Module):
    def __init__(self, block, layer_configs, num_classes=2):
        super(FlowNet3D, self).__init__()
        self.encoder = Encoder3D(block, layer_configs)
        self.decoder = Decoder3D(layer_configs, num_classes)
        
    def swap_axis_for_input(self, t):
        return t.permute(0, 4, 1, 2, 3)
    
    def reswap_axis_for_input(self, t):
        return t.permute(0, 2, 3, 4, 1)
    
    def forward(self, x):

        x = self.swap_axis_for_input(x)
        encoder_outputs = self.encoder(x)
        out = self.decoder(encoder_outputs)
        out = self.reswap_axis_for_input(out)
        return out

def flownet3d(layer_configs, num_classes = 2):
    return FlowNet3D(ResidualBlock, layer_configs, num_classes)



class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class FlowNet3DWithFeatureExtraction(nn.Module):
    def __init__(self, flownet3d, feature_dim=128, input_size=(16,64,128,1), freeze=True):
        super(FlowNet3DWithFeatureExtraction, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flownet3d = flownet3d.to(self.device)
        self.feature_dim = feature_dim
        self.num_lstm_layers = 2
        self.bidirectional = False
        self.num_directions = 1
        
        # Encoder와 Decoder 파라미터 고정
        for param in self.flownet3d.encoder.parameters():
            param.requires_grad = not freeze
        for param in self.flownet3d.decoder.parameters():
            param.requires_grad = not freeze
        
        # WBA 값으로부터 LSTM 초기 상태를 생성하는 레이어
        self.wba_to_hidden = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_lstm_layers * feature_dim)
        ).to(self.device)
        
        self.wba_to_cell = nn.Sequential(
            nn.Linear(1, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_lstm_layers * feature_dim)
        ).to(self.device)
        
        # Dummy forward pass로 LSTM input size 계산
        D, H, W, C = input_size
        with torch.no_grad():
            dummy_input = torch.zeros(1, D, H, W, C).to(self.device)
            dummy_output = self.flownet3d(dummy_input)
            dummy_output = self.flownet3d.swap_axis_for_input(dummy_output)
            self.lstm_input_size = dummy_output.shape[1] * dummy_output.shape[3] * dummy_output.shape[4]
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=feature_dim,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=False
        ).to(self.device)
        
        # FC 레이어
        self.fc_layers = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        ).to(self.device)

    def init_lstm_states(self, wba_input, batch_size):
        # WBA 값으로부터 초기 hidden state와 cell state 생성
        h0 = self.wba_to_hidden(wba_input)
        c0 = self.wba_to_cell(wba_input)
        
        h0 = h0.view(batch_size, 
                     self.num_lstm_layers, 
                     self.feature_dim).transpose(0, 1).contiguous()
        
        c0 = c0.view(batch_size, 
                     self.num_lstm_layers, 
                     self.feature_dim).transpose(0, 1).contiguous()
        
        return h0, c0

    def forward(self, x, wba_input):
        # FlowNet 처리
        x = self.flownet3d.swap_axis_for_input(x)
        encoder_outputs = self.flownet3d.encoder(x)
        decoder_output = self.flownet3d.decoder(encoder_outputs)
        
        # LSTM을 위한 데이터 재구성
        batch_size = decoder_output.size(0)
        time_steps = decoder_output.size(2)
        
        lstm_input = decoder_output.permute(0, 2, 1, 3, 4)
        lstm_input = lstm_input.reshape(batch_size, time_steps, self.lstm_input_size)
        
        # WBA 값으로부터 LSTM 초기 상태 생성
        h0, c0 = self.init_lstm_states(wba_input, batch_size)
        
        # LSTM 처리
        lstm_output, _ = self.lstm(lstm_input, (h0, c0))
        lstm_features = lstm_output[:, -1, :]  # 마지막 시점의 출력만 사용
        
        # 최종 예측
        output = self.fc_layers(lstm_features)
        
        return output


def loss_function_mse(pred, target):
    loss = F.mse_loss(pred, target)
    return loss

def loss_function_bce(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)
 