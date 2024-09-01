import torch
import torch.nn as nn
import torch.nn.functional as F

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

def loss_function(pred, target):
    loss = F.mse_loss(pred, target)
    return loss

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
    def __init__(self, flownet3d, feature_dim=128, input_size=(16, 64, 128)):
        super(FlowNet3DWithFeatureExtraction, self).__init__()
        self.flownet3d = flownet3d
        depth, height, width = input_size
        
        # Encoder와 Decoder의 파라미터를 고정 (freeze)
        for param in self.flownet3d.encoder.parameters():
            param.requires_grad = False
        for param in self.flownet3d.decoder.parameters():
            param.requires_grad = False
        
        # Decoder 각 단계에서 나오는 feature를 변환하는 FC layer 정의
        self.fc_layers = nn.ModuleList()
        for i in range(len(self.flownet3d.decoder.upconvs)):
            in_channels = flownet3d.decoder.convs[i].out_channels
            
            # 각 레이어에서 Conv와 UpConv가 적용된 후의 출력 크기를 계산
            depth = self._calculate_conv_output_size(depth, kernel_size=3, padding=1, stride=1)
            height = self._calculate_conv_output_size(height, kernel_size=3, padding=1, stride=1)
            width = self._calculate_conv_output_size(width, kernel_size=3, padding=1, stride=1)
            
            # 그 결과에 따라 Linear 레이어의 입력 크기를 동적으로 결정
            linear_input_size = feature_dim * depth * height * width
            
            self.fc_layers.append(nn.Sequential(
                nn.Conv3d(in_channels, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(linear_input_size, feature_dim),  # Adjusted dynamically based on input size
                nn.ReLU(inplace=True)
            ))
        
        # 최종 스칼라 값을 출력하는 FC layer
        self.final_fc = nn.Linear(feature_dim * len(self.fc_layers), 1)

    def _calculate_conv_output_size(self, size, kernel_size, padding, stride):
        """Convolution 레이어가 적용된 후의 크기를 계산하는 함수"""
        return (size - kernel_size + 2 * padding) // stride + 1
    
    def forward(self, x):
        x = self.flownet3d.swap_axis_for_input(x)
        encoder_outputs = self.flownet3d.encoder(x)
        
        features = []
        decoder_output = encoder_outputs[-1]
        for i in range(len(self.flownet3d.decoder.upconvs)):
            decoder_output = self.flownet3d.decoder.upconvs[i](decoder_output)
            decoder_output = torch.cat([decoder_output, encoder_outputs[-(i+2)]], dim=1)
            decoder_output = F.relu(self.flownet3d.decoder.convs[i](decoder_output))
            
            # 각 단계에서의 feature 추출
            feature = self.fc_layers[i](decoder_output)
            features.append(feature)
        
        # 모든 feature를 종합하여 최종 스칼라 값 출력
        combined_features = torch.cat(features, dim=1)
        scalar_output = self.final_fc(combined_features)
        
        return scalar_output

 



#예시: 다양한 형태의 layer_configs를 입력으로 모델을 구성
# layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]
# model = flownet3d(layer_configs, num_classes=2)

# # 테스트용 입력 데이터 생성
# x = torch.randn((2, 32, 128, 128, 1))  # Batch size: 1, Channels: 4, Depth: 32, Height: 128, Width: 128
# output = model(x)

# loss = loss_function(x, output)
# print(loss.item())

# print(output.shape)  # 출력 크기 확인