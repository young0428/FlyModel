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
    def __init__(self, flownet3d, feature_dim=128, input_size = (16,64,128,1)):
        super(FlowNet3DWithFeatureExtraction, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.flownet3d = flownet3d.to(self.device)
        self.feature_dim = feature_dim
        
        
        # Encoder와 Decoder의 파라미터를 고정 (freeze)
        for param in self.flownet3d.encoder.parameters():
            param.requires_grad = False
        for param in self.flownet3d.decoder.parameters():
            param.requires_grad = False
        
        # Decoder 각 단계에서 나오는 feature를 변환하는 layer들을 초기화
        self.spatial_attentions = nn.ModuleList()
        self.feature_conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()

        # 레이어 초기화는 forward pass에서 수행

        # 최종 스칼라 값을 출력하는 FC layer
        # 실제 크기는 forward pass에서 결정됨
        self.final_fc = None
        D, H, W, C = input_size
        
        x = torch.randn((1,16,64,128,1)).to(self.device)
        
        x = self.flownet3d.swap_axis_for_input(x)
        encoder_outputs = self.flownet3d.encoder(x)
        
        decoder_outputs = []
        decoder_output = encoder_outputs[-1]
        for i in range(len(self.flownet3d.decoder.upconvs)):
            decoder_output = self.flownet3d.decoder.upconvs[i](decoder_output)
            decoder_output = torch.cat([decoder_output, encoder_outputs[-(i+2)]], dim=1)
            decoder_output = F.relu(self.flownet3d.decoder.convs[i](decoder_output))
            decoder_outputs.append(decoder_output)

        # 첫 번째 forward pass에서 레이어 초기화
        if not self.feature_conv_layers:
            self._initialize_layers(decoder_outputs)
        

    def _initialize_layers(self, decoder_outputs):
        for i, decoder_output in enumerate(decoder_outputs):
            _, C, D, H, W = decoder_output.shape
            
            self.spatial_attentions.append(SpatialAttention())
            
            self.feature_conv_layers.append(nn.Sequential(
                nn.Conv3d(C, self.feature_dim, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv3d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1, stride=2),
                nn.ReLU(inplace=True),
            ))
            
            # Conv3D 레이어 통과 후 크기 계산 (각 레이어마다 크기가 절반으로 줄어듦)
            D_out = max(D // 8, 1)
            H_out = max(H // 8, 1)
            W_out = max(W // 8, 1)
            
            linear_input_size = D_out * H_out * W_out * self.feature_dim
            
            self.fc_layers.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(linear_input_size, self.feature_dim),
                nn.ReLU(inplace=True)
            ))
        
        # 최종 FC 레이어 초기화
        self.final_fc = nn.Linear(self.feature_dim * len(self.feature_conv_layers), 1)
        
        self.spatial_attentions = self.spatial_attentions.to(self.device)
        self.feature_conv_layers = self.feature_conv_layers.to(self.device)
        self.fc_layers = self.fc_layers.to(self.device)
        self.final_fc = self.final_fc.to(self.device)

    def forward(self, x):
        x = self.flownet3d.swap_axis_for_input(x)
        encoder_outputs = self.flownet3d.encoder(x)
        
        decoder_outputs = []
        decoder_output = encoder_outputs[-1]
        for i in range(len(self.flownet3d.decoder.upconvs)):
            decoder_output = self.flownet3d.decoder.upconvs[i](decoder_output)
            decoder_output = torch.cat([decoder_output, encoder_outputs[-(i+2)]], dim=1)
            decoder_output = F.relu(self.flownet3d.decoder.convs[i](decoder_output))
            decoder_outputs.append(decoder_output)

        # # 첫 번째 forward pass에서 레이어 초기화
        # if not self.feature_conv_layers:
        #     self._initialize_layers(decoder_outputs)

        features = []
        for i, decoder_output in enumerate(decoder_outputs):
            # 각 단계에서의 feature 추출
            attention = self.spatial_attentions[i](decoder_output)
            feature = self.feature_conv_layers[i](attention * decoder_output)
            feature = self.fc_layers[i](feature)
            features.append(feature)
        
        # 모든 feature를 종합하여 최종 스칼라 값 출력
        combined_features = torch.cat(features, dim=1)
        output = self.final_fc(combined_features)
        
        return output


def loss_function_mse(pred, target):
    loss = F.mse_loss(pred, target)
    return loss

def loss_function_bce(pred, target):
    return F.binary_cross_entropy_with_logits(pred, target)
 



#예시: 다양한 형태의 layer_configs를 입력으로 모델을 구성
# layer_configs = [[64, 2], [128, 2], [256, 2], [512, 2]]
# model = flownet3d(layer_configs, num_classes=2)

# # 테스트용 입력 데이터 생성
# x = torch.randn((2, 32, 128, 128, 1))  # Batch size: 1, Channels: 4, Depth: 32, Height: 128, Width: 128
# output = model(x)

# loss = loss_function(x, output)
# print(loss.item())

# print(output.shape)  # 출력 크기 확인