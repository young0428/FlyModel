import torch
import torch.nn as nn
import torch.nn.functional as F

class P3DBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(P3DBlock, self).__init__()
        self.downsample = downsample
        self.stride = stride
        
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=(3, 1, 1), stride=(stride, 1, 1), padding=(1, 0, 0), bias=False)
        
        self.bn1 = nn.BatchNorm3d(planes)
        self.bn2 = nn.BatchNorm3d(planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out+identity)

        return out

class P3DResNet(nn.Module):
    def __init__(self, block, input_ch, layers, num_classes=400):
        super(P3DResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(input_ch, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 레이어를 동적으로 생성합니다
        self.layers = nn.ModuleList()
        for i, [planes, num_blocks] in enumerate(layers):
            stride = 2 if i != 0 else 1  # 첫 레이어 이후의 모든 레이어는 stride 2를 사용
            self.layers.append(self._make_layer(block, planes, num_blocks, stride))
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(layers[-1][0] * block.expansion, num_classes)
        
        
        self.left_fc = nn.Sequential(
            nn.Linear(num_classes,1),
            nn.Sigmoid()
        )
        self.right_fc = nn.Sequential(
            nn.Linear(num_classes,1),
            nn.Sigmoid()
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)
    
    def swap_axis_for_input(self, t):
        return t.permute(0,4,1,2,3)

    # (b, t, h, w, c)
    # (b, c, t, h, w)
    def forward(self, input):
        x = self.swap_axis_for_input(input)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        x = F.gelu(x)

        left_pred = self.left_fc(x)
        right_pred = self.right_fc(x)
        total_pred = torch.cat((left_pred, right_pred),dim=-1)

        return total_pred

def loss_function(pred, target):
    criterion = nn.BCELoss()
    left_loss = criterion(pred[:, 0], target[:, 0])
    right_loss = criterion(pred[:, 1], target[:, 1])
    return left_loss + right_loss

def p3d_resnet(input_ch, block_list = [[64, 2], [128, 2], [256, 2], [512, 2]], num_classes=400):
    return P3DResNet(P3DBlock, input_ch, block_list, num_classes=num_classes)

if __name__ == '__main__':
    import numpy as np
    
    model = p3d_resnet(1)
    x = torch.randn(5, 32, 128, 256,1 )  # 임의의 입력 데이터 (배치 크기, 채널, 시간, 높이, 너비)
    target = torch.zeros(5, 2)
    output = model(x)
    print(np.shape(output))
    loss = loss_function(output,target)
    loss.backward()
    print(loss)





