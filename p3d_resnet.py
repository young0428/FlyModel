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
        self.dropout = nn.Dropout(p=0.2)
        self.image_dropout = nn.Dropout3d(p=0.1)
        
        
        self.fc_binary = nn.Sequential(
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
        x = self.image_dropout(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for layer in self.layers:
            x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.dropout(x)
        
        x = self.fc_binary(x)



        # left_pred = self.left_fc(x)
        # right_pred = self.right_fc(x)
        # total_pred = torch.cat((left_pred, right_pred),dim=-1)

        return x
    
import torch
import torch.nn as nn

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet3D(nn.Module):
    def __init__(self, block, layers, num_classes=400):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc_binary = nn.Sequential(
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
    def forward(self, x):
        x= self.swap_axis_for_input(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        x = self.fc_binary(x)

        return x

def resnet3d18(num_classes=400):
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2], num_classes=num_classes)

def resnet3d34(num_classes=400):
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes)

def resnet3d50(num_classes=400):
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3], num_classes=num_classes)

def resnet3d101(num_classes=400):
    return ResNet3D(BasicBlock3D, [3, 4, 23, 3], num_classes=num_classes)

def resnet3d152(num_classes=400):
    return ResNet3D(BasicBlock3D, [3, 8, 36, 3], num_classes=num_classes)



def loss_function(pred, target):
    criterion = nn.BCELoss()
    loss = criterion(pred, target)
    #right_loss = criterion(pred[:, 1], target[:, 1])
    return loss

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





