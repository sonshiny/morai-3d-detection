import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Bottleneck (변함없음)
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# 2. 피드백이 반영된 완벽한 ResNet50_FPN
class ResNet50_FPN(nn.Module):
    def __init__(self, block):
        super(ResNet50_FPN, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 3, stride=1)   # C2
        self.layer2 = self._make_layer(block, 128, 4, stride=2)  # C3
        self.layer3 = self._make_layer(block, 256, 6, stride=2)  # C4
        self.layer4 = self._make_layer(block, 512, 3, stride=2)  # C5

        # --- 1x1 Conv (채널 256 통일) ---
        self.lateral4 = nn.Conv2d(2048, 256, 1)
        self.lateral3 = nn.Conv2d(1024, 256, 1)
        self.lateral2 = nn.Conv2d(512, 256, 1)
        self.lateral1 = nn.Conv2d(256, 256, 1)

        # --- ★ 피드백 반영: 3x3 Conv (Aliasing 방지 및 정제) ---
        self.output4 = nn.Conv2d(256, 256, 3, padding=1)
        self.output3 = nn.Conv2d(256, 256, 3, padding=1)
        self.output2 = nn.Conv2d(256, 256, 3, padding=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, planes, stride, downsample))
        self.in_channels = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        c1 = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        # Top-down FPN
        p5 = self.lateral4(c5)
        p4 = self.lateral3(c4) + F.interpolate(p5, scale_factor=2, mode='nearest')
        p3 = self.lateral2(c3) + F.interpolate(p4, scale_factor=2, mode='nearest')
        p2 = self.lateral1(c2) + F.interpolate(p3, scale_factor=2, mode='nearest')

        # ★ 피드백 반영: 멀티스케일 전체 리턴 및 3x3 정제 거치기
        return [self.output2(p2), self.output3(p3), self.output4(p4), p5]

# --- 테스트 ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50_FPN(Bottleneck).to(device)
    imgs = torch.randn(6, 3, 224, 224).to(device)
    
    features = model(imgs)
    
    print("🚀 멀티스케일 특징 도서관(FPN) 생성 완료!")
    for i, feat in enumerate(features):
        print(f"P{i+2} 스케일 크기: {feat.shape}")
