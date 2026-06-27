import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights, resnet50

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

class ResNet50_FPN(nn.Module):
    def __init__(self, block=None, pretrained=True):
        super(ResNet50_FPN, self).__init__()
        weights = None
        if pretrained:
            try:
                weights = ResNet50_Weights.DEFAULT
            except Exception as exc:
                print(f"[ResNet50_FPN] pretrained weight unavailable, fallback to scratch: {exc}")
                weights = None

        try:
            backbone = resnet50(weights=weights)
        except Exception as exc:
            print(f"[ResNet50_FPN] failed to load pretrained weight, fallback to scratch: {exc}")
            backbone = resnet50(weights=None)
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1   # C2
        self.layer2 = backbone.layer2   # C3
        self.layer3 = backbone.layer3   # C4
        self.layer4 = backbone.layer4   # C5

        # --- 1x1 Conv (채널 256 통일) ---
        self.lateral4 = nn.Conv2d(2048, 256, 1)
        self.lateral3 = nn.Conv2d(1024, 256, 1)
        self.lateral2 = nn.Conv2d(512, 256, 1)
        self.lateral1 = nn.Conv2d(256, 256, 1)

        # --- ★ 피드백 반영: 3x3 Conv (Aliasing 방지 및 정제) ---
        self.output4 = nn.Conv2d(256, 256, 3, padding=1)
        self.output3 = nn.Conv2d(256, 256, 3, padding=1)
        self.output2 = nn.Conv2d(256, 256, 3, padding=1)

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
    model = ResNet50_FPN(pretrained=False).to(device)
    imgs = torch.randn(3, 3, 256, 704).to(device)
    
    features = model(imgs)
    
    print("🚀 멀티스케일 특징 도서관(FPN) 생성 완료!")
    for i, feat in enumerate(features):
        print(f"P{i+2} 스케일 크기: {feat.shape}")
