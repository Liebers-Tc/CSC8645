import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CBAM(nn.Module):
    """ 增强 / 引导特征 """
    def __init__(self, channels, reduction=16, kernel_size=7):  # reduction
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
            )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
            )

    def forward(self, x):
        # Channel Attention
        avg_pool = F.adaptive_avg_pool2d(x, 1)
        max_pool = F.adaptive_max_pool2d(x, 1)
        channel_att = self.channel_attention(avg_pool) + self.channel_attention(max_pool)
        x = x * torch.sigmoid(channel_att)

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_attention(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class ASPP(nn.Module):
    """  提取 / 捕捉多尺度语义 """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0, dilation=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )
        self.conv_output = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5)
            )

    def forward(self, x):
        size = x.shape[2:]
        image_features = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=False)
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x = torch.cat([x1, x2, x3, x4, image_features], dim=1)
        return self.conv_output(x)
    

class ResNetEncoderWithCBAM(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        assert hasattr(models, backbone), f"{backbone} not in torchvision.models"
        resnet = getattr(models, backbone)(pretrained=pretrained)

        # Save the number of output channels per layer and adapt to different ResNet version
        self.out_channels = [
        resnet.layer1[-1].bn3.num_features if hasattr(resnet.layer1[-1], 'bn3') else resnet.layer1[-1].bn2.num_features,
        resnet.layer2[-1].bn3.num_features if hasattr(resnet.layer2[-1], 'bn3') else resnet.layer2[-1].bn2.num_features,
        resnet.layer3[-1].bn3.num_features if hasattr(resnet.layer3[-1], 'bn3') else resnet.layer3[-1].bn2.num_features,
        resnet.layer4[-1].bn3.num_features if hasattr(resnet.layer4[-1], 'bn3') else resnet.layer4[-1].bn2.num_features,
        ]

        self.initial = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            )
        self.pooling = resnet.maxpool
        self.layer1 = nn.Sequential(resnet.layer1, CBAM(self.out_channels[0]))
        self.layer2 = nn.Sequential(resnet.layer2, CBAM(self.out_channels[1]))
        self.layer3 = nn.Sequential(resnet.layer3, CBAM(self.out_channels[2]))
        self.layer4 = nn.Sequential(resnet.layer4, CBAM(self.out_channels[3]))

    def forward(self, x):
        skips = []
        x = self.initial(x)
        skips.append(x)
        x = self.pooling(x)
        x = self.layer1(x)
        skips.append(x)
        x = self.layer2(x)
        skips.append(x)
        x = self.layer3(x)
        skips.append(x)
        x = self.layer4(x)
        return x, skips


class Decoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        self.doubleconv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
            )

    def forward(self, x, skip):
        x = self.up(x)
        # if x.size()[2:] != skip.size()[2:]:
        #     x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.doubleconv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(x)