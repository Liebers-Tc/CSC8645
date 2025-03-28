import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ (3*3 Convolution -> ReLU) * 2 -> 2*2 Maxpooling """

    def __init__(self, in_channels, out_channels, maxpooling=True):
        super().__init__()

        self.encoder = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            ])
        if maxpooling:
            self.encoder.append(nn.MaxPool2d(2))

    def forward(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x



class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.decoder = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            ])

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        for layer in self.decoder:
            x = layer(x)
        return x


class OutConv(nn.Module):
    """ Output layer """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.outconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.outconv(x)
