from .resnetcbamaspp_parts import *


class ResNetCBAMASPP(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', encoder_pretrained=True):
        super().__init__()

        self.encoder = ResNetEncoderWithCBAM(backbone=backbone, pretrained=encoder_pretrained)
        bottleneck_channels = self.encoder.out_channels[-1]
        self.aspp = ASPP(bottleneck_channels, 1024)

        self.decoder1 = Decoder(in_channels=1024, out_channels=512)
        self.decoder2 = Decoder(in_channels=512, out_channels=256)
        self.decoder3 = Decoder(in_channels=256, out_channels=128)
        self.decoder4 = Decoder(in_channels=128, out_channels=64)

        self.outc = OutConv(in_channels=64, out_channels=num_classes)

    def forward(self, x):
        orig_size = x.shape[2:]
        bottleneck = self.encoder(x)
        x = self.aspp(bottleneck)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)

        x = self.outc(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x