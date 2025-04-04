from .skip4_resnetcbamaspp_parts import *

class ResNetCBAMASPP(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', encoder_pretrained=True):
        super().__init__()

        # Encoder
        self.encoder = ResNetEncoderWithCBAM(backbone=backbone, pretrained=encoder_pretrained)

        # Bottleneck + ASPP
        bottleneck_channels = self.encoder.out_channels[-1]
        self.aspp = ASPP(bottleneck_channels, 1024)

        # decoder blocks
        self.decoder1 = Decoder(1024, 1024, 512)
        self.decoder2 = Decoder(512, 512, 256)
        self.decoder3 = Decoder(256, 256, 128)
        self.decoder4 = Decoder(128, 64, 128)
        self.outc = OutConv(128, num_classes)

    def forward(self, x):
        orig_size = x.shape[2:]
        bottleneck, skips = self.encoder(x)
        x = self.aspp(bottleneck)
        x = self.decoder1(x, skips[3])  # skip3 (layer3 1024) 
        x = self.decoder2(x, skips[2])  # skip2 (layer2 512)
        x = self.decoder3(x, skips[1])  # skip1 (layer1 256)
        x = self.decoder4(x, skips[0])  # skip0 (initial 64)
        x = self.outc(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x