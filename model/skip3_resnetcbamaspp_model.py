from .skip3_resnetcbamaspp_parts import *

class ResNetCBAMASPP(nn.Module):
    def __init__(self, num_classes, backbone='resnet50', encoder_pretrained=True):
        super().__init__()

        # Encoder
        self.encoder = ResNetEncoderWithCBAM(backbone=backbone, pretrained=encoder_pretrained)

        # Bottleneck + ASPP
        bottleneck_channels = self.encoder.out_channels[-1]
        self.aspp = ASPP(bottleneck_channels, 2048)

        # decoder blocks
        self.decoder1 = Decoder(2048, 1024, 1024)
        self.decoder2 = Decoder(1024, 512, 512)
        self.decoder3 = Decoder(512, 256, 256)
        self.decoder4 = Decoder(256, 0, 64)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        orig_size = x.shape[2:]
        bottleneck, skips = self.encoder(x)
        x = self.aspp(bottleneck)
        x = self.decoder1(x, skips[2])  # skip3 (layer3 1024) 
        x = self.decoder2(x, skips[1])  # skip2 (layer2 512)
        x = self.decoder3(x, skips[0])  # skip1 (layer1 256)
        x = self.decoder4(x, None)  # no skip
        x = self.outc(x)
        x = F.interpolate(x, size=orig_size, mode='bilinear', align_corners=False)
        return x