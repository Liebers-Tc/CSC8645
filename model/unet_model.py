from .unet_parts import *
from torch.utils.checkpoint import checkpoint_sequential


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=104, base_channels=64, bilinear=False):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = Encoder(in_channels, base_channels)              # 3 -> 64
        self.enc2 = Encoder(base_channels, base_channels * 2)        # 64 -> 128
        self.enc3 = Encoder(base_channels * 2, base_channels * 4)    # 128 -> 256
        self.enc4 = Encoder(base_channels * 4, base_channels * 8)    # 256 -> 512

        # Bottleneck without pooling
        self.bottleneck = Encoder(base_channels * 8, base_channels * 16, maxpooling=False)  # 512 -> 1024

        # Decoder with skip connection
        self.dec4 = Decoder(base_channels * 16, base_channels * 8, bilinear=bilinear)  # 1024 + 512 -> 512
        self.dec3 = Decoder(base_channels * 8, base_channels * 4, bilinear=bilinear)   # 512 + 256 -> 256
        self.dec2 = Decoder(base_channels * 4, base_channels * 2, bilinear=bilinear)   # 256 + 128 -> 128
        self.dec1 = Decoder(base_channels * 2, base_channels, bilinear=bilinear)       # 128 + 64 -> 64

        # Output
        self.out_conv = OutConv(base_channels, num_classes)  # 64 -> num_classes

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Bottleneck
        x5 = self.bottleneck(x4)

        # Decoder
        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)

        return self.out_conv(x)

    def use_checkpointing(self, segments=2):
        """ enable gradient checkpointing """

        def apply_checkpoint(module):
            if isinstance(module, nn.ModuleList):
                return checkpoint_sequential(module, segments)
            elif hasattr(module, 'encoder'):
                module.encoder = checkpoint_sequential(module.encoder, segments)
            elif hasattr(module, 'decoder'):
                module.decoder = checkpoint_sequential(module.decoder, segments)

        for m in [self.enc1, self.enc2, self.enc3, self.enc4, self.bottleneck,
                  self.dec1, self.dec2, self.dec3, self.dec4]:
            apply_checkpoint(m)