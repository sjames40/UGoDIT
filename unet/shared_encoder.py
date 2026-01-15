""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *
import torch
import torch.nn as nn

# Assuming DoubleConv, Down, Up, and OutConv are defined elsewhere
import torch
import torch.nn as nn

# Assume DoubleConv, Down, Up, and OutConv are defined elsewhere.

class SharedEncoder(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(SharedEncoder, self).__init__()
        self.factor = 2 if bilinear else 1
        self.inc = DoubleConv(n_channels, 64)       # x1: 64 channels
        self.down1 = Down(64, 128)                  # x2: 128 channels
        self.down2 = Down(128, 256)                 # x3: 256 channels
        # FIX: Use 512//factor so that if bilinear==True, output channels = 512//2 = 256.
        self.down3 = Down(256, 512 // self.factor)  # x4: 256 channels (if bilinear=True)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        # We return x4 as the deepest feature and also pass along the skip connections.
        # In the original MediumUNet forward, note that x3 (256 ch) was used as the “deep” feature.
        # Here, x4 now has 256 channels—matching that design.
        return x4, (x3, x2, x1)

class DecoderBlock(nn.Module):
    def __init__(self, n_classes, bilinear=True):
        super(DecoderBlock, self).__init__()
        factor = 2 if bilinear else 1
        # The first Up block receives:
        #   - the deepest feature x_enc from SharedEncoder (x4): 256 channels (if bilinear True)
        #   - the corresponding skip x3: 256 channels
        # Their concatenation is 512 channels.
        self.up1 = Up(512, 256 // factor, bilinear)  # Expected: 512 in, 256//factor out.
        # Next, up2 receives the output of up1 and skip from down1 (128 channels).
        self.up2 = Up((256 // factor) + 128, 128 // factor, bilinear)
        # Finally, up3 receives the output of up2 and skip from inc (64 channels).
        self.up3 = Up((128 // factor) + 64, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x_enc, skips):
        x3, x2, x1 = skips  # Note: the ordering matches how they were produced in SharedEncoder.
        x = self.up1(x_enc, x3)  # Concatenates (256 + 256 = 512 channels) → conv expects 512.
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        return self.outc(x)

class MultiDecoderUNet(nn.Module):
    def __init__(self, n_channels, n_classes, num_decoders=2, bilinear=True):
        super(MultiDecoderUNet, self).__init__()
        self.encoder = SharedEncoder(n_channels, bilinear)
        self.decoders = nn.ModuleList([
            DecoderBlock(n_classes, bilinear)
            for _ in range(num_decoders)
        ])

    def forward(self, x):
        x_enc, skips = self.encoder(x)
        outputs = []
        for decoder in self.decoders:
            outputs.append(decoder(x_enc, skips))
        return outputs


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels, n_classes, bilinear=True):
#         super(DecoderBlock, self).__init__()
#         factor = 2 if bilinear else 1
#         self.up1 = Up(in_channels, 512 // factor, bilinear)
#         self.up2 = Up(512, 256 // factor, bilinear)
#         self.up3 = Up(256, 128 // factor, bilinear)
#         self.up4 = Up(128, 64, bilinear)
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x_enc, x4, x3, x2, x1):
#         # Note: the order and number of skip connections here depends on your encoder.
#         x = self.up1(x_enc, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         return self.outc(x)

# class MultiDecoderUNet(nn.Module):
#     def __init__(self, n_channels, n_classes, num_decoders=2, bilinear=True):
#         super(MultiDecoderUNet, self).__init__()
#         factor = 2 if bilinear else 1

#         # Shared encoder (the downsampling path)
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512 // factor)
#         self.down4 = Down(512, 1024 // factor)

#         # Create multiple decoders (the upsampling path) with their own heads
#         self.decoders = nn.ModuleList([
#             DecoderBlock(1024 , n_classes, bilinear)
#             for _ in range(num_decoders)
#         ])

#     def forward(self, x):
#         # Run the shared encoder once.
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         #x_enc = self.down4(x4)

#         # For each decoder, pass the shared features along with the skip connections.
#         outputs = []
#         for decoder in self.decoders:
#             # The skip connections (x4, x3, x2, x1) are provided to each decoder.
#             out = decoder(x4, x3, x2, x1)
#             outputs.append(out)
#         return outputs

