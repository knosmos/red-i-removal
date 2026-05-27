import numpy as np
import torch
import torch.nn as nn

class RedIRemover(nn.Module):
    """
    Unet-like architecture for removing red 'i' from images. The model
    takes an input image and outputs a cleaned image without the red 'i'.

    This implementation adds a skip connection from the first encoder
    block to the decoder (concatenation), so the decoder receives
    high-resolution features from the encoder.
    """
    def __init__(self):
        super(RedIRemover, self).__init__()
        # Encoder blocks (exposed so we can create skip connections)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Decoder: after upsampling we concatenate the encoder features
        # (skip connection) so the conv expects 128 channels (64 + 64)
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Encoder forward (save first block output for skip connection)
        e1 = self.enc1(x)      # [B,64,H,W]
        e2 = self.enc2(e1)     # [B,128,H,W]
        p = self.pool(e2)      # [B,128,H/2,W/2]

        # Decoder forward
        u = self.up(p)         # [B,64,H,W]

        # If spatial sizes don't match exactly due to rounding, center-crop
        # the encoder feature to match the upsampled size.
        if e1.size(2) != u.size(2) or e1.size(3) != u.size(3):
            diff_y = e1.size(2) - u.size(2)
            diff_x = e1.size(3) - u.size(3)
            e1 = e1[:, :, diff_y // 2: e1.size(2) - (diff_y - diff_y // 2),
                    diff_x // 2: e1.size(3) - (diff_x - diff_x // 2)]

        # Concatenate along channel dimension
        cat = torch.cat([u, e1], dim=1)  # [B,128,H,W]
        out = self.dec_conv(cat)
        return out