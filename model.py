"""Custom lane-segmentation U-Net, trained from scratch (no pretrained weights).
Input : (B, 3, 48, 48) RGB in [0, 1]
Output: (B, 1, 48, 48) logits  -> sigmoid > 0.5 = lane mask
"""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    # GroupNorm (not BatchNorm) because batch size is only 2-4
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.GroupNorm(8, cout), nn.ReLU(inplace=True),
    )


class LaneUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8      # 16, 32, 64, 128
        self.enc1 = conv_block(3, c1)       # 48x48
        self.enc2 = conv_block(c1, c2)      # 24x24
        self.enc3 = conv_block(c2, c3)      # 12x12
        self.bott = conv_block(c3, c4)      # 6x6
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c1 * 2, c1)
        self.head = nn.Conv2d(c1, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bott(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.head(d1)


if __name__ == "__main__":
    m = LaneUNet()
    print(m(torch.zeros(2, 3, 48, 48)).shape, sum(p.numel() for p in m.parameters()), "params")
