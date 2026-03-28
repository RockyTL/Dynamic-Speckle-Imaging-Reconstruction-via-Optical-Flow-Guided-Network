import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se = SELayer(out_channels)


    def forward(self, x):
        return  self.se(self.double_conv(x))


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.relu(self.bn(self.up(x1)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//2, out_channels, kernel_size=1),
            # nn.Tanh()

            # nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.Sigmoid()
        )

    def forward(self, x):
        return self.out_conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels = 1, init_feature = 32):
        super(UNet, self).__init__()
        self.down1 = DoubleConv(in_channels, init_feature)
        self.down2 = Down(init_feature, init_feature * 2)
        self.down3 = Down(init_feature * 2, init_feature * 4)
        self.down4 = Down(init_feature * 4, init_feature * 8)
        self.down5 = Down(init_feature * 8, init_feature * 16)
        self.up4 = Up(init_feature * 16, init_feature * 8)
        self.up3 = Up(init_feature * 8, init_feature * 4)
        self.up2 = Up(init_feature * 4, init_feature * 2)
        self.up1 = Up(init_feature * 2, init_feature)
        self.out = OutConv(init_feature, 1)

    def forward(self, x):
        # Downsampling path
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        # Upsampling path
        x6 = self.up4(x5, x4)
        x7 = self.up3(x6, x3)
        x8 = self.up2(x7, x2)
        x9 = self.up1(x8, x1)

        out = self.out(x9)

        return out

