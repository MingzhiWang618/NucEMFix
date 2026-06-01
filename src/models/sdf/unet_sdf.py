import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv3D => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SDFWrinkleNet(nn.Module):
    def __init__(self, in_channels=2, base_feat=32, bilinear=True):
        """
        SDFWrinkleNet:
        """
        super(SDFWrinkleNet, self).__init__()
        self.in_channels = in_channels
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(in_channels, base_feat)
        self.down1 = Down(base_feat, base_feat * 2)
        self.down2 = Down(base_feat * 2, base_feat * 4)
        self.down3 = Down(base_feat * 4, base_feat * 8)
        self.down4 = Down(base_feat * 8, (base_feat * 16) // factor)

        self.up1 = Up(base_feat * 16, (base_feat * 8) // factor, bilinear)
        self.up2 = Up(base_feat * 8, (base_feat * 4) // factor, bilinear)
        self.up3 = Up(base_feat * 4, (base_feat * 2) // factor, bilinear)
        self.up4 = Up(base_feat * 2, base_feat, bilinear)

        

        self.sdf_head = nn.Conv3d(base_feat, 1, kernel_size=1)
        

        self.mask_head = nn.Conv3d(base_feat, 1, kernel_size=1)
        

        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear((base_feat * 16) // factor, base_feat * 4),
            nn.ReLU(inplace=True),
            nn.Linear(base_feat * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, img, sub_nuclei):

        x_in = torch.cat([img, sub_nuclei], dim=1)
        

        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        

        conf = self.confidence_head(x5)
        

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        

        pred_sdf = self.sdf_head(x)
        pred_mask = self.mask_head(x)
        
        return pred_sdf, pred_mask, conf

# def get_model(base_feat=32):
#     return SDFWrinkleNet(n_channels=2, base_feat=base_feat, bilinear=True)