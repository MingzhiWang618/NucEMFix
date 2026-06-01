import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import itertools

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
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
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

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class LossPredictionHead(nn.Module):
    """MLP to predict loss from concatenated global features"""
    def __init__(self, in_channels, hidden_dim=256):
        super(LossPredictionHead, self).__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        x_from_unet = [self.pool(f).flatten(1) for f in features]
        x = torch.cat(x_from_unet, dim=1)
            
        return self.mlp(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels=2, n_classes=1, bilinear=True, 
                 feature_layers_to_use=['x_up4']):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.feature_layers_to_use = feature_layers_to_use

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        channel_map = {
            'x1': 64, 'x2': 128, 'x3': 256, 'x4': 512, 'x5': 1024 // factor,
            'x_up1': 512 // factor, 'x_up2': 256 // factor, 'x_up3': 128 // factor,
            'x_up4': 64
        }
        in_channels_for_head = sum(channel_map[layer] for layer in self.feature_layers_to_use)

        self.loss_prediction_head = LossPredictionHead(in_channels_for_head, hidden_dim=256)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder
        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        logits = self.outc(x_up4)

        feature_dict = {
            'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5,
            'x_up1': x_up1, 'x_up2': x_up2, 'x_up3': x_up3, 'x_up4': x_up4
        }
        features = [feature_dict[layer] for layer in self.feature_layers_to_use]
        
        predicted_loss = self.loss_prediction_head(features)

        return logits, predicted_loss

def get_model(feature_layers=None):
    """
    """
    if feature_layers is not None:
        model = UNet3D(n_channels=2, n_classes=1, bilinear=True, feature_layers_to_use=feature_layers)
    else:
        model = UNet3D(n_channels=2, n_classes=1, bilinear=True)
    return model