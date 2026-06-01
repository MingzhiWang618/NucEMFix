import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv3D -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet3D_BC(nn.Module):
    """
    """
    def __init__(self, in_channels=1, n_classes=3, base_filters=32):
        super(UNet3D_BC, self).__init__()
        
        self.n_channels = in_channels
        self.n_classes = n_classes

        # Input: [B, 1, D, H, W] -> [B, 32, D, H, W]
        self.inc = DoubleConv(in_channels, base_filters)
        
        # [B, 32, D, H, W] -> [B, 64, D/2, H/2, W/2]
        self.down1 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_filters, base_filters * 2)
        )
        
        # [B, 64, D/2, H/2, W/2] -> [B, 128, D/4, H/4, W/4]
        self.down2 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_filters * 2, base_filters * 4)
        )
        
        # [B, 128, D/4, H/4, W/4] -> [B, 256, D/8, H/8, W/8]
        self.down3 = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(base_filters * 4, base_filters * 8)
        )

        
        # Upsample 1: [B, 256, ...] -> [B, 128, ...]
        self.up1 = nn.ConvTranspose3d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(base_filters * 8, base_filters * 4)
        
        # Upsample 2: [B, 128, ...] -> [B, 64, ...]
        self.up2 = nn.ConvTranspose3d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(base_filters * 4, base_filters * 2)
        
        # Upsample 3: [B, 64, ...] -> [B, 32, ...]
        self.up3 = nn.ConvTranspose3d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(base_filters * 2, base_filters)

        # ---- Output Layer ----

        self.outc = nn.Conv3d(base_filters, n_classes, kernel_size=1)

    def forward(self, x):
        # x shape: [Batch, Channel, Z, Y, X]
        
        # Encoding
        x1 = self.inc(x)            # Level 1 features (skip connection 1)
        x2 = self.down1(x1)         # Level 2 features (skip connection 2)
        x3 = self.down2(x2)         # Level 3 features (skip connection 3)
        x4 = self.down3(x3)         # Bottleneck features
        
        # Decoding
        x = self.up1(x4)

        x = torch.cat([x3, x], dim=1) 
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv_up3(x)
        
        # Output
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet3D_BC(in_channels=1, n_classes=3).to(device)
    

    dummy_input = torch.randn(1, 1, 64, 128, 128).to(device)
    
    print(f"Model created. Testing with input shape: {dummy_input.shape}")
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
