import torch
import torch.nn as nn

class WrinkleUNet(nn.Module):
    def __init__(self, in_channels=2, base_feat=16):
        super(WrinkleUNet, self).__init__()
        

        self.enc1 = self._conv_block(in_channels, base_feat)
        self.enc2 = self._conv_block(base_feat, base_feat * 2)
        self.enc3 = self._conv_block(base_feat * 2, base_feat * 4)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        

        self.bottleneck = self._conv_block(base_feat * 4, base_feat * 8)
        

        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec3 = self._conv_block(base_feat * 8 + base_feat * 4, base_feat * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec2 = self._conv_block(base_feat * 4 + base_feat * 2, base_feat * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.dec1 = self._conv_block(base_feat * 2 + base_feat, base_feat)
        

        self.mask_head = nn.Sequential(
            nn.Conv3d(base_feat, 1, kernel_size=1),
            nn.Sigmoid() 
        )
        

        self.confidence_head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(base_feat * 8, base_feat * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base_feat * 2, 1),
            nn.Sigmoid() 
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, img, sub_nuclei):

        x = torch.cat([img, sub_nuclei], dim=1)
        
        # Encoder
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        
        # Bottleneck
        b = self.bottleneck(self.pool(s3))
        
        # Branch 1: Confidence
        conf = self.confidence_head(b)
        
        # Branch 2: Mask Decoder
        d3 = self.dec3(torch.cat([self.up3(b), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        
        pred_mask = self.mask_head(d1)

        return pred_mask, conf