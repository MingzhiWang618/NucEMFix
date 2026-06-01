import torch
import torch.nn as nn
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

class PointNet2Backbone(nn.Module):
    def __init__(self, input_channels=0, out_channels=128):
        super().__init__()
        

        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(PointnetSAModule(npoint=1024, radius=0.1, nsample=32, mlp=[input_channels, 32, 32, 64], use_xyz=True))    # SA1
        self.SA_modules.append(PointnetSAModule(npoint=256, radius=0.2, nsample=32, mlp=[64, 64, 64, 128], use_xyz=True))                 # SA2
        self.SA_modules.append(PointnetSAModule(npoint=64, radius=0.4, nsample=32, mlp=[128, 128, 128, 256], use_xyz=True))               # SA3
        self.SA_modules.append(PointnetSAModule(npoint=16, radius=0.8, nsample=32, mlp=[256, 256, 256, 512], use_xyz=True))               # SA4

        self.FP_modules = nn.ModuleList()

        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256])) 

        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 128])) 

        self.FP_modules.append(PointnetFPModule(mlp=[128 + 64, 128, 128]))  

        self.FP_modules.append(PointnetFPModule(mlp=[128 + input_channels, 128, 128, out_channels]))

    def forward(self, points):
        xyz = points.contiguous()
        features = None

        l_xyz, l_features = [xyz], [features]
        
        # Encoder
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        curr_features = l_features[-1]
        for i in range(len(self.FP_modules)):

            idx_low = len(self.SA_modules) - 1 - i # 3, 2, 1, 0
            idx_high = idx_low + 1                 # 4, 3, 2, 1
            
            curr_features = self.FP_modules[i](
                l_xyz[idx_low], l_xyz[idx_high], l_features[idx_low], curr_features
            )

        return curr_features # [B, out_channels, N]