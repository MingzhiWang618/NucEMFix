import functools
import spconv.pytorch as spconv
import torch
from collections import OrderedDict
from spconv.pytorch.modules import SparseModule
from torch import nn
from typing import Callable, Dict, List, Optional, Union
import numpy as np
import time

class MLP(nn.Sequential):
    def __init__(self, in_channels, out_channels, norm_fn=None, num_layers=2):
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(in_channels, in_channels))
            if norm_fn:
                modules.append(norm_fn(in_channels))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(in_channels, out_channels))
        return super().__init__(*modules)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self[-1].weight, 0, 0.01)
        nn.init.constant_(self[-1].bias, 0)
        
class ResidualBlock(SparseModule):

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 norm_fn: Union[Callable, Dict] = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
                 indice_key: Optional[str] = None,
                 normalize_before: bool = True):
        super().__init__()

        if in_channels == out_channels:
            self.i_branch = spconv.SparseSequential(nn.Identity())
        else:
            self.i_branch = spconv.SparseSequential(
                spconv.SubMConv3d(in_channels, out_channels, kernel_size=1, bias=False))

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop('type'))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        if normalize_before:
            self.conv_branch = spconv.SparseSequential(
                norm_fn(in_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key))
        else:
            self.conv_branch = spconv.SparseSequential(
                spconv.SubMConv3d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU(),
                spconv.SubMConv3d(
                    out_channels, out_channels, kernel_size=3, padding=1, bias=False, indice_key=indice_key),
                norm_fn(out_channels), nn.ReLU())

    def forward(self, input):
        identity = spconv.SparseConvTensor(input.features, input.indices, input.spatial_shape, input.batch_size)

        output = self.conv_branch(input)
        output = output.replace_feature(output.features + self.i_branch(identity).features)
        # output.features += self.i_branch(identity).features

        return output

class UBlock(nn.Module):

    def __init__(
        self,
        nPlanes: List[int],
        norm_fn: Union[Dict, Callable] = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1),
        block_reps: int = 2,
        block: Union[str, Callable] = ResidualBlock,
        indice_key_id: int = 1,
        normalize_before: bool = True,
        return_blocks: bool = False,
    ):

        super().__init__()

        self.return_blocks = return_blocks
        self.nPlanes = nPlanes

        # process block and norm_fn caller
        if isinstance(block, str):
            area = ['residual', 'vgg', 'asym']
            assert block in area, f'block must be in {area}, but got {block}'
            if block == 'residual':
                block = ResidualBlock

        if isinstance(norm_fn, Dict):
            norm_caller = gorilla.nn.get_torch_layer_caller(norm_fn.pop('type'))
            norm_fn = functools.partial(norm_caller, **norm_fn)

        blocks = {
            f'block{i}': block(
                nPlanes[0], nPlanes[0], norm_fn, normalize_before=normalize_before, indice_key=f'subm{indice_key_id}')
            for i in range(block_reps)
        }
        blocks = OrderedDict(blocks)
        self.blocks = spconv.SparseSequential(blocks)

        if len(nPlanes) > 1:
            if normalize_before:
                self.conv = spconv.SparseSequential(
                    norm_fn(nPlanes[0]), nn.ReLU(),
                    spconv.SparseConv3d(
                        nPlanes[0],
                        nPlanes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'))
            else:
                self.conv = spconv.SparseSequential(
                    spconv.SparseConv3d(
                        nPlanes[0],
                        nPlanes[1],
                        kernel_size=2,
                        stride=2,
                        bias=False,
                        indice_key=f'spconv{indice_key_id}'), norm_fn(nPlanes[1]), nn.ReLU())

            self.u = UBlock(
                nPlanes[1:],
                norm_fn,
                block_reps,
                block,
                indice_key_id=indice_key_id + 1,
                normalize_before=normalize_before,
                return_blocks=return_blocks)

            if normalize_before:
                self.deconv = spconv.SparseSequential(
                    norm_fn(nPlanes[1]), nn.ReLU(),
                    spconv.SparseInverseConv3d(
                        nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key=f'spconv{indice_key_id}'))
            else:
                self.deconv = spconv.SparseSequential(
                    spconv.SparseInverseConv3d(
                        nPlanes[1], nPlanes[0], kernel_size=2, bias=False, indice_key=f'spconv{indice_key_id}'),
                    norm_fn(nPlanes[0]), nn.ReLU())

            blocks_tail = {}
            for i in range(block_reps):
                blocks_tail[f'block{i}'] = block(
                    nPlanes[0] * (2 - i),
                    nPlanes[0],
                    norm_fn,
                    indice_key=f'subm{indice_key_id}',
                    normalize_before=normalize_before)
            blocks_tail = OrderedDict(blocks_tail)
            self.blocks_tail = spconv.SparseSequential(blocks_tail)

    def forward(self, input, previous_outputs: Optional[List] = None):
        output = self.blocks(input)
        identity = spconv.SparseConvTensor(output.features, output.indices, output.spatial_shape, output.batch_size)

        if len(self.nPlanes) > 1:
            output_decoder = self.conv(output)
            if self.return_blocks:
                output_decoder, previous_outputs = self.u(output_decoder, previous_outputs)
            else:
                output_decoder = self.u(output_decoder)
            output_decoder = self.deconv(output_decoder)

            output = output.replace_feature(torch.cat((identity.features, output_decoder.features), dim=1))
            # output.features = torch.cat((identity.features, output_decoder.features), dim=1)

            output = self.blocks_tail(output)

        if self.return_blocks:
            # NOTE: to avoid the residual bug
            if previous_outputs is None:
                previous_outputs = []
            previous_outputs.append(output)
            return output, previous_outputs
        else:
            return output

# def test_sparse_ublock_final():

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"--- Environment ---")
#     print(f"Device: {device}")
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")

#     batch_size = 2
#     in_channels = 3

    

#     spatial_shape = [128, 640, 320] 
#     print(f"Spatial Shape (32-aligned): {spatial_shape}")

#     print(f"Generating {num_points} random points...")

#     features = torch.randn(num_points, in_channels).to(device).float()
    

#     z = torch.randint(0, 100, (num_points, 1))
#     y = torch.randint(0, 600, (num_points, 1))
#     x = torch.randint(0, 300, (num_points, 1))
#     batch_idx = torch.randint(0, batch_size, (num_points, 1))
    
#     indices = torch.cat([batch_idx, z, y, x], dim=1).int().to(device)

#     input_tensor = spconv.SparseConvTensor(
#         features=features,
#         indices=indices,
#         spatial_shape=spatial_shape,
#         batch_size=batch_size
#     )

#     nPlanes = [in_channels, 32, 64, 128]
    
#     try:
#         model = UBlock(
#             nPlanes=nPlanes,
#             block_reps=2,
#             block=ResidualBlock,
#             normalize_before=True
#         ).to(device)
#         model.eval()
#     except NameError:
#         print("Error: UBlock class not found. Please make sure UBlock is defined in your script.")
#         return

#     print("Starting forward pass...")
#     if device.type == 'cuda':
#         torch.cuda.synchronize()
#         torch.cuda.reset_peak_memory_stats()
    
#     start_time = time.time()
    
#     with torch.no_grad():
#         try:
#             output = model(input_tensor)
            
#             if device.type == 'cuda':
#                 torch.cuda.synchronize()
#                 peak_mem = torch.cuda.max_memory_allocated() / 1024**2
#                 print(f"--- Result ---")
#                 print(f"Forward Success!")
#                 print(f"Time Taken: {time.time() - start_time:.4f}s")
#                 print(f"Peak Memory: {peak_mem:.2f} MB")
#                 print(f"Output Features Shape: {output.features.shape}")
                

#                 assert output.features.shape[0] == num_points, "Point count mismatch!"
#                 print("Consistency Check: OK")
                
#         except Exception as e:
#             print(f"Forward failed with error: {e}")
#             import traceback
#             traceback.print_exc()

# if __name__ == "__main__":

#     test_sparse_ublock_final()