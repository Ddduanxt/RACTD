from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False,
            dropout_rate=.2):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

        self.scale_dropout = nn.Dropout(dropout_rate)
        self.bias_dropout = nn.Dropout(dropout_rate)


    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        out = self.scale_dropout(out)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        out = self.bias_dropout(out)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        dropout_rate=.0,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        self.dropout_rate = dropout_rate

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))



        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale, dropout_rate=dropout_rate),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))


        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )


    def prepare_drop_generators(self):
        dropout_generator = torch.Generator().manual_seed(42)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.generator = dropout_generator
                if self.dropout_rate == 0.0:
                    module.generator = None


    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int],
            stoptime = None, 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x




class RewardResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels,
            dropout_rate=0.2, 
            kernel_size=3,
            n_groups=8):
        super().__init__()

        # self.blocks = nn.Sequential(
        #     nn.Linear(in_channels,out_channels),
        #     nn.Mish(),
        #     nn.Linear(out_channels, out_channels),
        # )
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        self.out_channels = out_channels

        # self.scale_dropout = nn.Dropout(dropout_rate)
        # self.bias_dropout = nn.Dropout(dropout_rate)

        # # make sure dimensions compatible
        # self.residual_linear = nn.Linear(in_channels, out_channels) \
        #     if in_channels != out_channels else nn.Identity()
        # self.time_mlp = nn.Sequential(
        #     nn.Mish(),
        #     nn.Linear(embed_dim, out_channels),
        #     Rearrange('batch t -> batch t 1'),
        # )

        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        '''
            x : [ batch_size x in_channels ]

            returns:
            out : [ batch_size x out_channels ]
        '''
        # out = self.blocks(x)
        # out = self.scale_dropout(out)
        # out = out + self.residual_linear(x)
        # out = self.bias_dropout(out)
        # return out
        out = self.blocks[0](x)
        # out = self.scale_dropout(out)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        # out = self.bias_dropout(out)
        return out
        


class ValueUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        # horizon, 
        kernel_size=5,
        diffusion_step_embed_dim=32,
        down_dims=[32,64,128,256],
        dropout_rate=.0,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        end_dim = down_dims[-1]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        
        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                RewardResidualBlock1D(dim_in, dim_out, kernel_size=kernel_size),
                RewardResidualBlock1D(dim_out, dim_out, kernel_size=kernel_size),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            # if not is_last:
            #     horizon = horizon // 2

        mid_dim = end_dim
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        out_dim = 1
        ##
        self.mid_block1 = RewardResidualBlock1D(mid_dim, mid_dim_2, kernel_size=kernel_size)
        self.mid_down1 = Downsample1d(mid_dim_2)
        # horizon = horizon // 2
        ##
        self.mid_block2 = RewardResidualBlock1D(mid_dim_2, mid_dim_3, kernel_size=kernel_size)
        self.mid_down2 = Downsample1d(mid_dim_3)
        # horizon = horizon // 2
        ##
        # fc_dim = mid_dim_3 * max(horizon, 1)
        fc_dim = mid_dim_3 

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

        # self.scale_dropout = nn.Dropout(dropout_rate)
        # self.bias_dropout = nn.Dropout(dropout_rate)

        # output_dim = 1
        # final_linear = nn.Linear(end_dim, output_dim)

        # self.blocks = blocks
        # self.final_linear = final_linear

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, **kwargs):
        """
        x: (B,input_dim) torch.Size([1024, 14])
        output: (B,1)
        """
        # # x = einops.rearrange(sample, 'b h t -> (b h) t')
        ## torch.Size([1024, 1, 14])
        x = sample[:, None, :]
        x = einops.rearrange(x, 'b h t -> b t h')
        # for idx, (resnet, resnet2) in enumerate(self.blocks):
        #     x = resnet(x)
        #     x = resnet2(x)

        # x = self.final_linear(x)

        # # x = einops.rearrange(x, 'bh t -> b h t', b=len(sample))

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x)
            x = resnet2(x)
            x = downsample(x)

        ##
        x = self.mid_block1(x)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(x)

        return out