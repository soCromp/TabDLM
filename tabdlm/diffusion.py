from typing import Callable, Union

import torch
import torch.nn as nn
import torch.optim
from torch.nn import Embedding, ModuleDict, Sigmoid, Sequential, Linear, Dropout
from .noise_schedule import PowerMeanNoise, PowerMeanNoise_PerColumn

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class Precond(nn.Module):
    def __init__(self,
                 sigma_data=0.5,  # Expected standard deviation of the training data.
                 net_conditioning="sigma",
                 ):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning

    def forward(self, x_num, t, sigma):

        x_num = x_num.to(torch.float32)

        sigma = sigma.to(torch.float32)
        # assert sigma.ndim == 2
        # if sigma.dim() > 1:  # if learnable column-wise noise schedule, sigma conditioning is set to the defaults schedule of rho=7
        #     sigma_cond = (0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))).pow(7)
        # else:
        #     sigma_cond = sigma
        sigma_cond = sigma
        dtype = torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4

        x_in = c_in * x_num
        if self.net_conditioning == "sigma":
            return c_skip, c_out, x_in, c_noise.flatten()
        elif self.net_conditioning == "t":
            return c_skip, c_out, x_in, t


class diffusion_model(nn.Module):
    def __init__(self,
                 sigma_data=0.5,  # Expected standard deviation of the training data.
                 net_conditioning="sigma",
                 input_dim=512,
                 output_dim=4096,
                 scheduler='power_mean_per_column',
                 tgt_rms=None,
                 noise_schedule_params={},
                 num_numerical_features=6,
                 edm_params={},
                 floatenc=None,
                 floatdec=None,
                 dropout=0.2,
                 ):
        super(diffusion_model, self).__init__()
        self.precond = Precond(sigma_data=sigma_data, net_conditioning=net_conditioning)
        self.map_noise = PositionalEmbedding(num_channels=input_dim)
        self.floatenc = floatenc
        self.floatdec = floatdec
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.scheduler = scheduler
        self.time_embed = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.SiLU(),
            nn.Linear(input_dim, input_dim)
        )
        self.edm_params = edm_params
        self.projector = Sequential(
            nn.LayerNorm(input_dim, eps=1e-6),
            Linear(input_dim, 1024),
            nn.SiLU(),
            Dropout(dropout),
            Linear(1024, output_dim),
            nn.LayerNorm(output_dim, eps=1e-6))
        self.out_projector = Sequential(
            nn.LayerNorm(output_dim, eps=1e-6),
            Linear(output_dim, 1024),
            nn.SiLU(),
            Dropout(dropout),
            Linear(1024, input_dim))

        self.register_buffer("tgt_embed_rms", torch.tensor(tgt_rms, dtype=torch.float32))

        if self.scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif self.scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(
                f"The noise schedule--{self.scheduler}-- is not implemented for contiuous data at CTIME ")

    def forward(self, x_num, t, sigma_num, sampling_stage=False):
        # Continuous forward diff
        x_num_t = x_num
        if not sampling_stage:
            # add noise to x_num when training
            if x_num.shape[1] > 0:
                noise = torch.randn_like(x_num)
                x_num_t = x_num + noise * sigma_num

        c_skip, c_out, x_in, c_noise = self.precond(x_num_t, t, sigma_num)
        x_in = x_in.unsqueeze(-1)
        num_features = self.floatenc(x_in)

        emb = self.map_noise(c_noise)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        if emb.size(0) == num_features.size(0):
            time_bias = emb[:, None, :].expand(-1, num_features.size(1), -1)
        else:
            time_bias = emb.reshape(num_features.size(0), num_features.size(1), -1)
        num_features = num_features + time_bias
        num_features = self.projector(num_features)

        cur_rms = num_features.float().pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-8)
        num_features = num_features * (self.tgt_embed_rms / cur_rms)

        return c_skip, x_num_t, c_out, num_features

    def back_projection(self, c_skip, x_num, c_out, num_features):
        num_features = self.out_projector(num_features)
        F_x = self.floatdec(num_features)
        F_x = F_x.reshape(x_num.size(0), x_num.size(1))

        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        return D_x

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
        # sigma = sigma.to(torch.float32)
        # floor = float(self.edm_params.get("weight_sigma_floor", 0.0) or 0.0)
        # if floor > 0:
        #     sigma_w = sigma.clamp(min=floor)
        # else:
        #     sigma_w = sigma
        # weight = (sigma_w ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma_w * self.edm_params['sigma_data']) ** 2

        target = y
        loss = weight * ((D_yn - target) ** 2)

        return loss