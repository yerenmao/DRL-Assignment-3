import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.config import NOISYNET_SIGMA_INIT

class NoisyNet(nn.Module):
    def __init__(self, input_dim, output_dim, sigma_init=NOISYNET_SIGMA_INIT):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_mean = nn.Parameter(torch.empty(output_dim, input_dim))
        self.weight_std = nn.Parameter(torch.empty(output_dim, input_dim))
        self.register_buffer('weight_noise', torch.empty(output_dim, input_dim))
        self.bias_mean = nn.Parameter(torch.empty(output_dim))
        self.bias_std = nn.Parameter(torch.empty(output_dim))
        self.register_buffer('bias_noise', torch.empty(output_dim))
        
        self.sigma_init = sigma_init
        self.reset_params()
        self.reset_noises()

    def reset_params(self):
        bound = 1 / np.sqrt(self.input_dim)
        nn.init.uniform_(self.weight_mean, -bound, bound)
        nn.init.constant_(self.weight_std, self.sigma_init / np.sqrt(self.input_dim))
        nn.init.uniform_(self.bias_mean, -bound, bound)
        nn.init.constant_(self.bias_std, self.sigma_init / np.sqrt(self.output_dim))

    def reset_noises(self):
        input_noise = self.__generate_noise_tensor(self.input_dim)
        output_noise = self.__generate_noise_tensor(self.output_dim)
        self.weight_noise.copy_(output_noise.ger(input_noise))
        self.bias_noise.copy_(output_noise)
    
    @staticmethod
    def __generate_noise_tensor(size):
        return torch.randn(size).sign() * torch.abs(torch.randn(size)).sqrt()
    
    def forward(self, input_tensor):
        if self.training:
            weight = self.weight_mean + self.weight_std * self.weight_noise
            bias = self.bias_mean + self.bias_std * self.bias_noise
            return F.linear(input_tensor, weight, bias)
        return F.linear(input_tensor, self.weight_mean, self.bias_mean)
    

class DDQN(nn.Module):
    def __init__(self, n_channels, n_actions):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten()
        )
        self.val_layer = nn.Sequential(
            NoisyNet(64 * 7 * 7, 512), nn.ReLU(),
            NoisyNet(512, 1)
        )
        self.adv_layer = nn.Sequential(
            NoisyNet(64 * 7 * 7, 512), nn.ReLU(),
            NoisyNet(512, n_actions)
        )

    def forward(self, input_tensor):
        x = self.conv_layers(input_tensor / 255.0)
        val = self.val_layer(x)
        adv = self.adv_layer(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def reset_noise(self):
        for module in self.val_noisy:
            if isinstance(module, NoisyNet):
                module.generate_noise()
        for module in self.adv_noisy:
            if isinstance(module, NoisyNet):
                module.generate_noise()