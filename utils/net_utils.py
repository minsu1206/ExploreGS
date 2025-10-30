import torch
import numpy as np
import torch.nn as nn

class FeatSplatDecoder(nn.Module):
    def __init__(self, bias=False):
        super(FeatSplatDecoder, self).__init__()
        
        self.mlp1 = nn.Conv2d(12, 6, kernel_size=1, bias=bias) # 6 dim per gaussian + 6 dim for rays 

        self.mlp2 = nn.Conv2d(6, 3, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, rays):
        # albedo, spec, timefeature = x.chunk(3,dim=1)
        albedo = x[:, :3]
        spec = x[:, 3:]
        # print(f"[DEBUG] : net_utils.py : FeatSplatDecoder : forward : spec.shape = {spec.shape}, specular.shape = {albedo.shape} , rays.shape = {rays.shape}")
        
        specular = torch.cat([spec, rays], dim=1) # 6 + 6
        
        specular = self.mlp1(specular)
        specular = self.relu(specular)
        specular = self.mlp2(specular)

        result = albedo + specular
        result = self.sigmoid(result) 
        return result
    
def build_decoder(model):
    if model == "featsplat":
        return FeatSplatDecoder()
    elif model == "hier":
        raise NotImplementedError("Hier decoder not implemented")
    else:
        raise ValueError(f"Unknown model: {model}")