import random

import torch as tch
import torch.nn as nn
import numpy as np


class SpatialTransform(nn.Module):
    def __init__(self, r=45, s=1.5):
        super(SpatialTransform, self).__init__()
        self.r = r
        self.s = s
        self.inverse_dower_transform_8 = tch.tensor([
        [0.156, -0.010, -0.172, -0.074, 0.122, 0.231, 0.239, 0.194],
        [-0.227, 0.887, 0.057, -0.019, -0.106, -0.022, 0.041, 0.048],
        [0.022, 0.102, -0.229, -0.310, -0.246, -0.063, 0.055, 0.108]
        ])
        self.dowers_transform = tch.tensor([
            [0.632, 0.235, -0.515, 0.044, 0.882, 1.213, 1.125, 0.831],
            [-0.235, 1.066, 0.157, 0.164, 0.098, 0.127, 0.127, 0.076],
            [0.059, -0.132, -0.917, -1.387, -1.277, -0.601, -0.086, 0.230]
        ]).T
    
    def forward(self, ecg):
        vcg_data = self.inverse_dower_transform_8 @ ecg

        #Rotation
        rot_matrix = self.rotation_matrix_xyz(self.r)
        rotated_vcg_data = rot_matrix @ vcg_data

        #Scaling
        mask = tch.rand(3) > 0.5
        scaler = 1 + tch.rand(3) * (self.s - 1)
        scaler = tch.where(mask, scaler, 1/scaler)
        scaling_matrix = tch.diag(scaler)
        scaled_rotated_vcg_data = scaling_matrix @ rotated_vcg_data

        
        transoformed_ecg = self.dowers_transform @ scaled_rotated_vcg_data
        return transoformed_ecg
    
    def rotation_matrix_xyz(self, r):
        rotations = tch.FloatTensor(3).uniform_(-r, r)
        rad = tch.deg2rad(rotations) 

        cx, sx = tch.cos(rad[0]), tch.sin(rad[0])
        cy, sy = tch.cos(rad[1]), tch.sin(rad[1])
        cz, sz = tch.cos(rad[2]), tch.sin(rad[2])
        
        R_x = tch.tensor([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        
        R_y = tch.tensor([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        
        R_z = tch.tensor([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        
        # Combining rotations: ZYX order
        return R_z @ R_y @ R_x  # Using @ for matrix multiplication

class ZeroMask(nn.Module):
    def __init__(self, r=0.5):
        super(ZeroMask, self).__init__()
        self.r = r
    
    def forward(self ,x):
        leads, L = x.shape
        num_samples_mask = int(self.r * L)

        X_masked = x.clone()
        
        for lead in range(leads):
            start_point = tch.randint(0, L, (1,)).item()
            ending_point = start_point + num_samples_mask
            if ending_point > L:
                ending_point = L
                mask = list(range(0, start_point + num_samples_mask - L)) + list(range(start_point, ending_point))
            else:
                mask = list(range(start_point, ending_point))
            X_masked[lead, mask] = 0
    

        return X_masked
    
class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x).float()
        k = self.base_transform(x).float()
        return q, k