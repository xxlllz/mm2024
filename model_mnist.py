import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from collections import OrderedDict
import numpy as np
import math
from models.STB import *
from models.warp import Warp
from models.fourier_modules import *
class PhyCell_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.lstm_hidden_dim = input_dim
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.height = input_shape[0]
        self.weight = input_shape[1]
        self.hidden_dim = input_dim * 2

        self.F = nn.Sequential()
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))
        self.F.add_module('bn1',nn.GroupNorm( 7 ,F_hidden_dim))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(3,3), stride=(1,1), padding=(1,1)))

        self.Swin = SwinTransformerBlocks(dim=input_dim, input_resolution=(self.height, self.weight), depth=8,
                                          num_heads=8, window_size=4, mlp_ratio=4.,
                                          qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                                          drop_path=0.1, norm_layer=nn.LayerNorm)
        self.conv_gates = Mlp(self.input_dim * 2, self.input_dim * 4, self.input_dim * 4)
                
        self.conv_can = Mlp(self.input_dim * 2, self.input_dim * 4, self.input_dim)

        self.gate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(1,1), bias=self.bias)

        self.out = Mlp(self.input_dim, self.input_dim * 4, self.input_dim)
        
        self.blocks = nn.ModuleList([FourierNetBlock(
            dim=self.input_dim,
            h=self.height,
            w=self.weight)
            for i in range(8)
        ])

        self.w2 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim,  
                      out_channels=self.input_dim,  
                      kernel_size=(1,1)),
            nn.GroupNorm(16, self.input_dim),
            nn.Conv2d(in_channels=self.input_dim, 
                      out_channels=self.input_dim,  
                      kernel_size=(1,1)))
        self.w1 = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim,  
                      out_channels=self.input_dim,  
                      kernel_size=(1,1)),
            nn.GroupNorm(16, self.input_dim),
            nn.Conv2d(in_channels=self.input_dim, 
                      out_channels=self.input_dim,  
                      kernel_size=(1,1)))
        
        self.red = Mlp(self.input_dim * 2, self.input_dim * 4, self.input_dim)

    def forward(self, x, hidden, lstm_hidden): # x [batch_size, hidden_dim, height, width]
        ft = self.Swin(x, hidden)
        B, N, C = ft.shape
        H = self.height 
        W = self.weight 
        # ft = ft.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)
        # hidden = hidden.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
        h_cur, c_cur = lstm_hidden
        combined = torch.cat([ft, h_cur], dim=-1)
        combined_conv = self.conv_gates(combined)        
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.lstm_hidden_dim, dim=-1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        out = self.out(h_next) #计算k

        hidden_tilde = hidden + (ft - hidden) * torch.sigmoid(out)

        x = torch.cat((x, hidden), dim=-1)
        x = self.red(x)
        for blk in self.blocks:
            x = blk(x)
        f_out = x

        # hidden_tilde = f_gate * self.wh(hidden_tilde) + (1-f_gate) * self.wf(f_out + f_init) + hidden

        hidden_tilde = hidden_tilde + f_out

        hidden_tilde = hidden_tilde.reshape(B, H, W, C).permute(0, 3, 1, 2)

        hidden_1 = self.F(hidden_tilde)
        hidden_2 = self.F(hidden_1 + hidden_tilde)
        h_combine = torch.cat([hidden_1, hidden_2], dim=1)
        g = torch.sigmoid(self.gate(h_combine))
        next_hidden = hidden_tilde + g * self.w1(hidden_1) + (1-g) * self.w2(hidden_2) 
        
        # next_hidden = next_hidden + f_out

        return next_hidden.reshape(B, C, -1).permute(0, 2, 1), h_next, c_next

class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.tilde_H = []
        self.h_weight = []
        self.device = device         
        cell_list = []
        for i in range(0, self.n_layers):
            cell_list.append(PhyCell_Cell(input_shape=input_shape,
                                          input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j, cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                x = input_
            else:
                x = self.H[j-1]
        
            self.H[j], self.h[j], self.c[j] = cell(x, self.H[j], (self.h[j],self.c[j]))
        return self.H , self.H
    
    def initHidden(self,batch_size):
        self.H = [] 
        self.h = []
        self.c = []
        for i in range(self.n_layers):
            self.H.append(torch.zeros(batch_size, self.input_shape[0]*self.input_shape[1], self.input_dim).to(self.device) )
            self.h.append(torch.zeros(batch_size, self.input_shape[0]*self.input_shape[1], self.input_dim).to(self.device))
            self.c.append(torch.zeros(batch_size, self.input_shape[0]*self.input_shape[1], self.input_dim).to(self.device))


    def setHidden(self, H):
        self.H = H

class EncoderRNN(torch.nn.Module):
    def __init__(self, phycell, device, nc=1, img_size=(128, 128), patch_size=(4, 4)):
        super(EncoderRNN, self).__init__()
        self.phycell = phycell.to(device)
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=nc, embed_dim=64).to(device)
        patches_resolution = self.patch_embed.patches_resolution
        self.PatchInflated = PatchInflated(in_chans=nc, embed_dim=64, input_resolution=patches_resolution).to(device)
        
    def forward(self, input, first_timestep=False):
        input, skips = self.patch_embed(input) # 64, 64, 64\
        hidden1, output1 = self.phycell(input, first_timestep)

        output_image = torch.sigmoid(self.PatchInflated(output1[-1], skips))
        return output_image

