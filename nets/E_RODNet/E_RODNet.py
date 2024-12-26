import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple, DropPath, trunc_normal_
from einops import rearrange
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from thop import profile
import math


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        return x * self.scale


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """
    def __init__(self, scale_value=1.0, bias_value=0.0,
        scale_learnable=True, bias_learnable=True,
        mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
            requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
            requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x)**2 + self.bias


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, kernel_size=(3, 7, 7), padding=(1, 3, 3),
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv3d(
            med_channels, med_channels, kernel_size=kernel_size, stride=1,
            padding=padding, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 4, 1, 2, 3)  # [B, T, H, W, C] -> [B, C, T, H, W]
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # [B, C, T, H, W] -> [B, T, H, W, C]
        x = self.act2(x)
        x = self.pwconv2(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MetaFormerBlock(nn.Module):
    """
    Implementation of one MetaFormer block.
    """

    def __init__(self, dim,
                 token_mixer=nn.Identity, mlp=Mlp,
                 norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 layer_scale_init_value=None, res_scale_init_value=None
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale1 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp(dim=dim, drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale2 = Scale(dim=dim, init_value=layer_scale_init_value) \
            if layer_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) \
            if res_scale_init_value else nn.Identity()

    def forward(self, x):
        x = self.res_scale1(x) + \
            self.layer_scale1(
                self.drop_path1(
                    self.token_mixer(self.norm1(x))
                )
            )
        x = self.res_scale2(x) + \
            self.layer_scale2(
                self.drop_path2(
                    self.mlp(self.norm2(x))
                )
            )
        return x


class SFF_Module(nn.Module):
    def __init__(self, in_chirps=4, out_channels=32):
        super(SFF_Module, self).__init__()
        self.in_chirps = in_chirps
        self.out_channels = out_channels
        self.v_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=0),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU()
        )
        self.s_net = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=out_channels // 2, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            nn.BatchNorm3d(int(out_channels // 2)),
            nn.GELU(),
            nn.MaxPool3d(kernel_size=(4, 1, 1))
        )
        self.merge_net = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(out_channels)),
            nn.GELU()
        )

    def forward(self, x):
        batch_size, n_channels, win_size, in_chirps, w, h = x.shape
        x_out = torch.zeros((batch_size, self.out_channels, win_size, w, h)).cuda()
        for win in range(win_size):
            x_velocity = self.v_net(x[:, :, win, :, :, :]).squeeze(2)  # (B, C/2, 128, 128)
            x_space = self.s_net(x[:, :, win, :, :, :]).squeeze(2)  # (B, C/2, 128, 128)
            x_merge = torch.cat([x_velocity, x_space], dim=1)  # (B, C, 128, 128)
            x_out[:, :, win, :, :] = self.merge_net(x_merge)
        return x_out
        

class GFF_Module(nn.Module):
    def __init__(self, dim=256, kernel_size=(4, 8, 8), expansion_ratio=4):
        super().__init__()
        self.globalpatch = nn.AvgPool3d(kernel_size=kernel_size, stride=kernel_size, padding=0)
        self.proj_layer = nn.Sequential(nn.Linear(dim*expansion_ratio, dim*expansion_ratio//2),
                                        nn.LayerNorm(normalized_shape=dim*expansion_ratio//2),
                                        nn.Linear(dim*expansion_ratio//2, dim),
                                        nn.GELU(),
                                        nn.Linear(dim, dim),
                                        )
        self.catlayer = nn.Sequential(nn.Conv3d(in_channels=dim*2, out_channels=dim, kernel_size=1, stride=1, padding=0),
                                      nn.GELU()
                                      )

    def forward(self, x):
        B, C, T, H, W = x.shape

        x1 = x / x.norm(dim=1, keepdim=True)
        x2 = self.globalpatch(x)  # B,256,1,2,2
        x2 = rearrange(x2, 'b c d h w-> b (c d h w)')
        x2 = self.proj_layer(x2)
        x20 = x2 / x2.norm(dim=1, keepdim=True)
        x20 = x20.view(B, C, 1, 1, 1).expand(B, C, T, H, W)
        x_global = self.catlayer(torch.cat([x1, x20], dim=1))

        return x_global


class RODEncode(nn.Module):

    def __init__(self, in_channels, depths=[2,2,4]):
        super(RODEncode, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64,
                      kernel_size=(9, 5, 5), stride=(2, 2, 2), padding=(4, 2, 2)),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.GELU()
        )

        self.block1 = nn.Sequential(
            *[MetaFormerBlock(dim=64, token_mixer=SepConv, mlp=Mlp, norm_layer=nn.LayerNorm,
                              drop=0.1, drop_path=0., layer_scale_init_value=None, res_scale_init_value=None)
              for i in range(depths[0])])

        self.downSample1 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        self.block2 = nn.Sequential(
            *[MetaFormerBlock(dim=128, token_mixer=SepConv, mlp=Mlp, norm_layer=nn.LayerNorm,
                              drop=0.1, drop_path=0., layer_scale_init_value=None, res_scale_init_value=None)
              for i in range(depths[1])])

        self.downSample2 = nn.Sequential(
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        self.block3 = nn.Sequential(
            *[MetaFormerBlock(dim=256, token_mixer=SepConv, mlp=Mlp, norm_layer=nn.LayerNorm,
                              drop=0.1, drop_path=0., layer_scale_init_value=None, res_scale_init_value=None)
              for i in range(depths[2])])

    def forward(self, x):
        x = self.stem(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.block1(x)
        x = x.permute(0, 4, 1, 2, 3)
        x1 = x

        x = self.downSample1(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.block2(x)
        x = x.permute(0, 4, 1, 2, 3)
        x2 = x

        x = self.downSample2(x)

        x = x.permute(0, 2, 3, 4, 1)
        x = self.block3(x)
        x = x.permute(0, 4, 1, 2, 3)

        return x, x1, x2


class RODDecode(nn.Module):

    def __init__(self, n_class, depths=[2, 2]):
        super(RODDecode, self).__init__()
        self.upSample1 = nn.Sequential(
            nn.Upsample(scale_factor=(1, 2, 2), mode="trilinear", align_corners=True),
            nn.GroupNorm(num_groups=64, num_channels=256),
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        self.linear1 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.GELU()
        )

        self.block4 = nn.Sequential(
            *[MetaFormerBlock(dim=128, token_mixer=SepConv, mlp=Mlp, norm_layer=nn.LayerNorm,
                              drop=0.1, drop_path=0., layer_scale_init_value=None, res_scale_init_value=None)
              for i in range(depths[0])])

        self.upSample2 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True),
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

        self.linear2 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.GELU()
        )

        self.block5 = nn.Sequential(
            *[MetaFormerBlock(dim=64, token_mixer=SepConv, mlp=Mlp, norm_layer=nn.LayerNorm,
                              drop=0.1, drop_path=0., layer_scale_init_value=None, res_scale_init_value=None)
              for i in range(depths[1])])

        self.upSample3 = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode="trilinear", align_corners=True),
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.Conv3d(in_channels=64, out_channels=n_class, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x, x1, x2):
        x = self.upSample1(x)

        x = self.linear1(torch.cat([x, x2], dim=1))

        x = x.permute(0, 2, 3, 4, 1)
        x = self.block4(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.upSample2(x)

        x = self.linear2(torch.cat([x, x1], dim=1))

        x = x.permute(0, 2, 3, 4, 1)
        x = self.block5(x)
        x = x.permute(0, 4, 1, 2, 3)

        x = self.upSample3(x)
        return x


class E_RODNet(nn.Module):
    def __init__(self, SFF_channels, n_class):
        super().__init__()
        self.mnet = SFF_Module(in_chirps=SFF_channels[0], out_channels=SFF_channels[1])
        self.encoder = RODEncode(in_channels=SFF_channels[1])
        self.globalfusion = GFF_Module(dim=256, kernel_size=(4, 8, 8))
        self.decoder = RODDecode(n_class=n_class)

    def forward(self, x):
        x = self.mnet(x)
        x, x1, x2 = self.encoder(x)
        x = self.globalfusion(x)
        x = self.decoder(x, x1, x2)
        return x

