import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from common.Top_heavy_i3d_backbone import InceptionI3d
from common.config import config
from common.layers import Unit1D

freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']

class I3D_BackBone(nn.Module):
    def __init__(self, final_endpoint='Logits', name='inception_i3d', in_channels=3,
                 freeze_bn=freeze_bn, freeze_bn_affine=freeze_bn_affine):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint=final_endpoint,
                                   name=name,
                                   in_channels=in_channels)
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine
        #101 for UCF101 and 51 for HMBD51
        self._model.replace_logits(101)

    def train(self, mode=True):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)

    def predict(self, x):
        return self._model.forward(x)


class BDNet_student(nn.Module):
    def __init__(self, in_channels=3, backbone_model=None, training=True,
                 frame_num=768):
        super(BDNet_student, self).__init__()

        self.reset_params()
        self.backbone = I3D_BackBone(in_channels=in_channels)
        self._training = training
        self.scales = [1, 4, 4]
        self.deconv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1024,
                out_channels=512,
                kernel_size=[1, 7, 7],
            ),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),)
        self.Att_Head_5c = nn.Sequential(
            Unit1D(512, 1024, 3, activation_fn=None),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8),
            nn.Sigmoid(),
        )

    @staticmethod
    def weight_init(m):
        def glorot_uniform_(tensor):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
            scale = 1.0
            scale /= max(1., (fan_in + fan_out) / 2.)
            limit = np.sqrt(3.0 * scale)
            return nn.init._no_grad_uniform_(tensor, -limit, limit)

        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) \
                or isinstance(m, nn.ConvTranspose3d):
            glorot_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x, mode='clf'):
        if mode == 'bone':
            feat_dict = self.backbone(x)
            return feat_dict
        if mode == 'clf':
            prediction = self.backbone.predict(x)
            return prediction
        if mode == 'att':
            feature_5c = x['Mixed_5c']
            feature_5c = self.deconv1(feature_5c)
            feature_5c = feature_5c.squeeze(-1)
            feature_5c = F.interpolate(feature_5c, [1024, 1]).squeeze(-1)
            return self.Att_Head_5c(feature_5c)


class CEncoder_832(nn.Module):
    def __init__(self):
        super(CEncoder_832, self).__init__()
        self.con2 = nn.Sequential(
            nn.Conv3d(
                in_channels=832,
                out_channels=832,
                kernel_size=[1, 6, 6],
            ),
            nn.GroupNorm(32, 832),
            nn.ReLU(inplace=True),)

        self.fc1 = nn.Linear(832 + 832, 832)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(832, 832)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(832, 2 * 832)  # mean + log_var

    def forward(self, x, att):
        x = self.con2(x)
        x = x.squeeze(-1).squeeze(-1)
        x = torch.cat([x, att], dim=1)  # feature dim
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x[:, :, :832], x[:, :, 832:]


class CDecoder_832(nn.Module):
    def __init__(self):
        super(CDecoder_832, self).__init__()

        self.fc1 = nn.Linear(832 + 832, 832)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(832, 832)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(832, 832)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=832,
                out_channels=832,
                kernel_size=[1, 6, 6],),
            nn.GroupNorm(32, 832),
            nn.ReLU(inplace=True),)

    def forward(self, z, att):
        att = att.permute(0, 2, 1)
        x = torch.cat([z, att], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.deconv(x)

        return x


class CEncoder(nn.Module):
    def __init__(self):
        super(CEncoder, self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=[1, 7, 7],
            ),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),)

        self.fc1 = nn.Linear(2048, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 2 * 1024)  # mean + log_var

    def forward(self, x, att):

        x = self.con1(x)
        x = x.squeeze(-1).squeeze(-1)
        x = torch.cat([x, att], dim=1)  # feature dim
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        return x[:, :, :1024], x[:, :, 1024:]


class CDecoder(nn.Module):
    def __init__(self):
        super(CDecoder, self).__init__()

        self.fc1 = nn.Linear(1024 + 1024, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 1024)
        self.deconv = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels=1024,
                out_channels=1024,
                kernel_size=[1, 7, 7],),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),)

    def forward(self, z, att):
        att = att.permute(0, 2, 1)
        x = torch.cat([z, att], dim=-1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.deconv(x)

        return x


class CVAE_student(nn.Module):
    def __init__(self):
        super(CVAE_student, self).__init__()

        self.cencoder = CEncoder()
        self.cdecoder = CDecoder()
        self.cencoder_832 = CEncoder_832()
        self.cdecoder_832 = CDecoder_832()

    def forward(self, mode, x=None, att=None):
        if mode == 'forward':
            if x.size(1) == 1024:
                means, log_var = self.cencoder(x, att)

                std = torch.exp(0.5 * log_var)
                eps = torch.randn(means.shape, device='cuda')
                z = means + eps * std

                recon_x = self.cdecoder(z, att)
            elif x.size(1) == 832:
                means, log_var = self.cencoder_832(x, att)

                std = torch.exp(0.5 * log_var)
                eps = torch.randn(means.shape, device='cuda')
                z = means + eps * std

                recon_x = self.cdecoder_832(z, att)

            return means, log_var, z, recon_x

        #######################################################
        elif mode == 'inference':
            att = att.squeeze(-1).squeeze(-1)
            att = att.permute(0, 2, 1)
            if att.size(-1) == 1024:
                z = torch.randn(*att.shape, device='cuda') + att
                att = att.permute(0, 2, 1)
                recon_x = self.cdecoder(z, att)
            elif att.size(-1) == 832:
                z = torch.randn(*att.shape, device='cuda') + att
                att = att.permute(0, 2, 1)
                recon_x = self.cdecoder_832(z, att)

            return recon_x
