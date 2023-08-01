import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from GKD.common.Top_heavy_i3d_backbone import InceptionI3d
from GKD.common.config import config
from GKD.common.layers import Unit1D, Unit3D
from GKD.prop_pooling.boundary_pooling_op import BoundaryMaxPooling

num_classes = config['dataset']['num_classes']
freeze_bn = config['model']['freeze_bn']
freeze_bn_affine = config['model']['freeze_bn_affine']

layer_num = 6
conv_channels = 512
feat_t = 256 // 4


class I3D_BackBone(nn.Module):
    def __init__(self, final_endpoint='Mixed_5c', name='inception_i3d', in_channels=3,
                 freeze_bn=freeze_bn, freeze_bn_affine=freeze_bn_affine):
        super(I3D_BackBone, self).__init__()
        self._model = InceptionI3d(final_endpoint=final_endpoint,
                                   name=name,
                                   in_channels=in_channels)
        self._model.build()
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

    def train(self, mode=True):
        super(I3D_BackBone, self).train(mode)
        if self._freeze_bn and mode:
            # print('freeze all BatchNorm3d in I3D backbone.')
            for name, m in self._model.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    # print('freeze {}.'.format(name))
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)

    def forward(self, x):
        return self._model.extract_features(x)


class ScaleExp(nn.Module):
    def __init__(self, init_value=1.0):
        super(ScaleExp, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return torch.exp(input * self.scale)


class ProposalBranch(nn.Module):
    def __init__(self, in_channels, proposal_channels):
        super(ProposalBranch, self).__init__()
        self.cur_point_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.lr_conv = nn.Sequential(
            Unit1D(in_channels=in_channels,
                   output_channels=proposal_channels * 2,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels * 2),
            nn.ReLU(inplace=True)
        )

        self.boundary_max_pooling = BoundaryMaxPooling()

        self.roi_conv = nn.Sequential(
            Unit1D(in_channels=proposal_channels,
                   output_channels=proposal_channels,
                   kernel_shape=1,
                   activation_fn=None),
            nn.GroupNorm(32, proposal_channels),
            nn.ReLU(inplace=True)
        )
        self.proposal_conv = nn.Sequential(
            Unit1D(
                in_channels=proposal_channels * 4,
                output_channels=in_channels,
                kernel_shape=1,
                activation_fn=None
            ),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature, frame_level_feature, segments, frame_segments):
        fm_short = self.cur_point_conv(feature)
        feature = self.lr_conv(feature)
        prop_feature = self.boundary_max_pooling(feature, segments)
        prop_roi_feature = self.boundary_max_pooling(frame_level_feature, frame_segments)
        prop_roi_feature = self.roi_conv(prop_roi_feature)
        prop_feature = torch.cat([prop_roi_feature, prop_feature, fm_short], dim=1)
        prop_feature = self.proposal_conv(prop_feature)
        return prop_feature, feature


class CoarsePyramid(nn.Module):
    def __init__(self, feat_channels, frame_num=256):
        super(CoarsePyramid, self).__init__()
        out_channels = conv_channels
        self.pyramids = nn.ModuleList()
        self.loc_heads = nn.ModuleList()
        self.frame_num = frame_num
        self.layer_num = layer_num
        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[0],
                output_channels=out_channels,
                kernel_shape=[1, 6, 6],
                padding='spatial_valid',
                use_batch_norm=False,
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))

        self.pyramids.append(nn.Sequential(
            Unit3D(
                in_channels=feat_channels[1],
                output_channels=out_channels,
                kernel_shape=[1, 3, 3],
                use_batch_norm=False,
                padding='spatial_valid',
                use_bias=True,
                activation_fn=None
            ),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ))
        for i in range(2, layer_num):
            self.pyramids.append(nn.Sequential(
                Unit1D(
                    in_channels=out_channels,
                    output_channels=out_channels,
                    kernel_shape=3,
                    stride=2,
                    use_bias=True,
                    activation_fn=None
                ),
                nn.GroupNorm(32, out_channels),
                nn.ReLU(inplace=True)
            ))

        loc_towers = []
        for i in range(2):
            loc_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.loc_tower = nn.Sequential(*loc_towers)
        conf_towers = []
        for i in range(2):
            conf_towers.append(
                nn.Sequential(
                    Unit1D(
                        in_channels=out_channels,
                        output_channels=out_channels,
                        kernel_shape=3,
                        stride=1,
                        use_bias=True,
                        activation_fn=None
                    ),
                    nn.GroupNorm(32, out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        self.conf_tower = nn.Sequential(*conf_towers)

        self.loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )
        self.conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.loc_proposal_branch = ProposalBranch(out_channels, 512)
        self.conf_proposal_branch = ProposalBranch(out_channels, 512)

        self.prop_loc_head = Unit1D(
            in_channels=out_channels,
            output_channels=2,
            kernel_shape=1,
            activation_fn=None
        )
        self.prop_conf_head = Unit1D(
            in_channels=out_channels,
            output_channels=num_classes,
            kernel_shape=1,
            activation_fn=None
        )

        self.center_head = Unit1D(
            in_channels=out_channels,
            output_channels=1,
            kernel_shape=3,
            stride=1,
            use_bias=True,
            activation_fn=None
        )

        self.deconv = nn.Sequential(
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 3, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            Unit1D(out_channels, out_channels, 1, activation_fn=None),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

        self.priors = []
        t = feat_t
        for i in range(layer_num):
            self.loc_heads.append(ScaleExp())
            self.priors.append(
                torch.Tensor([[(c + 0.5) / t] for c in range(t)]).view(-1, 1)
            )
            t = t // 2

    def forward(self, feat_dict, ssl=False):
        pyramid_feats = []
        locs = []
        confs = []
        centers = []
        prop_locs = []
        prop_confs = []
        trip = []
        x2 = feat_dict['Mixed_5c']
        x1 = feat_dict['Mixed_4f']
        batch_num = x1.size(0)
        for i, conv in enumerate(self.pyramids):
            if i == 0:
                x = conv(x1)
                x = x.squeeze(-1).squeeze(-1)
            elif i == 1:
                x = conv(x2)
                x = x.squeeze(-1).squeeze(-1)
                x0 = pyramid_feats[-1]
                y = F.interpolate(x, x0.size()[2:], mode='nearest')
                pyramid_feats[-1] = x0 + y
            else:
                x = conv(x)

            pyramid_feats.append(x)
        frame_level_feat = pyramid_feats[0].unsqueeze(-1)
        frame_level_feat = F.interpolate(
            frame_level_feat, [self.frame_num, 1]).squeeze(-1)
        frame_level_feat = self.deconv(frame_level_feat)
        trip.append(frame_level_feat.clone())
        start_feat = frame_level_feat[:, :256]
        end_feat = frame_level_feat[:, 256:]
        start = start_feat.permute(0, 2, 1).contiguous()
        end = end_feat.permute(0, 2, 1).contiguous()

        for i, feat in enumerate(pyramid_feats):
            loc_feat = self.loc_tower(feat)
            conf_feat = self.conf_tower(feat)
            locs.append(
                self.loc_heads[i](self.loc_head(loc_feat))
                    .view(batch_num, 2, -1)
                    .permute(0, 2, 1).contiguous()
            )

            confs.append(
                self.conf_head(conf_feat).view(batch_num, num_classes, -1)
                    .permute(0, 2, 1).contiguous()
            )
            t = feat.size(2)
            with torch.no_grad():
                segments = locs[-1] / self.frame_num * t
                priors = self.priors[i].expand(batch_num, t, 1).to(feat.device)
                new_priors = torch.round(priors * t - 0.5)
                plen = segments[:, :, :1] + segments[:, :, 1:]
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)

                l_segment = new_priors - segments[:, :, :1]
                r_segment = new_priors + segments[:, :, 1:]
                segments = torch.cat([
                    torch.round(l_segment - out_plen),
                    torch.round(l_segment + in_plen),
                    torch.round(r_segment - in_plen),
                    torch.round(r_segment + out_plen)
                ], dim=-1)

                decoded_segments = torch.cat(
                    [priors[:, :, :1] * self.frame_num - locs[-1][:, :, :1],
                     priors[:, :, :1] * self.frame_num + locs[-1][:, :, 1:]],
                    dim=-1)
                plen = decoded_segments[:, :, 1:] - \
                    decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)

            loc_prop_feat, loc_prop_feat_ = self.loc_proposal_branch(loc_feat, frame_level_feat,
                                                                     segments, frame_segments)
            conf_prop_feat, conf_prop_feat_ = self.conf_proposal_branch(conf_feat, frame_level_feat,
                                                                        segments, frame_segments)
            if i == 0:
                trip.extend([loc_prop_feat_.clone(), conf_prop_feat_.clone()])
                ndim = loc_prop_feat_.size(1) // 2
                start_loc_prop = loc_prop_feat_[
                    :, :ndim, ].permute(0, 2, 1).contiguous()
                end_loc_prop = loc_prop_feat_[
                    :, ndim:, ].permute(0, 2, 1).contiguous()
                start_conf_prop = conf_prop_feat_[
                    :, :ndim, ].permute(0, 2, 1).contiguous()
                end_conf_prop = conf_prop_feat_[
                    :, ndim:, ].permute(0, 2, 1).contiguous()
                if ssl:
                    return trip
            prop_locs.append(self.prop_loc_head(loc_prop_feat).view(batch_num, 2, -1)
                             .permute(0, 2, 1).contiguous())
            prop_confs.append(self.prop_conf_head(conf_prop_feat).view(batch_num, num_classes, -1)
                              .permute(0, 2, 1).contiguous())
            centers.append(
                self.center_head(loc_prop_feat).view(batch_num, 1, -1)
                    .permute(0, 2, 1).contiguous()
            )

        loc = torch.cat([o.view(batch_num, -1, 2) for o in locs], 1)

        conf = torch.cat([o.view(batch_num, -1, num_classes)
                         for o in confs], 1)

        prop_loc = torch.cat([o.view(batch_num, -1, 2) for o in prop_locs], 1)
        prop_conf = torch.cat([o.view(batch_num, -1, num_classes)
                              for o in prop_confs], 1)
        center = torch.cat([o.view(batch_num, -1, 1) for o in centers], 1)
        priors = torch.cat(self.priors, 0).to(loc.device)
        # priors = torch.cat(self.priors, 0).to(loc.device).unsqueeze(0)
        return loc, conf, prop_loc, prop_conf, center, priors, start, end, \
            start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop


class BDNet_student(nn.Module):
    def __init__(self, in_channels=3, backbone_model=None, training=True,
                 frame_num=768):
        super(BDNet_student, self).__init__()

        self.coarse_pyramid_detection = CoarsePyramid([832, 1024])
        self.reset_params()
        self.boundary_max_pooling = BoundaryMaxPooling()
        self.backbone = I3D_BackBone(in_channels=in_channels)
        self._training = training
        # if self._training:
        #     if backbone_model is None:
        #         self.backbone.load_pretrained_weight()
        #     else:
        #         self.backbone.load_pretrained_weight(backbone_model)
        self.scales = [1, 4, 4]
        self.deconv1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1024,
                out_channels=512,
                kernel_size=[1, 3, 3],
            ),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),)
        self.deconv2 = nn.Sequential(
            nn.Conv3d(
                in_channels=832,
                out_channels=512,
                kernel_size=[1, 6, 6],
            ),
            nn.GroupNorm(32, 512),
            nn.ReLU(inplace=True),)
        self.Att_Head_4f = nn.Sequential(

            Unit1D(512, 832, 3, activation_fn=None),
            nn.GroupNorm(32, 832),
            nn.ReLU(inplace=True),
            nn.Linear(832, 64),
            nn.Sigmoid(),
        )
        self.Att_Head_5c = nn.Sequential(
            Unit1D(512, 1024, 3, activation_fn=None),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 32),
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
        # Initialization
        for modules in [
            self.coarse_pyramid_detection.loc_tower, self.coarse_pyramid_detection.conf_tower,
            self.coarse_pyramid_detection.loc_head, self.coarse_pyramid_detection.conf_head,
            self.coarse_pyramid_detection.loc_proposal_branch,
            self.coarse_pyramid_detection.conf_proposal_branch,
            self.coarse_pyramid_detection.prop_loc_head,
            self.coarse_pyramid_detection.prop_conf_head,
            self.coarse_pyramid_detection.center_head
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv1d):
                    torch.nn.init.normal_(layer.weight, mean=0, std=0.01)
                    torch.nn.init.constant_(layer.bias, 0)

    def forward(self, x, proposals=None, ssl=False, mode='clf'):
        if mode == 'bone':
            feat_dict = self.backbone(x)
            return feat_dict
        if mode == 'clf':
            feat_dict = x
            if ssl:
                top_feat = self.coarse_pyramid_detection(feat_dict, ssl)
                decoded_segments = proposals[0].unsqueeze(0)
                plen = decoded_segments[:, :, 1:] - \
                    decoded_segments[:, :, :1] + 1.0
                in_plen = torch.clamp(plen / 4.0, min=1.0)
                out_plen = torch.clamp(plen / 10.0, min=1.0)
                frame_segments = torch.cat([
                    torch.round(decoded_segments[:, :, :1] - out_plen),
                    torch.round(decoded_segments[:, :, :1] + in_plen),
                    torch.round(decoded_segments[:, :, 1:] - in_plen),
                    torch.round(decoded_segments[:, :, 1:] + out_plen)
                ], dim=-1)
                anchor, positive, negative = [], [], []
                for i in range(3):
                    bound_feat = self.boundary_max_pooling(
                        top_feat[i], frame_segments / self.scales[i])
                    # for triplet loss
                    ndim = bound_feat.size(1) // 2
                    anchor.append(bound_feat[:, ndim:, 0])
                    positive.append(bound_feat[:, :ndim, 1])
                    negative.append(bound_feat[:, :ndim, 2])

                return anchor, positive, negative
            else:
                loc, conf, prop_loc, prop_conf, center, priors, start, end, \
                    start_loc_prop, end_loc_prop, start_conf_prop, end_conf_prop = \
                    self.coarse_pyramid_detection(feat_dict)
                return {
                    'loc': loc,
                    'conf': conf,
                    'priors': priors,
                    'prop_loc': prop_loc,
                    'prop_conf': prop_conf,
                    'center': center,
                    'start': start,
                    'end': end,
                    'start_loc_prop': start_loc_prop,
                    'end_loc_prop': end_loc_prop,
                    'start_conf_prop': start_conf_prop,
                    'end_conf_prop': end_conf_prop
                }
        if mode == 'att':
            feature_5c = x['Mixed_5c']
            feature_4f = x['Mixed_4f']
            feature_5c = self.deconv1(feature_5c)
            feature_5c = feature_5c.squeeze(-1)
            feature_5c = F.interpolate(feature_5c, [1024, 1]).squeeze(-1)
            feature_4f = self.deconv2(feature_4f)
            feature_4f = feature_4f.squeeze(-1)
            feature_4f = F.interpolate(feature_4f, [832, 1]).squeeze(-1)

            return self.Att_Head_5c(feature_5c), self.Att_Head_4f(feature_4f)


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
                kernel_size=[1, 3, 3],
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
                kernel_size=[1, 3, 3],),
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
