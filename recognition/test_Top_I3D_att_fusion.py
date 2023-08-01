import os
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import numpy as np
from torch.utils.data import DataLoader
from BDNet_student import BDNet_student
from common.config import config
import video_transforms
import datasets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

new_length = 64
batch_size = 1
rgb_checkpoint_path = config['testing'].get('checkpoint_path')
flow_checkpoint_path = './models/ucf101/ucf101-flow.ckpt'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def forward_one_epoch(net, feature_fg_5c, target, training=False, mode='clf'):
    if training:
        output = net(feature_fg_5c, mode='clf')
    else:
        with torch.no_grad():
            output = net(feature_fg_5c, mode='clf')
    return output

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate_flow(val_loader, net):
    output_list = []
    targets = []

    net.eval().cuda()

    with tqdm.tqdm(val_loader, total=len(val_loader), ncols=0) as pbar:
        for n_iter, (input, target) in enumerate(pbar):
            input = input.float().cuda()
            targets.append(target)
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            video_feature = net(input_var, mode='bone')
            output= forward_one_epoch(net, video_feature['Mixed_5c'], target_var, training=False)
            output_list.append(output.data)
    return output_list, targets

def validate_rgb(val_loader, net):
    output_list = []
    targets = []

    # switch to evaluate mode
    net.eval().cuda()

    with tqdm.tqdm(val_loader, total=len(val_loader), ncols=0) as pbar:
        for n_iter, (input, target) in enumerate(pbar):
            input = input.float().cuda()
            target = target.cuda()
            targets.append(target)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            video_feature = net(input_var, mode='bone')
            feature_5c = video_feature['Mixed_5c']
            attention_5c = net(video_feature, mode='att')
            attention_5c = attention_5c.unsqueeze(-1).unsqueeze(-1)
            feature_fg_5c = feature_5c * attention_5c
            feature_fg_5c = feature_fg_5c / feature_fg_5c.sum() * feature_5c.sum()
            output= forward_one_epoch(net, feature_fg_5c, target_var, training=False)
            output_list.append(output.data)

    return output_list, targets

if __name__ == '__main__':
    rgb_net = BDNet_student(in_channels=3, training=False)
    flow_net = BDNet_student(in_channels=2, training=False)

    rgb_net.load_state_dict(torch.load(rgb_checkpoint_path))
    # flow_net.load_state_dict(torch.load(flow_checkpoint_path))

    # load pretrained param
    dict_trained = torch.load(flow_checkpoint_path)
    dict_new = flow_net.state_dict().copy()

    new_list = list(flow_net.state_dict().keys())
    trained_list = list(dict_trained.keys())

    for i in range(0,344):
        dict_new[new_list[i]] = dict_trained[trained_list[i]]
    flow_net.load_state_dict(dict_new)

    print('rgb: ' + rgb_checkpoint_path)
    print('flow: ' + flow_checkpoint_path)

    # rgb data
    is_color = True
    scale_ratios = [1.0, 0.875, 0.75, 0.66]
    clip_mean = [0.485, 0.456, 0.406] * new_length
    clip_std = [0.229, 0.224, 0.225] * new_length
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
    rgb_transform = video_transforms.Compose([
            video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])

    rgb_setting_file = "val_%s_split1.txt" % ('rgb')
    rgb_split_file = os.path.join("./datasets/settings/ucf101", rgb_setting_file)
    if not os.path.exists(rgb_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % ("./datasets/settings/ucf101"))

    rgb_dataset = datasets.__dict__['ucf101'](root="/Datasets/ucf101_frames",
                                                  source=rgb_split_file,
                                                  phase="val",
                                                  modality='rgb',
                                                  is_color=is_color,
                                                  new_length=new_length,
                                                  video_transform=rgb_transform)
    rgb_loader = torch.utils.data.DataLoader(
        rgb_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)


    # flow data
    is_color = False
    scale_ratios = [1.0, 0.875, 0.75]
    clip_mean = [0.5, 0.5] * new_length
    clip_std = [0.226, 0.226] * new_length
    normalize = video_transforms.Normalize(mean=clip_mean, std=clip_std)
    flow_transform = video_transforms.Compose([
            video_transforms.CenterCrop((224)),
            video_transforms.ToTensor(),
            normalize,
        ])
    
    flow_setting_file = "val_%s_split1.txt" % ('flow')
    flow_split_file = os.path.join("./datasets/settings/ucf101", flow_setting_file)
    if not os.path.exists(flow_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % ("./datasets/settings/ucf101"))

    flow_dataset = datasets.__dict__['ucf101'](root="/Datasets/ucf101_frames",
                                                  source=flow_split_file,
                                                  phase="val",
                                                  modality='flow',
                                                  is_color=is_color,
                                                  new_length=new_length,
                                                  video_transform=flow_transform)
    flow_loader = torch.utils.data.DataLoader(
        flow_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True)
    
    rgb_output, rgb_targets = validate_rgb(rgb_loader, rgb_net)
    flow_output, flow_targets = validate_flow(flow_loader, flow_net)
    top1 = AverageMeter()
    top5 = AverageMeter()
    for i in range(len(rgb_output)):
        fusion = rgb_output[i] + flow_output[i]
        fusion = F.softmax(fusion, dim=1)
        fusion = torch.sum(fusion, dim=2)
        fusion = F.softmax(fusion, dim=1)
        rgb_targets[i] = rgb_targets[i].cuda()
        prec1, prec5 = accuracy(fusion, rgb_targets[i], topk=(1, 5))
        top1.update(prec1.item())
        top5.update(prec5.item())
    print('* Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
