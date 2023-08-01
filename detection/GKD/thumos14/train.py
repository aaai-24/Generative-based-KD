import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer

import tqdm
import numpy as np
from GKD.common.thumos_dataset import THUMOS_Dataset, get_video_info, \
    load_video_data, detection_collate, get_video_anno
from torch.utils.data import DataLoader
from GKD.thumos14.BDNet_teacher import BDNet_teacher, CVAE_teacher
from GKD.thumos14.BDNet_student import BDNet_student, CVAE_student
from GKD.thumos14.multisegment_loss import MultiSegmentLoss, loss_cvae
from GKD.common.config import config

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
num_classes = config['dataset']['num_classes']
checkpoint_path = config['training']['checkpoint_path']
pretrained_path = config['training']['pretrained_path']
pretrained_path_cvae = config['training']['pretrained_path_cvae']
focal_loss = config['training']['focal_loss']
random_seed = config['training']['random_seed']
beta = config['training']['beta']

train_state_path = os.path.join(checkpoint_path, 'training')
if not os.path.exists(train_state_path):
    os.makedirs(train_state_path)

resume = config['training']['resume']


def print_training_info():
    print('batch size: ', batch_size)
    print('learning rate: ', learning_rate)
    print('weight decay: ', weight_decay)
    print('max epoch: ', max_epoch)
    print('checkpoint path: ', checkpoint_path)
    print('loc weight: ', config['training']['lw'])
    print('cls weight: ', config['training']['cw'])
    print('iou weight: ', config['training']['piou'])
    print('ssl weight: ', config['training']['ssl'])
    print('piou:', config['training']['piou'])
    print('resume: ', resume)
    print('beta:', beta)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


gpu_id = [0]
GLOBAL_SEED = 1


def worker_init_fn(worker_id):
    set_seed(GLOBAL_SEED + worker_id)


def interval(N):
    res = []
    if N[0][0] != 0:
        res.append([0, N[0][0], 0])
    for i in range(0, len(N) - 1):
        res.append([N[i][1], N[i+1][0], 0])
    if N[-1][1] != 1:
        res.append([N[-1][1], 1, 0])
    return res


def get_rng_states():
    states = []
    states.append(random.getstate())
    states.append(np.random.get_state())
    states.append(torch.get_rng_state())
    if torch.cuda.is_available():
        states.append(torch.cuda.get_rng_state())
    return states


def set_rng_state(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state(states[3])


def save_model(epoch, model, optimizer, cvae, optimizer_cvae):
    torch.save(model.module.state_dict(),
               os.path.join(checkpoint_path, 'checkpoint-{}.ckpt'.format(epoch)))
    torch.save({'optimizer': optimizer.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, 'checkpoint-{}_optimizer.ckpt'.format(epoch)))
    torch.save(cvae.module.state_dict(), os.path.join(
        train_state_path, 'checkpoint-{}_cvae.ckpt'.format(epoch)))
    torch.save({'optimizer_cvae': optimizer_cvae.state_dict(),
                'state': get_rng_states()},
               os.path.join(train_state_path, 'checkpoint-{}_optimizer_cvae.ckpt'.format(epoch)))


def resume_training(resume, model, optimizer, cvae, optimizer_cvae):
    start_epoch = 1
    if resume > 0:
        start_epoch += resume
        model_path = os.path.join(
            checkpoint_path, 'checkpoint-{}.ckpt'.format(resume))
        model.module.load_state_dict(torch.load(model_path))
        train_path = os.path.join(
            train_state_path, 'checkpoint-{}_optimizer.ckpt'.format(resume))
        state_dict = torch.load(train_path)
        optimizer.load_state_dict(state_dict['optimizer'])
        cvae_path = os.path.join(
            train_state_path, 'checkpoint-{}_cvae.ckpt'.format(resume))
        cvae.module.load_state_dict(torch.load(cvae_path))
        train_cvae_path = os.path.join(
            train_state_path, 'checkpoint-{}_optimizer_cvae.ckpt'.format(resume))
        state_dict = torch.load(train_cvae_path)
        optimizer_cvae.load_state_dict(state_dict['optimizer_cvae'])
        set_rng_state(state_dict['state'])
    return start_epoch


def calc_bce_loss(start, end, scores):
    start = torch.tanh(start).mean(-1)
    end = torch.tanh(end).mean(-1)
    loss_start = F.binary_cross_entropy(start.view(-1),
                                        scores[:, 0].contiguous(
    ).view(-1).cuda(),
        reduction='mean')
    loss_end = F.binary_cross_entropy(end.view(-1),
                                      scores[:, 1].contiguous(
    ).view(-1).cuda(),
        reduction='mean')
    return loss_start, loss_end


def forward_one_epoch(net, clips, targets, scores=None, training=True, ssl=True, mode='clf'):
    targets = [t.cuda() for t in targets]

    if training:
        if ssl:
            output_dict = net(clips, proposals=targets, ssl=ssl, mode='clf')
        else:
            output_dict = net(clips, ssl=False, mode='clf')

    else:
        with torch.no_grad():
            output_dict = net(clips, mode='clf')

    if ssl:
        anchor, positive, negative = output_dict
        loss_ = []
        weights = [1, 0.1, 0.1]
        for i in range(3):
            loss_.append(nn.TripletMarginLoss()(
                anchor[i], positive[i], negative[i]) * weights[i])
        trip_loss = torch.stack(loss_).sum(0)
        return trip_loss
    else:
        loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct = CPD_Loss(
            [output_dict['loc'], output_dict['conf'],
             output_dict['prop_loc'], output_dict['prop_conf'],
             output_dict['center'], output_dict['priors']],
            targets)
        loss_start, loss_end = calc_bce_loss(
            output_dict['start'], output_dict['end'], scores)
        scores_ = F.interpolate(scores, scale_factor=1.0 / 4)
        loss_start_loc_prop, loss_end_loc_prop = calc_bce_loss(output_dict['start_loc_prop'],
                                                               output_dict['end_loc_prop'],
                                                               scores_)
        loss_start_conf_prop, loss_end_conf_prop = calc_bce_loss(output_dict['start_conf_prop'],
                                                                 output_dict['end_conf_prop'],
                                                                 scores_)
        loss_start = loss_start + 0.1 * \
            (loss_start_loc_prop + loss_start_conf_prop)
        loss_end = loss_end + 0.1 * (loss_end_loc_prop + loss_end_conf_prop)
        return loss_l, loss_c, loss_prop_l, loss_prop_c, loss_ct, loss_start, loss_end


def run_one_epoch(epoch, student, optimizer_student, cvae_student, optimizer_cvae_student, teacher, cvae_teacher, data_loader, epoch_step_num, training=True):
    if training:
        student.train()
        cvae_student.train()
    else:
        student.eval()
        cvae_student.eval()

    loss_loc_val = 0
    loss_conf_val = 0
    loss_prop_l_val = 0
    loss_prop_c_val = 0
    loss_ct_val = 0
    loss_start_val = 0
    loss_end_val = 0
    loss_trip_val = 0
    loss_recon_val = 0
    loss_KD_val = 0
    cost_val = 0
    loss_cvae_val = 0

    """
    KD loss
    """
    MSE_loss = nn.MSELoss()

    """
    stage 1: update cvae
    """
    for name, param in student.named_parameters():
        param.requires_grad = False
        param.grad = None
    for name, param in cvae_student.named_parameters():
        param.requires_grad = True
        param.grad = None

    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):
            loss_4f, loss_5c = 0, 0
            loss = 0
            clips = clips.cuda()
            video_feature = student(clips, mode='bone')

            attention_5c, attention_4f = student(video_feature, mode='att')

            means_5c, log_var_5c, z_5c, recon_feature_5c = cvae_student(
                'forward', video_feature['Mixed_5c'], attention_5c)
            means_4f, log_var_4f, z_4f, recon_feature_4f = cvae_student(
                'forward', video_feature['Mixed_4f'], attention_4f)

            loss_4f += loss_cvae(recon_feature_4f,
                                 video_feature['Mixed_4f'], means_4f, log_var_4f, attention_4f)
            loss_5c += loss_cvae(recon_feature_5c,
                                 video_feature['Mixed_5c'], means_5c, log_var_5c, attention_5c)
            loss = loss_4f + loss_5c

            means_5c, log_var_5c, z_5c, recon_feature_5c_teacher = cvae_teacher(
                'forward', video_feature['Mixed_5c'], attention_5c)
            means_4f, log_var_4f, z_4f, recon_feature_4f_teacher = cvae_teacher(
                'forward', video_feature['Mixed_4f'], attention_4f)

            KD_loss_cvae = MSE_loss(recon_feature_4f, recon_feature_4f_teacher) + \
                MSE_loss(recon_feature_5c, recon_feature_5c_teacher)
            loss += KD_loss_cvae * 0.01

            optimizer_cvae_student.zero_grad()
            loss.backward()
            optimizer_cvae_student.step()

            loss_cvae_val += loss.cpu().detach().numpy()
            loss_cvae_KD_val += KD_loss_cvae.cpu().detach().numpy()

            pbar.set_postfix(Total='{:.5f}'.format(float(loss.cpu().detach().numpy())), loss='{:.5f}'.format(float(loss.cpu().detach(
            ).numpy()) - float(KD_loss_cvae.cpu().detach().numpy()) * 0.01), KD_loss='{:.5f}'.format(float(KD_loss_cvae.cpu().detach().numpy()) * 0.01))

    loss_cvae_val /= (n_iter + 1)
    loss_cvae_KD_val /= (n_iter + 1)

    plog = 'CVAE total - {:.5f}, KD - {:.5f}'.format(loss_cvae_val, loss_cvae_KD_val)
    print(plog)

    """
    stage 2: update backbone + detector
    """
    for name, param in student.named_parameters():
        param.requires_grad = True
        param.grad = None
    for name, param in cvae_student.named_parameters():
        param.requires_grad = False
        param.grad = None

    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (clips, targets, scores, ssl_clips, ssl_targets, flags) in enumerate(pbar):

            clips = clips.cuda()
            targets = [t.cuda() for t in targets]
            video_feature = student(clips, mode='bone')

            feat_dict = video_feature
            attention_5c, attention_4f = student(video_feature, mode='att')

            feature_5c = video_feature['Mixed_5c']
            feature_4f = video_feature['Mixed_4f']
            attention_5c = attention_5c.unsqueeze(-1).unsqueeze(-1)
            attention_4f = attention_4f.unsqueeze(-1).unsqueeze(-1)

            feature_fg_5c = (feature_5c * attention_5c)/(attention_5c.sum())
            feature_bg_5c = (feature_5c * (1 - attention_5c)
                             ) / ((1 - attention_5c).sum())
            feature_fg_5c = feature_fg_5c
            feature_bg_5c = feature_bg_5c

            feature_fg_4f = (feature_4f * attention_4f) / (attention_4f.sum())
            feature_bg_4f = (feature_4f * (1 - attention_4f)
                             ) / ((1 - attention_4f).sum())
            feature_fg_4f = feature_fg_4f
            feature_bg_4f = feature_bg_4f

            feat_dict['Mixed_5c'] = feature_fg_5c
            feat_dict['Mixed_4f'] = feature_fg_4f
            loss_l_fg, loss_c_fg, loss_prop_l_fg, loss_prop_c_fg, loss_ct_fg, loss_start_fg, loss_end_fg = forward_one_epoch(
                student, feat_dict, targets, scores, training=training, ssl=False)

            result = []
            for t in targets:
                new_targets = t.cpu().numpy()
                new_targets = interval(new_targets)
                new_targets = torch.tensor(new_targets).cuda()
                result.append(new_targets)

            feat_dict['Mixed_5c'] = feature_bg_5c
            feat_dict['Mixed_4f'] = feature_bg_4f
            loss_l_bg, loss_c_bg, loss_prop_l_bg, loss_prop_c_bg, loss_ct_bg, loss_start_bg, loss_end_bg = forward_one_epoch(
                student, feat_dict, result, scores, training=training, ssl=False)

            loss_l_fg = loss_l_fg * config['training']['lw']
            loss_c_fg = loss_c_fg * config['training']['cw']
            loss_prop_l_fg = loss_prop_l_fg * config['training']['lw']
            loss_prop_c_fg = loss_prop_c_fg * config['training']['cw']
            loss_ct_fg = loss_ct_fg * config['training']['cw']
            cost_fg = loss_l_fg + loss_c_fg + loss_prop_l_fg + \
                loss_prop_c_fg + loss_ct_fg + loss_start_fg + loss_end_fg

            loss_l_bg = loss_l_bg * config['training']['lw']
            loss_c_bg = loss_c_bg * config['training']['cw']
            loss_prop_l_bg = loss_prop_l_bg * config['training']['lw']
            loss_prop_c_bg = loss_prop_c_bg * config['training']['cw']
            loss_ct_bg = loss_ct_bg * config['training']['cw']
            cost_bg = loss_l_bg + loss_c_bg

            cost = cost_bg * 35 + cost_fg

            KD_loss_ssl = 0
            if flags[0]:
                ssl_clips = ssl_clips.cuda()
                ssl_video_feature = student(ssl_clips, mode='bone')
                ssl_attention_5c, ssl_attention_4f = student(
                    ssl_video_feature, mode='att')

                ssl_feature_5c = ssl_video_feature['Mixed_5c']

                ssl_feature_4f = ssl_video_feature['Mixed_4f']
                ssl_attention_5c = ssl_attention_5c.unsqueeze(-1).unsqueeze(-1)
                ssl_attention_4f = ssl_attention_4f.unsqueeze(-1).unsqueeze(-1)

                ssl_feature_fg_5c = (
                    ssl_feature_5c * ssl_attention_5c)/(ssl_attention_5c.sum())

                ssl_feature_fg_5c = ssl_feature_fg_5c

                ssl_feature_fg_4f = (
                    ssl_feature_4f * ssl_attention_4f) / (ssl_attention_4f.sum())

                ssl_feature_fg_4f = ssl_feature_fg_4f

                feat_dict['Mixed_5c'] = ssl_feature_fg_5c
                feat_dict['Mixed_4f'] = ssl_feature_fg_4f
                loss_trip = forward_one_epoch(student, feat_dict, ssl_targets, training=training,
                                              ssl=True)
                loss_trip *= config['training']['ssl']
                cost = cost + loss_trip
                loss_trip_val += loss_trip.cpu().detach().numpy()
                """
                teacher output(ssl)
                """
                ssl_video_feature_teacher = teacher(ssl_clips, mode='bone')
                ssl_attention_5c_teacher, ssl_attention_4f_teacher = teacher(
                    ssl_video_feature_teacher, mode='att')
                ssl_attention_5c_teacher = ssl_attention_5c_teacher.unsqueeze(
                    -1).unsqueeze(-1)
                ssl_attention_4f_teacher = ssl_attention_4f_teacher.unsqueeze(
                    -1).unsqueeze(-1)

                KD_loss_ssl += MSE_loss(ssl_attention_4f_teacher, ssl_attention_4f) + \
                    MSE_loss(ssl_attention_5c_teacher, ssl_attention_5c)
                cost += KD_loss_ssl * beta

            l_recon_4f, l_recon_5c = 0, 0
            recon_feature_4f = cvae_student('inference', att=attention_4f)
            l_recon_4f += (recon_feature_4f -
                           video_feature['Mixed_4f']).pow(2).mean()
            recon_feature_5c = cvae_student('inference', att=attention_5c)
            l_recon_5c += (recon_feature_5c -
                           video_feature['Mixed_5c']).pow(2).mean()

            l_recon = l_recon_4f + l_recon_5c

            cost += min(epoch, 20)/20 * 0.5 * l_recon

            """
            teacher output
            """
            video_feature_teacher = teacher(clips, mode='bone')
            attention_5c_teacher, attention_4f_teacher = teacher(
                video_feature_teacher, mode='att')
            attention_5c_teacher = attention_5c_teacher.unsqueeze(
                -1).unsqueeze(-1)
            attention_4f_teacher = attention_4f_teacher.unsqueeze(
                -1).unsqueeze(-1)

            KD_loss = MSE_loss(attention_4f, attention_4f_teacher) + \
                MSE_loss(attention_5c, attention_5c_teacher)
            cost += KD_loss * beta

            if training:
                optimizer_student.zero_grad()
                cost.backward()
                optimizer_student.step()

            loss_loc_val += loss_l_fg.cpu().detach().numpy()
            loss_conf_val += loss_c_fg.cpu().detach().numpy()
            loss_prop_l_val += loss_prop_l_fg.cpu().detach().numpy()
            loss_prop_c_val += loss_prop_c_fg.cpu().detach().numpy()
            loss_ct_val += loss_ct_fg.cpu().detach().numpy()
            loss_start_val += loss_start_fg.cpu().detach().numpy()
            loss_end_val += loss_end_fg.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            loss_recon_val += l_recon.cpu().detach().numpy()
            loss_KD_val += KD_loss.cpu().detach().numpy()
            if isinstance(KD_loss_ssl, int):
                loss_KD_val += KD_loss_ssl
            else:
                loss_KD_val += KD_loss_ssl.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy(
            ))), KD_loss='{:.5f}'.format(float((KD_loss+KD_loss_ssl).cpu().detach().numpy())))

    loss_loc_val /= (n_iter + 1)
    loss_conf_val /= (n_iter + 1)
    loss_prop_l_val /= (n_iter + 1)
    loss_prop_c_val /= (n_iter + 1)
    loss_ct_val /= (n_iter + 1)
    loss_start_val /= (n_iter + 1)
    loss_end_val /= (n_iter + 1)
    loss_trip_val /= (n_iter + 1)
    loss_recon_val /= (n_iter + 1)
    cost_val /= (n_iter + 1)
    loss_KD_val /= (n_iter + 1)

    if training:
        prefix = 'Train'
        save_model(epoch, student, optimizer_student,
                   cvae_student, optimizer_cvae_student)
    else:
        prefix = 'Val'

    plog = 'Epoch-{} {} Loss: Total - {:.5f}, loc - {:.5f}, conf - {:.5f}, prop_loc - {:.5f}, ' \
           'prop_conf - {:.5f}, IoU - {:.5f}, start - {:.5f}, end - {:.5f}'.format(
               i, prefix, cost_val, loss_loc_val, loss_conf_val, loss_prop_l_val, loss_prop_c_val,
               loss_ct_val, loss_start_val, loss_end_val
           )
    plog = plog + ', Triplet - {:.5f}'.format(loss_trip_val)
    plog = plog + ', recon - {:.5f}'.format(loss_recon_val)
    plog = plog + ', KD - {:.5f}'.format(loss_KD_val)
    print(plog)


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    """
    Setup teacher model and load pretrained weight
    """
    net_teacher = BDNet_teacher(
        in_channels=config['model']['in_channels'], training=False)
    net_teacher.load_state_dict(torch.load(pretrained_path))
    for param in net_teacher.parameters():
        param.requires_grad = False
    net_teacher.eval().cuda()
    cvae_teacher = CVAE_teacher()
    cvae_teacher.load_state_dict(torch.load(pretrained_path_cvae))
    for param in cvae_teacher.parameters():
        param.requires_grad = False
    cvae_teacher.eval().cuda()
    """
    setup student model
    """
    net_student = BDNet_student(in_channels=config['model']['in_channels'],
                                backbone_model=config['model']['backbone_model'])
    net_student = nn.DataParallel(net_student, device_ids=gpu_id).cuda()
    cvae_student = CVAE_student()
    cvae_student = nn.DataParallel(cvae_student, device_ids=gpu_id).cuda()
    """
    Setup optimizer
    """
    optimizer_student = torch.optim.Adam(filter(lambda p: p.requires_grad, net_student.parameters()),
                                         lr=learning_rate,
                                         betas=(0.9, 0.999),
                                         weight_decay=weight_decay)
    optimizer_cvae_student = torch.optim.Adam(cvae_student.parameters(),
                                              lr=0.0001,
                                              betas=(0.8, 0.999))
    """
    Setup loss
    """
    piou = config['training']['piou']
    CPD_Loss = MultiSegmentLoss(
        num_classes, piou, 1.0, use_focal_loss=focal_loss)
    """
    Setup dataloader
    """
    train_video_infos = get_video_info(
        config['dataset']['training']['video_info_path'])
    train_video_annos = get_video_anno(train_video_infos,
                                       config['dataset']['training']['video_anno_path'])
    train_data_dict = load_video_data(train_video_infos,
                                      config['dataset']['training']['video_data_path'])
    train_dataset = THUMOS_Dataset(train_data_dict,
                                   train_video_infos,
                                   train_video_annos)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=4, worker_init_fn=worker_init_fn,
                                   collate_fn=detection_collate, pin_memory=True, drop_last=True)
    epoch_step_num = len(train_dataset) // batch_size
    """
    Start training
    """
    start_epoch = resume_training(
        resume, net_student, optimizer_student, cvae_student, optimizer_cvae_student)

    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net_student, optimizer_student, cvae_student, optimizer_cvae_student, net_teacher,
                      cvae_teacher, train_data_loader, len(train_dataset) // batch_size)
