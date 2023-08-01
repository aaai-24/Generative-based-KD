import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import optimizer

import tqdm
import numpy as np
from torch.utils.data import DataLoader
from BDNet_student import BDNet_student, CVAE_student
from BDNet_teacher import BDNet_teacher, CVAE_teacher
from multisegment_loss import loss_cvae
from common.config import config
import video_transforms
import datasets

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


batch_size = config['training']['batch_size']
learning_rate = config['training']['learning_rate']
weight_decay = config['training']['weight_decay']
max_epoch = config['training']['max_epoch']
checkpoint_path = config['training']['checkpoint_path']
pretrained_teacher = config['training']['pretrained_teacher']
pretrained_cvae = config['training']['pretrained_cvae']
random_seed = config['training']['random_seed']
modality = config['training']['modality']
new_length = config['training']['new_length']
iter_size = config['training']['iter_size']
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
    print('resume: ', resume)
    print('new length: ', new_length)
    print('mode: ', modality)
    print('iter: ', iter_size)
    print('beta: ', beta)


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
        set_rng_state(state_dict['state'])
        cvae_path = os.path.join(
            train_state_path, 'checkpoint-{}_cvae.ckpt'.format(resume))
        cvae.module.load_state_dict(torch.load(cvae_path))
        train_cvae_path = os.path.join(
            train_state_path, 'checkpoint-{}_optimizer_cvae.ckpt'.format(resume))
        state_dict = torch.load(train_cvae_path)
        optimizer_cvae.load_state_dict(state_dict['optimizer_cvae'])
        set_rng_state(state_dict['state'])
    return start_epoch


def forward_one_epoch(net, feature_5c, target, training=True, mode='clf'):
    if training:
        output = net(feature_5c, mode='clf')
    else:
        with torch.no_grad():
            output = net(feature_5c, mode='clf')
    criterion = nn.CrossEntropyLoss().cuda()
    output = F.softmax(output, dim=1)
    output = torch.sum(output, dim=2)
    loss = criterion(output, target)
    return loss


def run_one_epoch(epoch, net, optimizer, cvae, optimizer_cvae, teacher, cvae_teacher, data_loader, epoch_step_num, training=True):
    if training:
        net.train()
        cvae.train()
    else:
        net.eval()
        cvae.eval()

    loss_fg_val = 0
    cost_val = 0
    loss_recon_val = 0
    loss_cvae_val = 0
    loss_cvae_KD_val = 0
    loss_KD_val = 0

    MSE_loss = nn.MSELoss()

    """
    stage 1: update cvae
    """
    for name, param in net.named_parameters():
        param.requires_grad = False
        param.grad = None
    for name, param in cvae.named_parameters():
        param.requires_grad = True
        param.grad = None

    optimizer_cvae.zero_grad()

    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (input, target) in enumerate(pbar):

            loss_5c = 0
            loss = 0
            input = input.float().cuda()
            input_var = torch.autograd.Variable(input)

            video_feature = net(input_var, mode='bone')
            attention_5c = net(video_feature, mode='att')

            means_5c, log_var_5c, z_5c, recon_feature_5c = cvae('forward', video_feature['Mixed_5c'], attention_5c)
            loss_5c += loss_cvae(recon_feature_5c, video_feature['Mixed_5c'], means_5c, log_var_5c, attention_5c)
            
            means_5c_teacher, log_var_5c_teacher, z_5c_teacher, recon_feature_5c_teacher = cvae_teacher(
                'forward', video_feature['Mixed_5c'], attention_5c)
            KD_loss_cvae = MSE_loss(recon_feature_5c, recon_feature_5c_teacher)
            loss = (loss_5c + KD_loss_cvae * 0.01) / iter_size
            loss.backward()

            if (n_iter+1) % iter_size == 0:
                optimizer_cvae.step()
                optimizer_cvae.zero_grad()

            loss_cvae_val += loss.cpu().detach().numpy()
            loss_cvae_KD_val += KD_loss_cvae.cpu().detach().numpy()

            pbar.set_postfix(loss='{:.5f}'.format(float(loss.cpu().detach().numpy()))
            , KD_loss='{:.5f}'.format(float(KD_loss_cvae.cpu().detach().numpy())))

    loss_cvae_val /= (n_iter + 1)
    loss_cvae_KD_val /= (n_iter + 1)

    plog = 'Train Loss: CVAE - {:.5f}, KD - {:.5f}'.format(loss_cvae_val, loss_cvae_KD_val)
    print(plog)

    """
    stage 2: update I3D
    """
    for name, param in net.named_parameters():
        param.requires_grad = True
        param.grad = None
    for name, param in cvae.named_parameters():
        param.requires_grad = False
        param.grad = None

    optimizer.zero_grad()

    with tqdm.tqdm(data_loader, total=epoch_step_num, ncols=0) as pbar:
        for n_iter, (input, target) in enumerate(pbar):

            input = input.float().cuda()
            input_var = torch.autograd.Variable(input)
            target = target.cuda()
            target_var = torch.autograd.Variable(target)

            video_feature = net(input_var, mode='bone')
            feature_5c = video_feature['Mixed_5c']
            attention_5c = net(video_feature, mode='att')
            attention_5c = attention_5c.unsqueeze(-1).unsqueeze(-1)
            feature_fg_5c = feature_5c * attention_5c
            feature_fg_5c = feature_fg_5c / feature_fg_5c.sum() * feature_5c.sum()
            loss_fg= forward_one_epoch(net, feature_fg_5c, target_var, training=training)
            cost = loss_fg

            l_recon_5c = 0
            recon_feature_5c = cvae('inference', att=attention_5c)
            l_recon_5c += (recon_feature_5c - video_feature['Mixed_5c']).pow(2).mean()
            l_recon = l_recon_5c
            cost += min(epoch, 20)/20 * 0.5 * l_recon

            video_feature_teacher = teacher(input_var, mode='bone')
            attention_5c_teacher = teacher(video_feature_teacher, mode='att')
            attention_5c_teacher = attention_5c_teacher.unsqueeze(-1).unsqueeze(-1)
            KD_loss = MSE_loss(attention_5c, attention_5c_teacher)
            cost += KD_loss * beta
            
            cost /= iter_size
            cost.backward()

            if (n_iter+1) % iter_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            loss_fg_val += loss_fg.cpu().detach().numpy()
            loss_recon_val += l_recon.cpu().detach().numpy()
            cost_val += cost.cpu().detach().numpy()
            loss_KD_val += KD_loss.cpu().detach().numpy()
            pbar.set_postfix(loss='{:.5f}'.format(float(cost.cpu().detach().numpy()))
            , recon='{:.5f}'.format(float(l_recon.cpu().detach().numpy()))
            , KD='{:.5f}'.format(float(KD_loss.cpu().detach().numpy())))

    loss_fg_val /= (n_iter + 1)
    loss_recon_val /= (n_iter +1)
    cost_val /= (n_iter + 1)
    loss_KD_val /= (n_iter + 1)

    if training:
        prefix = 'Train'
        save_model(epoch, net, optimizer, cvae, optimizer_cvae)
    else:
        prefix = 'Val'

    plog = 'Epoch-{} {} Loss: Total - {:.5f}, recon - {:.5f}, KD - {:.5f}'.format(i, prefix, cost_val, loss_recon_val, loss_KD_val)
    print(plog)


if __name__ == '__main__':
    print_training_info()
    set_seed(random_seed)
    """
    Setup model
    """
    net = BDNet_student(in_channels=config['model']['in_channels'], backbone_model=config['model']['backbone_model'])
    net = nn.DataParallel(net, device_ids=gpu_id).cuda()
    cvae = CVAE_student()
    cvae = nn.DataParallel(cvae, device_ids=gpu_id).cuda()

    net_teacher = BDNet_teacher(in_channels=config['model']['in_channels'], backbone_model=config['model']['backbone_model'])
    net_teacher.load_state_dict(torch.load(pretrained_teacher))
    net_teacher.eval().cuda()
    cvae_teacher = CVAE_teacher()
    cvae_teacher.load_state_dict(torch.load(pretrained_cvae))
    cvae_teacher.eval().cuda()
    """
    Setup optimizer
    """
    optimizer = torch.optim.SGD(net.parameters(), learning_rate,
                                momentum=0.9,
                                weight_decay=weight_decay)
    optimizer_cvae = torch.optim.Adam(cvae.parameters(),
                                    lr=0.0001,
                                    betas=(0.8, 0.999))
    """
    Setup dataloader
    """
    if modality == "rgb":
        is_color = True
        scale_ratios = [1.0, 0.875, 0.75, 0.66]
        clip_mean = [0.485, 0.456, 0.406] * new_length
        clip_std = [0.229, 0.224, 0.225] * new_length
    elif modality == "flow":
        is_color = False
        scale_ratios = [1.0, 0.875, 0.75]
        clip_mean = [0.5, 0.5] * new_length
        clip_std = [0.226, 0.226] * new_length
    
    normalize = video_transforms.Normalize(mean=clip_mean,
                                           std=clip_std)
    train_transform = video_transforms.Compose([
            video_transforms.MultiScaleCrop((224,224), scale_ratios),
            video_transforms.RandomHorizontalFlip(),
            video_transforms.ToTensor(),
            normalize,
        ])
    
    # data loading
    train_setting_file = "train_%s_split1.txt" % (modality)
    train_split_file = os.path.join("./datasets/settings/ucf101", train_setting_file)
    if not os.path.exists(train_split_file):
        print("No split file exists in %s directory. Preprocess the dataset first" % ("./datasets/settings/ucf101"))

    train_dataset = datasets.__dict__['ucf101'](root="/Datasets/ucf101_frames",
                                                    source=train_split_file,
                                                    phase="train",
                                                    modality=modality,
                                                    is_color=is_color,
                                                    new_length=new_length,
                                                    video_transform=train_transform)

    print('{} train samples found.'.format(len(train_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    epoch_step_num = len(train_dataset) // batch_size

    """
    Start training
    """
    start_epoch = resume_training(resume, net, optimizer, cvae, optimizer_cvae)

    for i in range(start_epoch, max_epoch + 1):
        run_one_epoch(i, net, optimizer, cvae, optimizer_cvae, net_teacher, cvae_teacher, train_loader, len(train_dataset) // batch_size)
