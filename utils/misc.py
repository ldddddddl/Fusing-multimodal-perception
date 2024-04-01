# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import errno
import os
import sys
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from PIL import Image
__all__ = ['get_mean_and_std', 'init_params', 'mkdir_p', 'AverageMeter', 'PSNR','ACC', 'SSIM']

def norm_to_rgb(in_tensor):
    try:
        img_data_nextframe = np.transpose(in_tensor.cpu().detach(), (1, 2, 0))
        img_data_trans = img_data_nextframe.numpy()
    except:
        img_data_trans = np.transpose(in_tensor, (1, 2, 0))
    # 显示图片
    img_data_nextframe_scaled = (img_data_trans - img_data_trans.min()) / (img_data_trans.max() - img_data_trans.min() + 1e-10) * 255.0
    img_data_nextframe_uint8 = img_data_nextframe_scaled.astype('uint8')
    return img_data_nextframe_uint8

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


def mkdir_p(path):
    '''make dir if not exist'''
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
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


def PSNR(output, target, max_val = 1.0):
    """ Input shape : (N, channel, height, width) """
    output = output.clamp(0.0, max_val)
    target = target.clamp(0.0, max_val)
    if torch.cuda.is_available():
        output = output.cuda()
    mse = torch.pow(target - output, 2).mean()
    if mse == 0:
        return 100, 100
    else:
        return 20 * math.log10(max_val / math.sqrt(mse.item())), mse
    
def ACC(output,target):
    acc=0
    # for i in range(len(output.size(0))):
    #     if output[i,0]>output[i,1]:
    #         targetflag=1
    #     else:
    #         targetflag=0
    #     if targetflag==target[i,:].cpu().data().numpy()
    return 0

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size))
    return window


def SSIM(img1, img2):
    (channel, _, _) = img1.size()
    window_size = 11
    if torch.cuda.is_available():
        img1 = img1.cuda()
    window = create_window(window_size, channel).cuda()
    mu1 = F.conv2d(img1, window, padding = int(window_size/2), groups = channel)
    mu2 = F.conv2d(img2, window, padding = int(window_size/2), groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = int(window_size/2), groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = int(window_size/2), groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = int(window_size/2), groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def flow2rgb(flow_map, max_value=None):
    global args
    c, h, w = flow_map.shape
    # flow_map[:,(flow_map[0] == 0) & (flow_map[1] == 0)] = float('nan')
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map / max_value
    else:
        normalized_flow_map = flow_map / (np.abs(flow_map).max())
    rgb_map[:, :, 0] += normalized_flow_map[0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[:, :, 2] += normalized_flow_map[1]
    return rgb_map.clip(0, 1)

def EPE(input_flow, target_flow, device, sparse=False, mean=True):
    # torch.cuda.init()

    EPE_map = torch.norm(target_flow.cpu() - input_flow.cpu(), 2, (1, 2, 3, 4)) # 2: [8, 8, 112, 112]
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        #target_flow[:, 0] == 0判断x方向上为0的像素点target_flow[:, 1]同理 
        #两个都为0时，标记为1
        mask = (target_flow[:, 0] == 0) & (target_flow[:, 1] == 0)
        EPE_map = EPE_map[~mask.data]
    if mean:
        epe_map_result = EPE_map.mean().cuda()
        return epe_map_result
    else:
        return (EPE_map.sum() / batch_size).cuda()


def realEPE(output, target, device='cuda', sparse=False):
    b, d, n, h, w = target.size()

    # upsampled_output = nn.functional.upsample(output, size=(h, w), mode="bilinear")
    if output.shape == target.shape:
        upsampled_output = output
    else:
        upsampled_output = nn.functional.interpolate(output, size=(n, h, w), mode="trilinear", align_corners=False)
    return EPE(upsampled_output, target, device, sparse, mean=True), upsampled_output



def kl_normal(qm, qv, pm, pv):
    element_wise = 0.5 * (
        torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1
    )
    kl = torch.sum(element_wise, dim=-1)
    return kl

def save_results_images(label_images, predict_images, formatted_datetime, epoch=101):

    for batch in range(label_images.size(0)):
        for img_num in range(label_images.size(2)):
            label_img_array = label_images[batch, :, img_num, :, :]
            predict_img_array = predict_images[batch, :, img_num, :, :]
            label_rgb = norm_to_rgb(label_img_array)
            predict_rgb = norm_to_rgb(predict_img_array)
            label_img = Image.fromarray(label_rgb)
            predict_img = Image.fromarray(predict_rgb)
            label_img.save(f'./checkpoint_sgd/results/{formatted_datetime}/label_epoch{epoch}_batch{batch}_img{img_num}.png')
            predict_img.save(f'./checkpoint_sgd/results/{formatted_datetime}/predict_epoch{epoch}_batch{batch}_img{img_num}.png')


def correlation_loss(img_encoded, tactile_encoded, flex_encoded=None, is_mean=True):
    mean_img = torch.mean(img_encoded, dim=-1)
    mean_tactile = torch.mean(tactile_encoded, dim=-1)
    if flex_encoded != None:
        mean_flex = torch.mean(flex_encoded, dim=-1)
    corr_img_tactile = torch.div(
        torch.sum(
            torch.mul(torch.sub(img_encoded, mean_img.unsqueeze(-1)), torch.sub(tactile_encoded, mean_tactile.unsqueeze(-1))), dim=-1
        ),
        torch.sqrt(
            torch.mul(torch.sum(torch.pow(torch.sub(img_encoded, mean_img.unsqueeze(-1)), 2) , -1), torch.sum(torch.pow(torch.sub(tactile_encoded, mean_tactile.unsqueeze(-1)), 2) , -1))
        )
    )
    if is_mean:
        corr_result =  torch.mean(corr_img_tactile)
    else:
        corr_result = torch.sum(corr_img_tactile, dim=-1)
    
    return corr_result
    
    
    
class Obj(object):
    def __init__(self, dic):
        self.__dict__.update(dic)
   




def l1_loss(pred, label):
    return torch.mean(torch.norm(label - pred, p=1, dim=(1, 2, 3, 4)))
    

def l2_loss(pred, label):
    return torch.norm(label - pred, p=2, dim=(1, 2, 3, 4))    
    
    
def normalize_tensor(tensor, eps=1e-10):
    norm_factor = torch.norm(tensor, dim=-1, keepdim=True)
    return tensor / (norm_factor + eps)


def cosine_distance(tensor0, tensor1, keep_axis=None):
    tensor0 = normalize_tensor(tensor0)
    tensor1 = normalize_tensor(tensor1)
    return torch.mean(torch.sum(torch.square(tensor0 - tensor1), dim=(1, 2, 3, 4))) / 2.0
    


def charbonnier_loss(x, epsilon=0.001):
    return torch.mean(torch.sqrt(torch.square(x) + torch.square(epsilon)))

def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        # if labels in (0.0, 1.0):
        #     labels = torch.tensor(labels, dtype=logits.dtype, shape=logits.get_shape())
        #     loss = torch.mean(F.binary_cross_entropy(logits, labels))
        # else:
        loss = torch.mean(sigmoid_kl_with_logits(F.sigmoid(logits), F.sigmoid(labels)))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = torch.mean(torch.square(logits - labels))
    elif gan_loss_type == 'SNGAN':
        # this is the form of the loss used in the official implementation of the SNGAN paper, but it leads to
        # worse results in our video prediction experiments
        if labels == 0.0:
            loss = torch.mean(torch.nn.Softplus(logits))
        elif labels == 1.0:
            loss = torch.mean(torch.nn.Softplus(-logits))
        else:
            raise NotImplementedError
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss


def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert targets.dtype==(torch.float32)

    entropy = - targets * torch.log(targets) - (1. - targets) * torch.log(1. - targets)
    return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits) * targets) - entropy


class GradientDifferenceLoss(nn.Module):
    def __init__(self):
        super(GradientDifferenceLoss, self).__init__()

    def forward(self, input_image, target_image):
        # Calculate gradients for both input and target images
        input_gradients_x = torch.abs(input_image[:, :, :, :, :-1] - input_image[:, :, :, :, 1:])
        input_gradients_y = torch.abs(input_image[:, :, :, :-1, :] - input_image[:, :, :, 1:, :])

        target_gradients_x = torch.abs(target_image[:, :, :, :, :-1] - target_image[:, :, :, :, 1:])
        target_gradients_y = torch.abs(target_image[:, :, :, :-1, :] - target_image[:, :, :, 1:, :])

        # Calculate the gradient difference loss
        gdl_loss = torch.sum(torch.sum(torch.abs(input_gradients_x - target_gradients_x), dim=(1, 2, 3, 4)) +
                             torch.sum(torch.abs(input_gradients_y - target_gradients_y), dim=(1, 2, 3, 4)))

        return gdl_loss