# -*- coding: utf-8 -*-

import argparse
import os
import torch
import random
import torch.backends.cudnn as cudnn
import sys
from utils import misc
from utils.misc import mkdir_p


class Options(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    
    def initialize(self): 
        # self.parser = argparse.ArgumentParser( description='STdeblur training.' )
        # # Training

        self.parser.add_argument('--epochs', default=100, type=int, metavar='N',
                                help='number of total epochs to run')
        self.parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                                help='manual epoch number (useful on restarts)') 
        self.parser.add_argument('--batchSize', default=8, type=int, metavar='N',
                                help='input batch size')
       # self.parser.add_argument('--lr', '--learning-rate', default=1e-7, type=float,
                               # metavar='LR', help='initial learning rate')
        self.parser.add_argument('--lr', '--learning-rate', default=3e-5, type=float,
                                metavar='LR', help='initial learning rate')
        self.parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                                help='momentum')
        self.parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                                metavar='W', help='weight decay (default: 1e-4)')
        self.parser.add_argument('--schedule', type=int, nargs='+', default=20,
                                help='Decrease learning rate at these epochs.')
        self.parser.add_argument('--gamma', type=float, default=0.9, help='LR is mult-\
                                 iplied by gamma on schedule.')
        # GPU
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: \
                                e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--manualSeed', type=int, default=9720, help='set a manual seed or None') # 9720
        # Dataset
        self.parser.add_argument('--dataroot', type=str, default="./ICIPDataset", help='path to\
                                images (should have subfolders train/blurred, train/sharp,\
                                val/blurred, val/sharp, test/blurred, test/sharp etc)')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val,\
                                test, etc')
        self.parser.add_argument('--cropWidth', type=int, default=112, help='Crop to\
                                this width')  #112
        self.parser.add_argument('--cropHeight', type=int, default=112, help='Crop to\
                                this height')  #112

        self.parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                                help='number of data loading workers (default: 0)')
        # Checkpoints
        self.parser.add_argument('-c', '--checkpoint', default='checkpoint_sgd', type=str, metavar=\
                                'PATH', help='path to save checkpoint (default: checkpoint)')
        self.parser.add_argument('--resume', default='', type=str, metavar='PATH',
                                help='path to latest checkpoint (default: none)')#./checkpoint/model_best.pth.tar
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of\
                                the experiment. It decides where to store samples and models')
        # miscs
        self.parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                                help='evaluate model on validation set')
        self.parser.add_argument('--model_arch',type=str,default='C3D',help='The model arch you selected')
        #self.parser.add_argument('--model_arch',type=str,default='C3D2',help='The model arch you selected')
        self.parser.add_argument('--deterministic', type=bool, default=False, help='deterministic testing')
        self.parser.add_argument('--z_dim', type=int, default=128, help='Latent space hidden variable')
        self.parser.add_argument('--encoder_mode', type=str, default='convlstm', help='is use convlstm(normal) or not')
        self.parser.add_argument('--is_use_cross_conv', type=bool, default=True, help='use cross convolution')
        self.parser.add_argument('--is_use_poe', type=bool, default=False, help='use product of expert')
        self.parser.add_argument('--num_img', type=int, default=12, help='number of images include training set and label')
        self.parser.add_argument('--slide_window_size', type=int, default=8, help='numbers of training image')
        self.parser.add_argument('--loss_mode', type=str, default='only_l2', help='only_l2/gdl_l1/gdl_l2')
        self.parser.add_argument('--data_mode', type=str, default='full_data', help='full_data/zero_flex_tactile/zero_flex/zero_tactile')
        self.parser.add_argument('--fusion_model', type=str, default='Multimodal_fusion_model', help='Multimodal_fusion_model/POE')
        self.parser.add_argument('--is_use_three_phase_train', type=bool, default=True, help='is use three training phase or not')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()

        self.opt = self.parser.parse_args()
       
        # GPU
       # os.environ['CUDA_VISIBLE_DEVICES'] = self.opt.gpu_ids
        os.environ['CUDA_VISIBLE_DEVICES'] = "0"
        self.opt.use_cuda = torch.cuda.is_available()
        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # Random seed
        if self.opt.manualSeed is None:
            self.opt.manualSeed = random.randint(1, 10000)
        random.seed(self.opt.manualSeed)
        torch.manual_seed(self.opt.manualSeed)
        if self.opt.use_cuda:
            torch.cuda.manual_seed_all(self.opt.manualSeed)
            cudnn.benchmark = True
            cudnn.enabled = True

        args = vars(self.opt)
        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        expr_dir = os.path.join(self.opt.checkpoint, self.opt.name)
        mkdir_p(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt