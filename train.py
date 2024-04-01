# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import os
import shutil
import time
import torch.nn.parallel
from options import Options
from xela_dataloader1 import *

import sys
from utils.progress.progress import bar
from utils.progress.progress.bar import Bar
from utils import misc,logger
from utils.logger import Logger,savefig
from utils.misc import AverageMeter,ACC

from models.models import *    #引入全部
import random
from sklearn.metrics import accuracy_score,recall_score
import torch.nn.functional as F
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR #调整学习率
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, random_split
import warnings 
from utils.misc import (kl_normal, 
                        realEPE,
                        SSIM,PSNR, 
                        norm_to_rgb, 
                        save_results_images, 
                        correlation_loss,
                        l1_loss, l2_loss,
                        cosine_distance, 
                        charbonnier_loss, 
                        gan_loss,
                        GradientDifferenceLoss)
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc, roc_auc_score
from models.encoders import frame_to_stream

# 忽略VisibleDeprecationWarning警告
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)    
# 忽略FutureWarning警告
warnings.filterwarnings("ignore", category=FutureWarning)


OPT = Options().parse()
def main():
        start_epoch = OPT.start_epoch  # start from epoch 0 or last checkpoint epoch
        OPT.phase = 'train'
        transform_v = transforms.Compose([transforms.Resize([OPT.cropWidth, OPT.cropHeight]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_t = transforms.Compose([transforms.Resize([4, 4]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        transform_f = transforms.Compose([transforms.Resize([4, 4]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        #trainset = MyDataset('../Grasping_state_assessment_master(2)/graspingdata',6,3,3,transform_v,transform_t,transform_f,log,flag = opt.phase)    
        #一张图片对应三个f、t的数据点
        trainset = MyDataset1('../Grasping_state_assessment_master/sorted data/traindata', OPT.num_img , 3 * (OPT.num_img + 1), 3 * (OPT.num_img + 1), transform_v,transform_t,transform_f, OPT.slide_window_size, flag = OPT.phase)
        train_loader = torch.utils.data.DataLoader(
            dataset=trainset,
            #dataset=train_set,
            batch_size=OPT.batchSize,
            shuffle=True,
            num_workers=0
        )

        #testset = MyDataset('../Grasping_state_assessment_master(2)/graspingdata',6,3,3,transform_v,transform_t,transform_f,log,flag = opt.phase)
        OPT.phase = 'test'
        testset = MyDataset2('../Grasping_state_assessment_master/sorted data/testdata', OPT.num_img , 3 * (OPT.num_img + 1), 3 * (OPT.num_img + 1), transform_v,transform_t,transform_f, OPT.slide_window_size, flag = OPT.phase)
        val_loader = torch.utils.data.DataLoader(
            dataset=testset,
            batch_size=OPT.batchSize,
            shuffle=True,
            num_workers=0
        )

    # Model
        if OPT.model_arch == 'C3D_visual_only':
            model = C3D_visual_only(drop_p_v=0.2,visual_dim=4096, fc_hidden_1=128, num_classes=2)    
        elif OPT.model_arch == 'C1D_tactile_test1':
            model = C1D_tactile_test1()
        elif OPT.model_arch == 'C1D_flex':
            model = C1D_flex()
        elif OPT.model_arch == 'C3D':
            model = C3D(drop_p_v=0.2, 
                        visual_dim=4096, 
                        fc_hidden_1= 128,
                        fc_hidden_t = 20,
                        fc_hidden_f = 20,
                        num_classes = 2, 
                        num_img=OPT.num_img,
                        is_ablation=OPT.data_mode,
                        fusion_flag=OPT.fusion_model,
                        deterministic=OPT.deterministic,
                        z_dim=OPT.z_dim,
                        encoder_mode=OPT.encoder_mode,
                        is_use_cross_conv=OPT.is_use_cross_conv,
                        is_use_poe=OPT.is_use_poe,
                        slide_window_size=OPT.slide_window_size,
                        device='cuda' if torch.cuda.is_available() else 'cpu'
                    )   

        if OPT.use_cuda:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model.to( torch.device('cpu') )
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
    
        # Loss and OPTimizer
        # criterion = nn.CrossEntropyLoss(reduction='sum')
        # OPTimizer = torch.OPTim.Adam(model.parameters(), lr=OPT.lr,betas=(0.9, 0.999), eps=1e-08,weight_decay=OPT.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), lr=OPT.lr)
        # model.apply(init_weights_xavier)
        # Resume
        title = OPT.name
        if OPT.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isfile(OPT.resume), 'Error: no checkpoint directory found!'
            OPT.checkpoint = os.path.dirname(OPT.resume)
            checkpoint = torch.load(OPT.resume)
            best_acc = checkpoint['best_acc']
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger = Logger(os.path.join(OPT.checkpoint, 'log.txt'), title=title, resume=True)
        else:
            logger = Logger(os.path.join(OPT.checkpoint, 'log.txt'), title=title)
            logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Valid PSNR.'])
    
        if OPT.evaluate:
            print('\nEvaluation only')
            val_loss, val_psnr = valid(val_loader, model, start_epoch, OPT.use_cuda)
            print(' Test Loss:  %.8f, Test PSNR:  %.2f' % (val_loss, val_psnr))
            return
    
        # Train and val
    
        best_acc = 0
        train_acc_list=[]
        train_loss_list=[]
        train_loss_total_list = []
        test_acc_list=[]
        test_loss_list=[]
        test_loss_total_list = []
        kl_list = []
        SSIM_results_list = []
        PSNR_results_list = []
        mse_results_list = []
        train_tac_loss_list = []
        test_tac_loss_list = []
        train_corr_loss_list = []
        test_corr_loss_list = []
        tactile_dict = {}
        auc_results_dict = {'FPR': [], 'TPR': [], 'Thresholds': []}
        auc_list = []
        tsne_dict = {}
        frame1_ssim, frame2_ssim, frame3_ssim, frame4_ssim = [], [], [], []
        frame1_psnr, frame2_psnr, frame3_psnr, frame4_psnr = [], [], [], []
        frame1_mse, frame2_mse, frame3_mse, frame4_mse = [], [], [], []
        train_l2_loss_list, train_l1_loss_list, train_gdl_loss_list = [], [], []
        test_l2_loss_list, test_l1_loss_list, test_gdl_loss_list = [], [], []
        ctime = time.ctime()      
        ctime_datetime = datetime.strptime(ctime, "%a %b %d %H:%M:%S %Y")
        formatted_datetime = ctime_datetime.strftime("%y-%m-%d-%H-%M-%S")
        if not os.path.exists('./checkpoint_sgd/results'):
            os.mkdir('./checkpoint_sgd/results')   
        
        if not os.path.exists('./checkpoint_sgd/results/' + formatted_datetime):
            os.mkdir('./checkpoint_sgd/results/' + formatted_datetime)
            
        
        # OPT.epochs = 20   ###测试
        for epoch in range(start_epoch, OPT.epochs):
            '''
            定义训练阶段：
            generate(0-24)：训练生成器，隐变量z从标准高斯中采样
            inference(25-49)：训练推理网络，KL散度为0
            add_kl(50-74): 按比例加入KL，直到KL为预设比例
            full_training(75-100)： 完整训练整个网络
            '''
            trainning_phase = ''
            use_tired_training = OPT.is_use_three_phase_train
            t = epoch / OPT.epochs
            if epoch / OPT.epochs < 0.25 and use_tired_training:
                trainning_phase = 'generate'
            elif 0.25 <= epoch / OPT.epochs < 0.5 and use_tired_training:
                trainning_phase = 'inference'
            elif 0.5 <= epoch / OPT.epochs < 0.75 and use_tired_training:
                trainning_phase = 'add_kl' 
            elif 0.75 <= epoch / OPT.epochs and use_tired_training:
                trainning_phase = 'full_training'
                
            print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, OPT.epochs, OPT.lr))
            #train 

            
            train_return_data = train(train_loader, model, optimizer, epoch, OPT.use_cuda, OPT.deterministic,trainning_phase=trainning_phase)
            train_loss, train_acc, train_loss_total, kl, train_l2_loss, train_l1_loss, train_gdl_loss, train_tac_loss, train_corr_loss = train_return_data
                
            #valid
            label, test_upsample_flow, alpha_list, avg_list_avg, outputs, targets, grasping_pred, metric_data_list = valid(val_loader, model,  epoch, OPT.use_cuda, OPT.deterministic, training_phase='full_training')
            test_loss, test_acc, test_loss_total, avg_ssim, avg_psnr, avg_mse, test_tac_loss, test_corr_loss, test_l2_loss, test_l1_loss, test_gdl_loss  = avg_list_avg
        


            ##处理不同每一帧对应的SSIM,PSNR,MSE
            metric_array = numpy.array(metric_data_list)

            frame1_ssim.append(metric_array[0, :, 0].mean())
            frame2_ssim.append(metric_array[0, :, 1].mean())
            frame3_ssim.append(metric_array[0, :, 2].mean())
            frame4_ssim.append(metric_array[0, :, 3].mean())
        
            frame1_psnr.append(metric_array[1, :, 0].mean())
            frame2_psnr.append(metric_array[1, :, 1].mean())
            frame3_psnr.append(metric_array[1, :, 2].mean())
            frame4_psnr.append(metric_array[1, :, 3].mean())
        
            frame1_mse.append(metric_array[2, :, 0].mean())
            frame2_mse.append(metric_array[2, :, 1].mean())
            frame3_mse.append(metric_array[2, :, 2].mean())
            frame4_mse.append(metric_array[2, :, 3].mean())

            if epoch == OPT.epochs - 1:
                is_show_plt = True
            else:
                is_show_plt = False
            
            if not OPT.deterministic:
                low_dimension_data, true_labels = tsne_compute(outputs.grasping_result.detach().cpu().numpy(), label=targets.detach().cpu().numpy(), formatted_datetime=formatted_datetime, is_show_plt=is_show_plt, )

            # 检查是否包含多个类别
            unique_classes2 = np.unique(targets.detach().cpu().numpy())
            if len(unique_classes2) >= 2:
                fpr, tpr, auc, thresholds = auc_compute(torch.argmax(outputs.grasping_fused, dim=-1).detach().cpu().numpy(), targets.detach().cpu().numpy(), formatted_datetime, is_show_plt=is_show_plt)
                auc_results_dict['FPR'].append(fpr)
                auc_results_dict['TPR'].append(tpr)
                auc_results_dict['Thresholds'].append(thresholds)
                
                auc_list.append(auc)
                
            else:
                print("ROC AUC cannot be calculated as there is only one class in y_true.")
            '''
            保存最后十组图片
            '''
            if OPT.epochs - 30 < epoch :
                save_results_images(label[0], test_upsample_flow, formatted_datetime, epoch)
                temp = outputs.tactile_stream.detach().cpu().numpy()
                for t_count, t in enumerate(temp):
                    tactile_dict[f'pred_tac_{epoch}_batch{t_count}'] = t
                    if 1 in label[4].shape:
                        tactile_dict[f'label_{epoch}_batch{t_count}'] = label[4][t_count].detach().cpu().numpy().squeeze()
                    else:
                        tactile_dict[f'label_{epoch}_batch{t_count}'] = label[4][t_count].detach().cpu().numpy()
            '''
            保存最后30组tsne数据
            '''            
            if OPT.epochs - 30 < epoch and not OPT.deterministic:
                tsne_dict[f'low_dimension_data_{epoch}1'] = low_dimension_data[:, 0]
                tsne_dict[f'low_dimension_data_{epoch}2'] = low_dimension_data[:, 1]
                tsne_dict[f'true_labels_{epoch}'] = true_labels
            

            # append logger file
            logger.append([OPT.lr, train_loss, test_loss, test_acc])
            train_acc_list.append(train_acc)
            train_loss_list.append(train_loss)
            train_loss_total_list.append(train_loss_total)
            test_loss_list.append(test_loss)
            test_acc_list.append(test_acc)
            test_loss_total_list.append(test_loss_total)
            kl_list.append(kl)
            train_l2_loss_list.append(train_l2_loss)
            train_l1_loss_list.append(train_l1_loss)
            train_gdl_loss_list.append(train_gdl_loss)
            test_l2_loss_list.append(test_l2_loss)
            test_l1_loss_list.append(test_l1_loss)
            test_gdl_loss_list.append(test_gdl_loss)
            SSIM_results_list.append(avg_ssim)
            PSNR_results_list.append(avg_psnr)
            mse_results_list.append(avg_mse)
            train_tac_loss_list.append(train_tac_loss)
            test_tac_loss_list.append(test_tac_loss)
            train_corr_loss_list.append(train_corr_loss)
            test_corr_loss_list.append(test_corr_loss)
            # save best model
            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'model':model,
                    'acc': test_acc,
                    'best_acc': best_acc,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, checkpoint=OPT.checkpoint, filename=f'results/{formatted_datetime}/model_{formatted_datetime}.pth.tar')
            print('Best acc:')
            print(best_acc)
    
        logger.close()  
        
        '''
        保存数据
        '''

        total_data = {
            "Grasping training accuracy":train_acc_list,
            "Grasping test accuracy":test_acc_list,
            "Grasping training loss":train_loss_list,
            "Grasping test loss":test_loss_list,
            "train tactile loss":train_tac_loss_list,
            "test tactile loss":test_tac_loss_list,
            "train correlation loss":train_corr_loss_list,
            "test correlation loss":test_corr_loss_list,
            "training loss total":train_loss_total_list,
            "test loss total":test_loss_total_list,
            "KL":kl_list,
            "train l2 loss":train_l2_loss_list,
            "test l2 loss":test_l2_loss_list,            
            "train l1 loss":train_l1_loss_list,
            "test l1 loss":test_l1_loss_list,            
            "train gdl loss":train_gdl_loss_list,
            "test gdl loss":test_gdl_loss_list,
            "SSIM results":SSIM_results_list,
            "PSNR results":PSNR_results_list,
            "MSE results":mse_results_list,
        }
        ssim_data = {
            "ssim_avg":SSIM_results_list,
            "frame1":frame1_ssim,
            "frame2":frame2_ssim,
            "frame3":frame3_ssim,
            "frame4":frame4_ssim,
        }
        psnr_data = {
            "psnr_avg":PSNR_results_list,
            "frame1":frame1_psnr,
            "frame2":frame2_psnr,
            "frame3":frame3_psnr,
            "frame4":frame4_psnr,
        }
        mse_data = {
            "mse_avg":mse_results_list,
            "frame1":frame1_mse,
            "frame2":frame2_mse,
            "frame3":frame3_mse,
            "frame4":frame4_mse,
        }
        if not os.path.exists('./checkpoint_sgd/results'):
            os.mkdir('./checkpoint_sgd/results')   
        
        if not os.path.exists('./checkpoint_sgd/results/' + formatted_datetime):
            os.mkdir('./checkpoint_sgd/results/' + formatted_datetime)
        writer = pd.ExcelWriter(os.path.join(f'./checkpoint_sgd/results/{formatted_datetime}', formatted_datetime + '.xlsx'))    
        df = pd.DataFrame(total_data)
        df2 = pd.DataFrame(tactile_dict)
        df3 = pd.DataFrame(auc_results_dict)
        df4 = pd.DataFrame(tsne_dict)
        df5 = pd.DataFrame(ssim_data)
        df6 = pd.DataFrame(psnr_data)
        df7 = pd.DataFrame(mse_data)
        df.to_excel(writer, sheet_name='results', index=False, header=True)
        df2.to_excel(writer, sheet_name='tactile_prediction', index=False, header=True)
        df3.to_excel(writer, sheet_name='AUC data', index=False, header=True)
        df4.to_excel(writer, sheet_name='tSNE data', index=False, header=True)
        df5.to_excel(writer, sheet_name='SSIM data', index=False, header=True)
        df6.to_excel(writer, sheet_name='PSNR data', index=False, header=True)
        df7.to_excel(writer, sheet_name='MSE data', index=False, header=True)
        writer.close()
        '''
        保存模型及关键参数
        '''
        # torch.save(model, f'./checkpoint_sgd/results/{formatted_datetime}/model_{formatted_datetime}.pth')
        # torch.save(model.state_dict(), f'./checkpoint_sgd/results/{formatted_datetime}/model_state_{formatted_datetime}.pth')
        
        np.save(
            f'XELA_results/train/train_acc_{OPT.model_arch}_{OPT.batchSize}_{OPT.lr}_{formatted_datetime}.npy',
            train_acc_list)
        np.save(
            f'XELA_results/train/train_loss_{OPT.model_arch}_{OPT.batchSize}_{OPT.lr}_{formatted_datetime}.npy', 
                train_loss_list)
        np.save(
            f'XELA_results/train/train_loss_total_{OPT.model_arch}_{OPT.batchSize}_{OPT.lr}_{formatted_datetime}.npy', 
                train_loss_total_list)
        np.save(
            f'XELA_results/test/test_acc_{OPT.model_arch}_{OPT.batchSize}_{OPT.lr}_{formatted_datetime}.npy',
            test_acc_list)
        np.save(
            f'XELA_results/test/test_loss_{OPT.model_arch}_{OPT.batchSize}_{OPT.lr}_{formatted_datetime}.npy',
            test_loss_list)
        np.save(
            f'XELA_results/test/test_loss_total_{OPT.model_arch}_{OPT.batchSize}_{OPT.lr}_{formatted_datetime}.npy',
            test_loss_total_list)
        '''
        保存图片
        '''
        save_results_images(label_images=label[0], predict_images=test_upsample_flow, formatted_datetime=formatted_datetime)
        new_line = "\n"
        config = [
            f'alpha_optical_flow:{alpha_list[0]} \nalpha_kl:{alpha_list[1]} \nalpha_grasping:{alpha_list[2]} \nalpha_tac:{alpha_list[3]} \nalpha_corr:{alpha_list[4]} \nalpha_l1:{alpha_list[5]} \nalpha_gdl:{alpha_list[6]} \n',
            f'best_acc:{best_acc} \n'
            f'------------------------\n',
            f'{str(OPT)[10:-1].replace(",", new_line)}',
        ]
        with open(os.path.join(os.path.join(f'./checkpoint_sgd/results/{formatted_datetime}', formatted_datetime + '.txt')), 'w', encoding='utf-8') as f:
            for line in config:
                f.write(line)
            f.write('\n')
          
        fig1, axes_table = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
        axes_table[0][0].plot(train_loss_list, label='train_loss')
        axes_table[0][0].plot(test_loss_list, label='test_loss')
        axes_table[0][0].set_title('Grasping loss')
        
        axes_table[0][1].plot(train_acc_list, label='train_acc')
        axes_table[0][1].plot(test_acc_list, label='test_acc')
        axes_table[0][1].set_title('Grasping accuracy')
        
        axes_table[0][2].plot(train_tac_loss_list, label='train_tactile_loss')
        axes_table[0][2].plot(test_tac_loss_list, label='test_tactile_loss')
        axes_table[0][2].set_title('images to tactile loss')

        axes_table[1][0].plot(train_loss_total_list, label='train_loss_total')
        axes_table[1][0].plot(test_loss_total_list, label='test_loss_total')
        axes_table[1][0].set_title('Total loss')
        
        axes_table[1][1].plot(train_l2_loss_list if sum(train_l2_loss_list)/len(train_l2_loss_list) == -1 else train_l1_loss_list, label=f'train_{OPT.loss_mode}')
        axes_table[1][1].plot(test_l2_loss_list if sum(test_l2_loss_list)/len(test_l2_loss_list) == -1 else test_l1_loss_list, label=f'test_{OPT.loss_mode}')
        axes_table[1][1].set_title('generate_loss')
        
        axes_table[1][2].plot(train_corr_loss_list, label='train_corr_loss')
        axes_table[1][2].plot(test_corr_loss_list, label='test_corr_loss')
        axes_table[1][2].set_title('correlation loss')
        
        axes_table[2][0].plot(PSNR_results_list, label='PSNR_results')
        axes_table[2][0].set_title('Batch average PSNR')
        
        axes_table[2][1].plot(SSIM_results_list, label='SSIM_results')
        axes_table[2][1].set_title('Batch average SSIM')
        
        axes_table[2][2].plot(mse_results_list, label='mse_results')
        axes_table[2][2].set_title('Batch average MSE')

        plt.tight_layout()
        
        for i in axes_table:
            for j in i:
                j.legend()
        plt.savefig(os.path.join(f'./checkpoint_sgd/results/{formatted_datetime}', f'{fig1}'))
        
        visual_target = label[0][-1]
        test_upsample_flow_batch = test_upsample_flow[-1].cpu().detach().numpy()
        fig2, axes = plt.subplots(nrows=2, ncols=visual_target.size(1), figsize=(24, 24))
        
        # 迭代显示每张图片
        for img_idx in range(visual_target.size(1)):
            # 获取当前图片的数据
            img_data_nextframe = visual_target[:, img_idx, :, :]
            img_data_nextframe_uint8 = norm_to_rgb(img_data_nextframe)
            axes[0, img_idx].imshow(img_data_nextframe_uint8)
            axes[0, img_idx].set_title(f'label {img_idx}')
            axes[0, img_idx].axis('off')
            
        for img_idx in range(visual_target.size(1)):
            # 获取当前图片的数据
            visual = test_upsample_flow_batch[:, img_idx, :, :]
            img_data_predict_scaled = norm_to_rgb(visual)
            axes[1, img_idx].imshow(img_data_predict_scaled)
            axes[1, img_idx].set_title(f'predict {img_idx}')
            axes[1, img_idx].axis('off')
            
        # 调整子图之间的间距
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(f'./checkpoint_sgd/results/{formatted_datetime}', f'{fig2}.png'))
        
        # 显示图像
        plt.show()
        plt.pause(0)



def train(trainloader, model, optimizer, epoch, use_cuda, deterministic, trainning_phase):
    # switch to train mode
        model.train()
    
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_loss_grasping = AverageMeter()
        avg_loss_total_grasping = AverageMeter()
        avg_acc_grasping = AverageMeter()
        psnr_input = AverageMeter()
        avg_flow_loss = AverageMeter()
        avg_kl = AverageMeter()
        avg_mse = AverageMeter()
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_tac_loss = AverageMeter()
        avg_corr_loss = AverageMeter()
        avg_l1_loss = AverageMeter()
        avg_gdl_loss = AverageMeter()
        end = time.time()
        res_times = 10
        temp_grasping_loss = 0
        temp_ssim = 0
        is_res_connection = False

        bar = Bar('Processing', max=len(trainloader))
        for batch_idx, (x_visual,x_tactile,x_flex, label) in enumerate(trainloader):
            '''
            label{
                [0]:visual
                [1]:tactile 用于给图片生成加入额外的信息
                [2]:flex
                [3]:grasp_state
                [4]:tactile 用于跨模态预测的标签
            }
            win11的版本和win10的版本有差别 win10 = True, win11 = false
            '''
            is_new_version = True
            if is_new_version:
                x_flex = x_flex.squeeze(-1)
                x_tactile = x_tactile.squeeze(-1)
                label[1] = label[1].squeeze(-1)
                label[2] = label[2].squeeze(-1)
                label[4] = label[4].squeeze(-1)
            
            # measure data loading time

            data_time.update(time.time() - end)
            if label[3].shape[0] < trainloader.batch_size:
                continue
            x_tactile, x_visual, x_flex, targets = torch.autograd.Variable(x_tactile),torch.autograd.Variable(x_visual),torch.autograd.Variable(x_flex), torch.autograd.Variable(label[3])
            if use_cuda:
                # inputs = inputs.cuda()
                x_tactile=x_tactile.cuda()#[8,48]
                x_visual=x_visual.cuda()#[8,3,16,112,112]
                x_flex=x_flex.cuda()#[8,48]
                targets = targets.cuda(non_blocking=True)#[8]
                label = [_.cuda() for _ in label]
             # 改变触觉数据的 形式
            x_tactile = x_tactile.to(torch.float32)
            x_tactile = x_tactile.unsqueeze(1) 
            x_flex = x_flex.to(torch.float32)
            x_flex = x_flex.unsqueeze(1)
            label[1] = label[1].to(torch.float32)
            label[2] = label[2].to(torch.float32)
            label[4] = label[4].to(torch.float32)
            # compute output

            outputs = model(x_visual, x_tactile, x_flex, label, trainning_phase)# [8,2] 将视觉触觉拉伸输入进默认模型，得出的结果x即为outputs--》预测值（此时是概率）
            
            (grasping_pred, 
            upsample_flow, 
            loss_total, 
            avg_list,
            alpha_list,
            metric_data_list) = loss_calc((targets, label), 
                                outputs, 
                                use_cuda, 
                                trainloader.batch_size, 
                                deterministic,
                                avg_loss_grasping, 
                                avg_corr_loss,
                                avg_loss_total_grasping, 
                                avg_acc_grasping, 
                                avg_kl, 
                                avg_flow_loss,
                                avg_tac_loss,
                                avg_ssim, 
                                avg_psnr, 
                                avg_mse,
                                avg_l1_loss,
                                avg_gdl_loss,
                                training_phase=trainning_phase,
                                epoch=epoch,
                                train_visula_last_frame=x_visual
                                )

            [avg_loss_grasping, avg_acc_grasping, avg_loss_total_grasping, avg_kl, avg_flow_loss, 
                avg_l1_loss, avg_gdl_loss, avg_ssim, avg_psnr, avg_mse, avg_tac_loss, avg_corr_loss] = avg_list
            return_avg_list = [avg_loss_grasping.avg, avg_acc_grasping.avg, avg_loss_total_grasping.avg, avg_kl.avg, avg_flow_loss.avg, avg_l1_loss.avg, avg_gdl_loss.avg, avg_tac_loss.avg, avg_corr_loss.avg]
            
            ####改到这里， 还有valid函数的返回值，显示l1的损失值， 保存的损失值没改
            
            if abs(avg_ssim.avg - temp_ssim) > 0.001*alpha_list[0] and not is_res_connection or batch_idx < 20 or True:
                # compute gradient and do SGD step
                optimizer.zero_grad()   
                loss_total.backward()
                optimizer.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                temp_ssim = avg_ssim.avg
                bar.suffix  = '({batch}/{size}) Data:{data:.3f}s|Batch: {bt:.3f}s|Total: {total:}|ETA:{eta:}|'\
                    'LossTotal:{loss_total:.4f}|LossGrasping:{losses_grasping:.4f}|ACC(input):{acc_grasping:.4f}|'\
                    'Tac_loss:{tactile_loss:.4f}|Corr_loss:{correlation_loss:.4f}|KL:{kl:.4f}|SSIM:{ssim:.4f}|PSNR:{psnr:.4f}'.format(
                            batch=batch_idx + 1,
                            size=len(trainloader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss_total = avg_loss_total_grasping.avg, 
                            losses_grasping = avg_loss_grasping.avg,
                            acc_grasping = avg_acc_grasping.avg,
                            tactile_loss = avg_tac_loss.avg,
                            correlation_loss = avg_corr_loss.avg,
                            kl = avg_kl.avg,
                            ssim = avg_ssim.avg,
                            psnr = avg_psnr.avg,
                            
                            # psnr_in=psnr_input.avg
                            )
                bar.next()
            else:
                # print('/r-')
                continue
            
        bar.finish()
        return return_avg_list
    
def loss_calc(sample_label, 
                outputs, 
                device, 
                batch_size, 
                deterministic,
                avg_loss_grasping, 
                avg_corr_loss,
                avg_loss_total_grasping, 
                avg_acc_grasping, 
                avg_kl, 
                avg_flow_loss,
                avg_tac_loss,
                avg_ssim=None,
                avg_psnr=None,
                avg_mse=None,
                avg_l1_loss=None,
                avg_gdl_loss=None,
                training_phase='full_training',
                epoch=0,
                loss_mode=OPT.loss_mode,
                train_visula_last_frame=None
                ):
    '''
    sample_label: tuple, include grasping and generate
    predict_list: tuple, output data
    device: use_cuda
    batch_size: -
    '''
    (grasping_label, label_list) = sample_label
    visual_label, _, flex_label, grasp_state, tac_label = label_list
    # 编码器输出结果
    visual_encoded, flex_encoded, tactile_encoded = outputs.encoded_data  

    
    # Weights for loss
    alpha_tac = 20.0
    alpha_optical_flow = 20.0   # configs["opticalflow"]=10
    alpha_kl_final = 0.2
    alpha_grasping = 15.0
    alpha_corr = 100.0
    alpha_l1 = 0.5
    alpha_gdl = 0.02
    ## 图片生成触觉的损失
    mse_loss = nn.MSELoss().cuda()
    #gdl
    gdl_loss = GradientDifferenceLoss().to(device='cuda' if device else 'cpu')
    
    if len(tac_label.shape) > 2:
        tac_label = tac_label.squeeze(1)
    tac_loss = alpha_tac * mse_loss(outputs.tactile_stream, tac_label.float())
    # 相关性损失
    corr_loss = alpha_corr * correlation_loss(visual_encoded, tactile_encoded)
    if deterministic:
        kl = -1
    else:
        if training_phase == 'inference':
            alpha_kl = 0
        elif training_phase == 'add_kl':
            alpha_kl = (epoch + 1 - OPT.epochs * 0.5) / (OPT.epochs * 0.25) * alpha_kl_final
        else:
            alpha_kl = alpha_kl_final
        kl = alpha_kl * torch.mean(
                    kl_normal(outputs.mu_z_encoded, 
                            outputs.var_z_encoded, 
                            outputs.mu_prior_encoded, 
                            outputs.var_prior_encoded
                            )
            )
    if loss_mode in ['only_l2', 'gdl_l2']:
        
        '''
        先将flow2上采样到optical_flow_label大小
        然后使用torch.norm计算预测与真实图片的L2范数
        
        '''
        realEPE_reslt, _ = realEPE(
                outputs.next_stream, visual_label, device
            )
        flow_loss = alpha_optical_flow * realEPE_reslt
        '''
        抓取损失计算
        '''
    elif loss_mode in ['gdl_l1', 'only_l1']:
        
        l1_loss_result = alpha_l1 * l1_loss(outputs.next_stream, visual_label)
        #GAN
        # gan_loss_result = alpha_gan * gan_loss(outputs.next_stream, visual_label, gan_loss_type='GAN')
    if loss_mode in ['gdl_l1', 'gdl_l2']:
        gdl_loss_result = alpha_gdl * gdl_loss(outputs.next_stream, visual_label)
    loss_grasping = F.cross_entropy(outputs.grasping_fused, grasping_label, reduction='mean') #对loss求平均后返回，默认即为mean
    loss_grasping = alpha_grasping * loss_grasping
    if deterministic:
        loss_total = (
            flow_loss
            + loss_grasping
            + tac_loss
            - corr_loss
            ).requires_grad_(True)
    elif loss_mode == 'only_l2':
        loss_total = (
            flow_loss
            + loss_grasping
            + kl
            + tac_loss
            - corr_loss
            ).requires_grad_(True)
        l1_loss_result = -1
        gdl_loss_result = -1
    elif loss_mode == 'gdl_l1':
        loss_total = (
            l1_loss_result
            + gdl_loss_result
            + loss_grasping
            + kl
            + tac_loss
            - corr_loss
            ).requires_grad_(True)
        flow_loss = -1
    elif loss_mode == 'gdl_l2':
        loss_total = (
            flow_loss
            + gdl_loss_result
            + loss_grasping
            + kl
            + tac_loss
            - corr_loss
            ).requires_grad_(True)
        l1_loss_result = -1
    grasping_pred = torch.max(outputs.grasping_fused, 1)[1]  # y_pred != output dim=1表示输出所在行（每个样本）的预测类别 1--输出行最大值 0--列最大值
    acc_grasping =  accuracy_score(grasping_pred.cpu().data.numpy(), grasping_label.cpu().data.numpy())  #准确率             
    '''
    PSNR/SSIM/MSE
    取最后一个batch的最后一张图片，这张图片是没有进行训练的Ground Truth
    '''
    ssim_list = []
    psnr_list = []
    mse_list = []
    for num in range(visual_label.shape[2]):
        SSIM_result = SSIM(visual_label[-1][:, num, :, :], outputs.next_stream[-1][:, num, :, :])
        PSNR_result, mse = PSNR(visual_label[-1][:, num, :, :,].unsqueeze(0), outputs.next_stream[-1][:, num, :, :,].unsqueeze(0))
        ssim_list.append(SSIM_result.item())
        psnr_list.append(PSNR_result)
        mse_list.append(mse.item())
    SSIM_result = sum(ssim_list) / len(ssim_list)
    PSNR_result = sum(psnr_list) / len(psnr_list)
    mse = sum(mse_list) / len(mse_list)
    
    # measure the result
    avg_loss_grasping.update(loss_grasping.item(), batch_size)
    avg_acc_grasping.update(acc_grasping, batch_size)
    avg_loss_total_grasping.update(loss_total.item(), batch_size)

    avg_kl.update(kl if kl==-1 else kl.item(), batch_size)
    avg_flow_loss.update(flow_loss if flow_loss==-1 else flow_loss.item(), batch_size)
    avg_l1_loss.update(l1_loss_result if l1_loss_result==-1 else l1_loss_result.item(), batch_size)
    avg_gdl_loss.update(gdl_loss_result if gdl_loss_result==-1 else gdl_loss_result.item(), batch_size)
    avg_ssim.update(SSIM_result)
    avg_psnr.update(PSNR_result)
    avg_mse.update(mse)
    avg_tac_loss.update(tac_loss.item())
    avg_corr_loss.update(corr_loss.item())

    avg_list = [avg_loss_grasping, avg_acc_grasping, avg_loss_total_grasping, avg_kl, avg_flow_loss,
                avg_l1_loss, avg_gdl_loss, avg_ssim, avg_psnr, avg_mse, avg_tac_loss, avg_corr_loss]
    alpha_list = [alpha_tac, alpha_optical_flow, alpha_kl_final, alpha_grasping, alpha_corr, alpha_l1, alpha_gdl]
    # psnr_input.update(psnr_i, inputs.size(0))

    return (
        grasping_pred,
        outputs.next_stream,
        loss_total,
        avg_list,
        alpha_list,
        (ssim_list, psnr_list, mse_list)
    )

        
        
        
        
def valid(testloader, model, epoch, use_cuda, deterministic, training_phase='full_training'):
    # switch to test mode
        model.eval()
        
        batch_time = AverageMeter()
        data_time = AverageMeter()
        avg_loss_grasping = AverageMeter()
        avg_loss_total_grasping = AverageMeter()
        avg_acc_grasping = AverageMeter() 
        psnr_input = AverageMeter()
        avg_kl = AverageMeter()
        avg_flow_loss = AverageMeter()
        avg_mse = AverageMeter()
        avg_psnr = AverageMeter()
        avg_ssim = AverageMeter()
        avg_tac_loss = AverageMeter()
        avg_corr_loss = AverageMeter()
        avg_l1_loss = AverageMeter()
        avg_gdl_loss = AverageMeter()
        ###
        temp_ssim = 0
        filt_ssim_flag = True
        end = time.time()
        
        y_targets_list = []
        y_pred_list = []
        ssim_frame_list = []
        psnr_frame_list = []
        mse_frame_list = []        


        bar = Bar('Processing', max=len(testloader))
        for batch_idx, (x_visual, x_tactile, x_flex, label) in enumerate(testloader):
            '''
            win11的版本和win10的版本有差别 win10 = True, win11 = false
            '''
            is_new_version = True
            if is_new_version:
                x_flex = x_flex.squeeze(-1)
                x_tactile = x_tactile.squeeze(-1)
                label[1] = label[1].squeeze(-1)
                label[2] = label[2].squeeze(-1)
                label[4] = label[4].squeeze(-1)
            
            # measure data loading time
            data_time.update(time.time() - end)
            if label[3].shape[0] < testloader.batch_size:
                continue    
            x_tactile, x_visual,x_flex, targets = torch.autograd.Variable(x_tactile), torch.autograd.Variable(
                x_visual),torch.autograd.Variable(x_flex), torch.autograd.Variable(label[3])
            if use_cuda:
                # inputs = inputs.cuda()
                x_tactile = x_tactile.cuda()
                x_visual = x_visual.cuda()
                x_flex = x_flex.cuda()
                targets = targets.cuda(non_blocking=True)
                label = [_.cuda() for _ in label]
                
                        # 改变触觉数据的 形式
            x_tactile = x_tactile.to(torch.float32)
            x_tactile = x_tactile.unsqueeze(1) 
            x_flex = x_flex.to(torch.float32)
            x_flex = x_flex.unsqueeze(1)
            label[1] = label[1].to(torch.float32)
            label[2] = label[2].to(torch.float32)
            label[4] = label[4].to(torch.float32)
            # compute output
            outputs = model(x_visual,x_tactile,x_flex,label)  
            
            (grasping_pred, 
            upsample_flow, 
            loss_total, 
            avg_list,
            alpha_list,
            temp_metric_data_list) = loss_calc(
                        (targets, label), 
                        outputs, 
                        use_cuda, 
                        testloader.batch_size, 
                        deterministic,
                        avg_loss_grasping,
                        avg_corr_loss,
                        avg_loss_total_grasping, 
                        avg_acc_grasping, 
                        avg_kl, 
                        avg_flow_loss, 
                        avg_tac_loss,
                        avg_ssim,
                        avg_psnr, 
                        avg_mse,
                        avg_l1_loss,
                        avg_gdl_loss,
                        training_phase=training_phase,
                        epoch=epoch,
                        train_visula_last_frame=x_visual
                    )
            
            [avg_loss_grasping, avg_acc_grasping, avg_loss_total_grasping, avg_kl, avg_flow_loss, 
                avg_l1_loss, avg_gdl_loss, avg_ssim, avg_psnr, avg_mse, avg_tac_loss, avg_corr_loss] = avg_list
            
            avg_list_avg = [avg_loss_grasping.avg, avg_acc_grasping.avg, avg_loss_total_grasping.avg, 
                            avg_ssim.avg, avg_psnr.avg, avg_mse.avg, avg_tac_loss.avg, avg_corr_loss.avg, avg_flow_loss.avg, avg_l1_loss.avg, avg_gdl_loss.avg,]
            
            ssim_frame_list.append(temp_metric_data_list[0])
            psnr_frame_list.append(temp_metric_data_list[1])
            mse_frame_list .append(temp_metric_data_list[2])
            pred = grasping_pred.cpu().numpy() #将tensor变量转化为numpy类型        
            Y_pred=pred.tolist() #将numpy类型转化为list类型
            y_pred_list.append(Y_pred)        
            #strNums=[str(Y_pred_i) for Y_pred_i in Y_pred]
            strNums=[str(Y_pred_i) for Y_pred_i in y_pred_list] #将list转化为string类型
            Ypred=",".join(strNums)
            

            target = targets.cpu().numpy()
            Targets=target.tolist() 
            y_targets_list.append(Targets)
            strNums=[str(Targets_i) for Targets_i in y_targets_list]
            Ytargets=",".join(strNums)
                  
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            last_label = label
            last_target = targets
            last_results_vis_predict = upsample_flow

            #让ssim曲线更平滑
            if filt_ssim_flag:
                if temp_ssim <= avg_ssim.avg :
                    temp_ssim = avg_ssim.avg
                else:
                    avg_ssim.avg = temp_ssim
            # plot progress | PSNR: {psnr: .4f} | PSNR(input): {psnr_in: .4f}
            bar.suffix  = '({batch}/{size}) Data:{data:.3f}s|Batch: {bt:.3f}s|Total: {total:}|ETA:{eta:}|'\
                'LossTotal:{loss_total:.4f}|LossGrasping:{losses_grasping:.4f}|ACC(input):{acc_grasping:.4f}|'\
                'Tac_loss:{tactile_loss:.4f}|Corr_loss:{correlation_loss:.4f}|KL:{kl:.4f}|SSIM:{ssim:.4f}|PSNR:{psnr:.4f}'.format(
                        batch=batch_idx + 1,
                        size=len(testloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss_total = avg_loss_total_grasping.avg, 
                        losses_grasping = avg_loss_grasping.avg,
                        acc_grasping = avg_acc_grasping.avg,
                        tactile_loss = avg_tac_loss.avg,
                        correlation_loss = avg_corr_loss.avg,
                        kl = avg_kl.avg,
                        ssim = avg_ssim.avg,
                        psnr = avg_psnr.avg,
                        # psnr_in=psnr_input.avg
                        )
            bar.next()
        bar.finish()
     #for循环后再保存结果   
     #将要输出保存的文件地址，若文件不存在，则会自动创建
        if not os.path.exists('./fu_Experimental_results/'):
            os.mkdir('./fu_Experimental_results')
            
        fileName='./fu_Experimental_results/'+ 'C3D' + '-' +'y_pred.txt'
        fw = open(fileName, 'w') 
    #这里平时print("test")换成下面这行，就可以输出到文本中了
    #file_handle=open('1.txt',mode='w')#w 写入模式   
    #将str类型数据存入本地文件1.txt中
        fw.write(Ypred)               
    # 换行
        fw.write("\n") 
        fw.close
        
        filename='./fu_Experimental_results/'+ 'C3D' + '-' +'y_targets.txt'
        FW = open(filename, 'w') 
        FW.write(Ytargets)
        FW.write("\n") 
        FW.close
        return  (
                last_label, 
                last_results_vis_predict,
                alpha_list,
                avg_list_avg,
                outputs, 
                last_target,
                pred,
                (ssim_frame_list, psnr_frame_list, mse_frame_list)
                )


def adjust_learning_rate(optimizer, epoch, opt):
        if epoch % OPT.schedule ==0 and epoch !=0 :
            OPT.lr *= OPT.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = OPT.lr


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    filepath = os.path.join(checkpoint, filename)

    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def tsne_compute(pred, formatted_datetime, label=None, is_show_plt=False):
    # if label == None:
    #     label = np.zeros_like(pred)
    # # 合并真实标签和预测概率
    # combined_data = np.column_stack((label, pred))

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, perplexity=2, n_iter=3000, random_state=0)
    low_dimension_data = tsne.fit_transform(pred)
    true_labels = label

    if is_show_plt:
        # 绘制 t-SNE 可视化结果
        plt.figure()
        scatter = plt.scatter(low_dimension_data[:, 0], low_dimension_data[:, 1], c=true_labels, cmap='coolwarm')
        plt.title('t-SNE Visualization')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.colorbar(scatter, label='Labels')
        plt.savefig(os.path.join(f'./checkpoint_sgd/results/{formatted_datetime}/tsne'))
        
    return low_dimension_data, true_labels
    
def auc_compute(pred, label, formatted_datetime, is_show_plt=True):    
    
    # 合并真实标签和预测概率
    combined_data = np.column_stack((label, pred))
    # 计算 AUC
    true_labels = combined_data[:, 0]
    predicted_probabilities = combined_data[:, 1]
    auc = roc_auc_score(true_labels, predicted_probabilities)
    fpr, tpr, thresholds = roc_curve(true_labels, predicted_probabilities)
    
    if is_show_plt:
        # 绘制 ROC 曲线
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve (AUC={auc:.2f})')
        plt.savefig(os.path.join(f'./checkpoint_sgd/results/{formatted_datetime}/auc'))
    
    return fpr, tpr, auc, thresholds


if __name__ == "__main__":
        main()

