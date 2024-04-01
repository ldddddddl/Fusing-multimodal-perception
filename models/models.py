# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import numpy
from utils.models_utils import (duplicate,gaussian_parameters,product_of_experts,sample_gaussian)#device
from models.encoders import FlexEncoder, TactileEncoder, ImageEncoder, frame_to_stream
from models.decoders import FutureFrameDecoder, TactileDecoder, Generate
from utils.misc import Obj
class C3D(nn.Module):
     def __init__(
         self,drop_p_v=0.2, 
         visual_dim=4096, 
         fc_hidden_1=128,
         fc_hidden_t = 20,
         fc_hidden_f = 20,
         num_classes=2,
         num_img = 12,
         batch_size=8,
         deterministic=True,
         z_dim = 128,
         device = 'cuda',
         is_ablation='fully_data' ,
         fusion_flag='Multimodal_fusion_model',
         encoder_mode='convlstm',
         is_use_cross_conv=True,
         is_use_poe = False,
         slide_window_size=8,
                  ):
        super(C3D, self).__init__()
        '''
        is_ablation{
            null_flex_tactile:直接去除flex, tactile 的数据
            null_flex:直接去除flex的数据
            null_tactile:直接去除tactile的数据
            zero_flex_tactile:将flex, tactile的结果替换为零向量
            zero_flex, zero_tactile同理
            fully_data: 包含所有的数据特征
        }
        
        '''

        '''
        抓取稳定性判断模型
        '''
        self.visual_c3d=C3D_visual_only(visual_dim=visual_dim, drop_p_v=drop_p_v, fc_hidden_1=fc_hidden_1,num_classes=2)
        self.tactile_c1d=C1D_tactile_test1()
        self.flex_c1d=C1D_flex()
        '''
        生成未来帧模型
        '''
        self.z_dim = z_dim
        self.ablation = is_ablation
        self.fusion_flag = fusion_flag
        self.num_img = num_img
        self.batch_size = batch_size
        self.encoder_mode = encoder_mode
        self.is_use_cross_conv = is_use_cross_conv
        self.is_use_poe = is_use_poe
        self.deterministic = deterministic
        self.slide_window_size = slide_window_size
        self.device = device
        self.flex_encoder = FlexEncoder(z_dim=self.z_dim, num_img=self.num_img)
        self.tactile_encoder = TactileEncoder(z_dim=self.z_dim, num_img=self.num_img)
        self.img_encoder = ImageEncoder(z_dim=self.z_dim, model_flag=self.encoder_mode, num_img=self.num_img)
        self.future_frame_decoder = FutureFrameDecoder(z_dim=self.z_dim)
        self.modal_fusion_model = MultiModalFusionModel(num_modalities=3, z_dim=self.z_dim, is_use_cross_conv=self.is_use_cross_conv)
        self.tac_decoder = TactileDecoder(z_dim=self.z_dim, num_img=self.num_img)
        self.generate_model = Generate(self.num_img, self.z_dim, self.batch_size, device=self.device, data_mode=self.ablation)
        self.dropout = nn.Dropout(drop_p_v)
        self.batch_nor = nn.BatchNorm1d(3)
        if not self.deterministic:
            self.grasping_fc1 = nn.Linear(fc_hidden_1+fc_hidden_t+fc_hidden_f, fc_hidden_1)
        else:
            self.grasping_fc1 = nn.Linear(2560, 1364)
            self.gen_fusion_fc1 = nn.Linear(3 * self.z_dim, int(1.5 * self.z_dim))
        if self.num_img == 8:
            self.fc2 = nn.Linear(916, num_classes) 
            self.fc3 = nn.Linear(256, num_classes)
            
        elif self.num_img == 12:
            self.fc2 = nn.Linear(1364, num_classes) 
            self.fc3 = nn.Linear(self.z_dim * 2, num_classes)
        # self.fc2 = nn.Linear(8,num_classes)
        self.drop_p = drop_p_v
        self.grasping_mu_z = None
        self.grasping_var_z = None
        self.gen_mu_z = None
        self.gen_var_z = None
        self.gen_img_out_convs = None
        self.encoded_data = None
        self.mu_prior_resized_encoded = None
        self.var_prior_resized_encoded = None
        self.z_prior_m_grasping = torch.nn.Parameter(
            torch.zeros(1, 84), requires_grad=False
        )
        self.z_prior_v_grasping = torch.nn.Parameter(
            torch.ones(1, 84), requires_grad=False
        )
        self.z_prior_grasping = (self.z_prior_m_grasping, self.z_prior_v_grasping)
        
        self.z_prior_m_encoded = torch.nn.Parameter(
            torch.zeros(1, self.z_dim//2), requires_grad=True
        )
        self.z_prior_v_encoded = torch.nn.Parameter(
            torch.ones(1, self.z_dim//2), requires_grad=True
        )
        self.z_prior_encoded = (self.z_prior_m_encoded, self.z_prior_v_encoded)
                

     def forward(self, x_3d_v, x_1d_t, x_1d_f, label, trainning_phase='full_training'):
        vis_label, tac_label, flex_label, grasp_label, _ = label
        #----------------grasping--------------------------------------------
        if self.is_use_poe:
            '''
            抓取部分编码器结果
            '''
            x_v = self.visual_c3d(x_3d_v)   #[8,128]
            x_t = self.tactile_c1d(x_1d_t)#[8,20]
            x_f = self.flex_c1d(x_1d_f)   #[8,20]
            x_v = x_v.view(8, -1)#[16, 512]
            x_t = x_t.view(8, -1)#[16, 1472]
            x_f = x_f.view(8, -1)#[16, 1472]

            '''
            ablation study
            '''
            if self.ablation == 'zero_flex_tactile':
                x_f = torch.zeros_like(x_f)
                x_t = torch.zeros_like(x_t)
            elif self.ablation == 'zero_flex':
                x_f = torch.zeros_like(x_f)
            elif self.ablation == 'zero_tactile':
                x_t = torch.zeros_like(x_t)
                
                
            if self.deterministic:
                x=torch.cat((x_v,x_t,x_f),-1) 
                x=F.relu(self.grasping_fc1(x))
                grasping_result = F.dropout(x, p=self.drop_p, training=self.training)
                # grasping_result = self.fc2(x)

            else:
                # prior
                mu_prior_grasping, var_prior_grasping = self.z_prior_grasping #[1,84],[1,84]
                
                mu_prior_resized_grasping = duplicate(mu_prior_grasping, x_v.shape[0])#[16,]
                var_prior_resized_grasping = duplicate(var_prior_grasping, x_v.shape[0])
                '''
                高斯分布均值和方差
                '''
                mu_z_vis_grasping, var_z_vis_grasping = gaussian_parameters(x_v, dim=1) #[16,]
                mu_z_tac_grasping, var_z_tac_grasping = gaussian_parameters(x_t, dim=1) #[16,]
                mu_z_flex_grasping, var_z_flex_grasping = gaussian_parameters(x_f, dim=1) #[16,]
                #cat grasping data          
                m_vect_grasping = torch.cat(
                    [mu_z_vis_grasping, mu_z_tac_grasping, mu_z_flex_grasping, mu_prior_resized_grasping], 
                    dim=-1
                    )#2   [8,168]
                var_vect_grasping = torch.cat(
                    [mu_z_vis_grasping, mu_z_tac_grasping, mu_z_flex_grasping, var_prior_resized_grasping],
                    dim=-1
                    )#2  
                # Fuse modalities mean / variances using product of experts
                mu_z_grasping, var_z_grasping = product_of_experts(m_vect_grasping, var_vect_grasping, 'grasping')#[8,168]
                self.grasping_mu_z, self.grasping_var_z = (mu_z_grasping, var_z_grasping)
                grasping_result = sample_gaussian(mu_z_grasping, var_z_grasping, self.device)#[16, 948]
            grasping_fused = self.fc2(grasping_result)#[8,2]    
        #---------------------------------------------------------------------------------------------------
            
        #-------------------------------generate------------------------------------------------------------    
        '''
        生成模型结果
        '''
        img_encoded, img_out_convs = self.img_encoder(torch.cat([x_3d_v, label[0]], dim=2))
        flex_encoded, flex_layers_results = self.flex_encoder(torch.cat([x_1d_f, label[2][:, :, :-3]], dim=-1))#torch.Size([8, 256, 1])
        tactile_encoded, tac_layers_results = self.tactile_encoder(torch.cat([x_1d_t, label[1][:, :, :-3]], dim=-1))#torch.Size([8, 256, 1])
        self.gen_img_out_convs = img_out_convs
        if self.deterministic:
            modal_data = [torch.transpose(img_encoded, 1, 2), torch.transpose(flex_encoded, 1, 2), torch.transpose(tactile_encoded, 1, 2)]
            # multimodal fusion model
            fusion_data = self.modal_fusion_model(modal_data).unsqueeze(-1)
            fusion_data = torch.cat([fusion_data, fusion_data], dim=-1)
            encoded_sample_result = fusion_data.view(self.batch_size, self.z_dim // 2, -1)[:, :, :img_out_convs[0].shape[2] // 2]
            self.encoded_data = [img_encoded.squeeze(-1), flex_encoded.squeeze(-1), tactile_encoded.squeeze(-1)]
            
        else:
            # Encoder priors
            mu_prior_encoded, var_prior_encoded = self.z_prior_encoded #[1,84],[1,84]
            '''
            ablation study
            '''
            if self.ablation == 'zero_flex_tactile':
                flex_encoded = torch.zeros_like(flex_encoded)
                tactile_encoded = torch.zeros_like(tactile_encoded)
                label[1] = torch.zeros_like(label[1])
            elif self.ablation == 'zero_flex':
                flex_encoded = torch.zeros_like(flex_encoded)
            elif self.ablation == 'zero_tactile':
                tactile_encoded = torch.zeros_like(tactile_encoded)
                label[1] = torch.zeros_like(label[1])
            self.encoded_data = [img_encoded.squeeze(-1), flex_encoded.squeeze(-1), tactile_encoded.squeeze(-1)]

            '''
            生成部分
            '''
            mu_prior_resized_encoded = duplicate(mu_prior_encoded, img_encoded.shape[0]).unsqueeze(2)
            var_prior_resized_encoded = duplicate(var_prior_encoded,img_encoded.shape[0]).unsqueeze(2)
            self.mu_prior_resized_encoded, self.var_prior_resized_encoded = (mu_prior_resized_encoded, var_prior_resized_encoded)
            if self.fusion_flag == 'POE':
                mu_z_img_encoded, var_z_img_encoded = gaussian_parameters(img_encoded, dim=1)
                mu_z_flex_encoded, var_z_flex_encoded = gaussian_parameters(flex_encoded, dim=1)
                mu_z_tactile_encoded, var_z_tactile_encoded= gaussian_parameters(tactile_encoded, dim=1)
                fusion_flag = ['full_data', 'zero_flex_tactile', 'zero_flex', 'zero_tactile']
                if self.ablation in fusion_flag:
                    '''
                    在dim=-1（第三维度）上进行拼接，tensor形状为[8, 128]
                    '''
                    m_vect_encoded = torch.cat(
                        [mu_z_img_encoded, mu_z_flex_encoded, mu_z_tactile_encoded, mu_prior_resized_encoded],
                        dim=-1
                        )#[8, 276]
                    var_vect_encoded = torch.cat(
                        [var_z_img_encoded, var_z_flex_encoded, var_z_tactile_encoded, var_prior_resized_encoded],
                        dim=-1
                        )#[8, 276]
                elif self.ablation == 'null_flex_tactile':
                    m_vect_encoded = [mu_z_img_encoded, mu_prior_resized_encoded]
                    var_vect_encoded = [var_z_img_encoded, var_prior_resized_encoded]
                elif self.ablation == 'null_flex':
                    m_vect_encoded = torch.cat([mu_z_img_encoded, mu_z_tactile_encoded, mu_prior_resized_encoded], dim=-1)
                    var_vect_encoded = torch.cat([var_z_img_encoded, var_z_tactile_encoded, var_prior_resized_encoded], dim=-1)  
                elif self.ablation == 'null_tactile':
                    m_vect_encoded = torch.cat([mu_z_img_encoded, mu_z_flex_encoded, mu_prior_resized_encoded], dim=-1)
                    var_vect_encoded  = torch.cat([var_z_img_encoded, var_z_flex_encoded, var_prior_resized_encoded], dim=-1)
                # POE
                mu_z_encoded, var_z_encoded = product_of_experts(m_vect_encoded, var_vect_encoded, 'generate')#[8, 128, 4]
            elif self.fusion_flag == 'Multimodal_fusion_model':
                modal_data = [torch.transpose(img_encoded, 1, 2), torch.transpose(flex_encoded, 1, 2), torch.transpose(tactile_encoded, 1, 2)]
                if self.is_use_cross_conv:
                    # multimodal fusion model
                    fusion_data = self.modal_fusion_model(modal_data)
                else:
                    concat_data = torch.cat(modal_data, dim=1)
                    concat_data = self.batch_nor(concat_data)
                    fusion_data = torch.sum(concat_data, dim=1)
                mu_z_encoded, var_z_encoded = gaussian_parameters(fusion_data, dim=1)           
            self.gen_mu_z, self.gen_var_z = (mu_z_encoded, var_z_encoded)
            # Sample Gaussian to get latent
            encoded_sample_result = sample_gaussian(mu_z_encoded.unsqueeze(-1).expand(-1, -1, img_out_convs[0].size(2) // 2), var_z_encoded.unsqueeze(-1).expand(-1, -1, img_out_convs[0].size(2) // 2), self.device, training_phase=trainning_phase)

        '''
        未来图像生成
        '''
        next_frame_list = []
        last_frame = None
        for seq in range(self.num_img - self.slide_window_size):
            if last_frame is None:
                last_frame = x_3d_v[:, :, -1, :, :]
                # error from pred and target
            error_pred_label = F.sigmoid(last_frame - label[0][:, :, seq, :, :])  
            error_label_pred = F.sigmoid(label[0][:, :, seq, :, :] - last_frame)
            error_all = torch.cat([error_label_pred, error_pred_label], dim=1)
            ##label[1][:, :, seq*3:seq*3+3] 可以考虑tac数据切片输入，但可能会出现数据不对齐的情况
            next_frame = self.generate_model(last_frame, label[1][:, :, seq * 3:(seq + 1) * 3], grasp_label, encoded_sample_result, error_all=error_all)
            next_frame_list.append(next_frame)
            last_frame = next_frame
            encoded_sample_result = sample_gaussian(mu_z_encoded.unsqueeze(-1).expand(-1, -1, img_out_convs[0].size(2) // 2), var_z_encoded.unsqueeze(-1).expand(-1, -1, img_out_convs[0].size(2) // 2), self.device, training_phase=trainning_phase)
        next_stream = frame_to_stream(next_frame_list)
        '''
        图片预测触觉信息
        '''
        tactile_predict_stream = self.tac_decoder(encoded_sample_result)
        
        '''
        抓取稳定性判断
        '''
        if not self.is_use_poe:
            out_dropout = self.dropout(encoded_sample_result)
            grasping_fused = self.fc3(out_dropout.contiguous().view(self.batch_size, -1))
        #---------------------------------------------------------------------------------------------------
                
        if self.deterministic:
            return Obj({"grasping_fused":grasping_fused,
                        "sample_result":encoded_sample_result, 
                        "encoded_data":self.encoded_data,
                        "next_stream":next_stream, 
                        "tactile_stream":tactile_predict_stream,
                        "img_out_convs":self.gen_img_out_convs,
                    })
        elif self.is_use_poe:
            return Obj({"grasping_fused":grasping_fused, 
                    "grasping_mu_z":self.grasping_mu_z,
                    "grasping_var_z":self.grasping_var_z,
                    "sample_result":encoded_sample_result, 
                    "encoded_data":self.encoded_data,
                    "next_stream":next_stream, 
                    "mu_z_encoded":self.gen_mu_z, 
                    "var_z_encoded":self.gen_var_z, 
                    "mu_prior_encoded":self.mu_prior_resized_encoded.squeeze(-1), 
                    "var_prior_encoded":self.var_prior_resized_encoded.squeeze(-1),
                    "tactile_stream":tactile_predict_stream,
                    "img_out_convs":self.gen_img_out_convs,
                    "grasping_result":grasping_result,
                })
        else:
            return Obj({"grasping_fused":grasping_fused, 
                    "grasping_mu_z":self.grasping_mu_z,
                    "grasping_var_z":self.grasping_var_z,
                    "sample_result":encoded_sample_result, 
                    "encoded_data":self.encoded_data,
                    "next_stream":next_stream, 
                    "mu_z_encoded":self.gen_mu_z, 
                    "var_z_encoded":self.gen_var_z, 
                    "mu_prior_encoded":self.mu_prior_resized_encoded.squeeze(-1), 
                    "var_prior_encoded":self.var_prior_resized_encoded.squeeze(-1),
                    "tactile_stream":tactile_predict_stream,
                    "img_out_convs":self.gen_img_out_convs,
                    "grasping_result":encoded_sample_result.view(self.batch_size, -1),
            })


            
                
class C3D_visual(nn.Module):
    """
    The C3D network.
    """
    def __init__(self, pretrained=False,length=6, img_size=112):
        super(C3D_visual, self).__init__()
        self.img_size=img_size
        
        if length == 6:
            if img_size == 112:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1,2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
                #self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(1,0,0))
                #self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0,0,0))
                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
                

        self.dropout = nn.Dropout(p=0.5)  #正则化
        #
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):

        if self.img_size ==112:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }
        Path = ''
        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):           
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()




class CrossConvolution(nn.Module):
    def __init__(self, z_dim=128, in_channels=3, out_channels=1):
        super(CrossConvolution, self).__init__()
        # self.img_tac_conv = nn.Conv1d(in_channels - 1, out_channels, kernel_size=3, padding=1)
        # self.flex_conv = nn.Conv1d(in_channels - 1, out_channels, kernel_size=3, padding=1)
        self.cross_conv3d = nn.Conv3d(3, 3, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, multimodal_dat):
        return self.leakyrelu(self.cross_conv3d(multimodal_dat))

class MultiModalAttention(nn.Module):
    def __init__(self, z_dim=128, num_modalities=3):
        super(MultiModalAttention, self).__init__()
        self.num_modalities = num_modalities
        self.z_dim = z_dim
        self.fc = nn.Linear(self.z_dim, 1)

    def forward(self, modalities):
        # Compute attention weights
        attention_weights = F.softmax(self.fc(modalities), dim=1)
        
        attended_modalities = torch.mul(attention_weights, modalities)
        
        return attended_modalities

class MultiModalFusionModel(nn.Module):
    def __init__(self, num_modalities=3, z_dim=128, is_use_cross_conv=True):
        super(MultiModalFusionModel, self).__init__()
        self.num_modalities = num_modalities
        self.z_dim = z_dim
        self.is_use_cross_conv = is_use_cross_conv
        # Define cross convolution layers for each modality
        self.fusion_model = CrossConvolution(in_channels=3, out_channels=1, z_dim=self.z_dim)
        # Define multi-modal attention
        self.attention = MultiModalAttention(num_modalities=num_modalities, z_dim=self.z_dim)
        self.cross_conv_fusion = nn.Conv1d(self.z_dim, self.z_dim, kernel_size=3, padding=1, stride=1)
        self.normal_modal = nn.BatchNorm1d(num_features=3)
        self.leakyrelu = nn.LeakyReLU(0.1, inplace=True)
        self.flatten = nn.Flatten()
        # Fully connected layers for classification
        self.fc1 = nn.Linear(self.z_dim, self.z_dim)

    def forward(self, modalities):
        '''
        arg:
        modaltities:[batch, modal_num, n*z_dim]
        '''

        modality_outputs = torch.cat(modalities, dim=1)
        batch_size, num_modal, num_features = modality_outputs.size()
        # img/tac与flex的数据差别较大。先做normal，保证后续权重可以快速学习到
        normal_result = self.normal_modal(modality_outputs)
        # Apply multi-modal attention
        attention_result = self.attention(normal_result)
        sum_result = torch.add(normal_result, attention_result)
        if self.is_use_cross_conv:
            fused_result = self.fusion_model(sum_result.view(batch_size, num_modal, num_features, 1, 1))
        else:
            fused_result = sum_result
        sum_fused_result = torch.sum(fused_result, dim=1)
        flatten_result = self.flatten(sum_fused_result)
        fc_out = self.fc1(flatten_result)

        return fc_out


        
class C3D_visual_only(nn.Module):
    def __init__(self, drop_p_v=0.2, visual_dim=4096, fc_hidden_1=128, num_classes=2):  # , fc_hidden2=128, num_classes=50):
        super(C3D_visual_only, self).__init__()      
        self.visual_c3d = C3D_visual(pretrained=False,length=6)
        self.fc1 = nn.Linear(visual_dim, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v):
        
        x_v = self.visual_c3d(x_3d_v) #  x_v.shape的结果是torch.Size([2, 4096])
        x = F.relu(self.fc1(x_v))
        # x = F.relu(self.fc2(x))
        return x  
    

#针对的是触觉长度为 1
class  C1D_tactile_test1(nn.Module):
    def __init__(self):
        super( C1D_tactile_test1, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(16, 32, 1, 1)
        self.max_pool2 = nn.MaxPool1d(1, 1)
        self.conv3 = nn.Conv1d(32, 64, 1, 1)
        self.max_pool3 = nn.MaxPool1d(3, 1)
        
        self.linear1 = nn.Linear(64, 32)
        # self.linear2 = nn.Linear(32, 16)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)#[8, 40, 46]
        
        x = x.view(-1, 64)#[368, 40]
        # x = x.view(32, -1)
        x = F.relu(self.linear1(x))#[368, 20]
        # x = F. relu(self.linear2(x))
        return x    
     
class  C1D_flex(nn.Module):
    def __init__(self):
        super( C1D_flex, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=1, stride=1)
        self.max_pool1 = nn.MaxPool1d(kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(16, 32, 1, 1)
        self.max_pool2 = nn.MaxPool1d(1, 1)
        self.conv3 = nn.Conv1d(32, 64, 1, 1)
        self.max_pool3 = nn.MaxPool1d(3, 1)
        
        self.linear1 = nn.Linear(64, 32)
        # self.linear2 = nn.Linear(32, 16)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.max_pool1(x)
        x = F.relu(self.conv2(x))
        x = self.max_pool2(x)
        x = F.relu(self.conv3(x))
        x = self.max_pool3(x)
        
        x = x.view(-1, 64)
        # x = x.view(32, -1)
        
        x = F.relu(self.linear1(x))
        # x = F. relu(self.linear2(x))
        
        return x            


from matplotlib import pyplot as plt
def img_show(*images):
    """
    显示多个RGB张量的图片

    参数:
    *images: 任意数量的RGB张量
    """
    num_images = len(images)

    # 设置子图的行和列
    rows = 1
    cols = num_images

    # 创建一个新的图形
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))

    # 如果只有一张图片，将axes转换为一个包含单个元素的列表
    if num_images == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        # 将RGB张量的值限制在0到1之间
        image_data = np.clip(images[i].permute(1, 2, 0).detach().cpu().numpy(), 0, 1)

        # 显示图片
        ax.imshow(image_data)
        ax.axis('off')

    plt.show()