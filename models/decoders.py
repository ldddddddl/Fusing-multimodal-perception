# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 10:03:37 2022

@author: admin
"""
from .encoders import reshape_tensor, ConvLSTM
import torch
import torch.nn as nn
from models.models_utils import init_weights
from models.layers2 import (
    conv3d,
    predict_flow,
    deconv,
    crop_like,
    conv2d,
    tac_deconv,
    tac_predict 
)
from utils.models_utils import duplicate
from torch.nn import functional as F
from utils.misc import norm_to_rgb
from matplotlib import pyplot as plt
import numpy as np

class FutureFrameDecoder(nn.Module):
    def __init__(self, z_dim=128, initailize_weights=True, use_skip_layers=4, num_masks=8):
        """
        Decodes the future frame
        """
        super(FutureFrameDecoder, self).__init__()
        self.z_dim = z_dim
        self.use_skip_layers = use_skip_layers
        self.num_masks = num_masks
        self.optical_flow_conv = conv2d(128, 256, kernel_size=3, stride=1)

        self.img_deconv6 = deconv(256, 128)
        self.img_deconv5 = deconv(128, 64)
        self.img_deconv4 = deconv(195, 64)
        self.img_deconv3 = deconv(195, 64)
        self.img_deconv2 = deconv(195, 32)

        self.predict_optical_flow6 = predict_flow(128)
        self.predict_optical_flow5 = predict_flow(195)
        self.predict_optical_flow4 = predict_flow(195)
        self.predict_optical_flow3 = predict_flow(195)
        self.predict_optical_flow2 = predict_flow(99)

        self.upsampled_optical_flow6_to_5 = nn.ConvTranspose2d(
            in_channels=3, 
            out_channels=3, 
            kernel_size=4, 
            stride=2, 
            padding=1, 
            bias=False,
        )
        self.upsampled_optical_flow5_to_4 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow4_to_3 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False
        )
        self.upsampled_optical_flow3_to_2 = nn.ConvTranspose2d(
            3, 3, 4, 2, 1, bias=False
        )
        self.upsample_conv = nn.ConvTranspose2d(
            99, 99, 4, 2, 1, bias=False
        )

        self.predict_optical_flow2_masks = nn.Conv2d(
            in_channels=99, 
            out_channels=9, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        self.predict_optical_flow2_mask = nn.Conv2d(
            in_channels=99, 
            out_channels=3, 
            kernel_size=3, 
            stride=1, 
            padding=1, 
            bias=False
        )
        
        #无跳连接
        self.deconv_out1 = deconv(128, 64)
        self.deconv_out2 = deconv(64, 32)
        self.deconv_out3 = deconv(32, 16)
        self.conv_pred = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_mask = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_masks = nn.Conv2d(16, 9, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.softmax = nn.Softmax(dim=1)
        if initailize_weights:
            init_weights(self.modules())

    def forward(self, tiled_feat, img_out_convs):
        """
        Predicts the future images.
        Args:
            tiled_feat:last frame
            img_out_convs: outputs of the image encoders (skip connections)
        """
        out_img_conv2, out_img_conv3, out_img_conv4, out_img_conv5, out_img_conv6 = (
            img_out_convs
        )
        '''
        改变z的形状，使高斯采样后的张量的维度与卷积层中流通的数据相同
        '''
        ### reshape
        if self.use_skip_layers != 0:

            optical_flow_in_f2 = self.optical_flow_conv(tiled_feat)
            optical_flow_in_feat = self.img_deconv6(optical_flow_in_f2)
        if self.use_skip_layers == 4:
            # predict optical flow pyramids
            # skip connection 1
            optical_flow6 = self.predict_optical_flow6(optical_flow_in_feat)
            optical_flow6_up = crop_like(
                self.upsampled_optical_flow6_to_5(optical_flow6), out_img_conv5
            )
            out_img_deconv5 = crop_like(
                self.img_deconv5(optical_flow_in_feat), out_img_conv5
            )
            concat5 = torch.cat((out_img_conv5, out_img_deconv5, optical_flow6_up), 1)
            
            # skip connection 2
            optical_flow5 = self.predict_optical_flow5(concat5)
            optical_flow5_up = crop_like(
                self.upsampled_optical_flow5_to_4(optical_flow5), out_img_conv4
            )
            out_img_deconv4 = crop_like(self.img_deconv4(concat5), out_img_conv4)
            concat4 = torch.cat((out_img_conv4, out_img_deconv4, optical_flow5_up), 1)
            
            # skip connection 3
            optical_flow4 = self.predict_optical_flow4(concat4)
            optical_flow4_up = crop_like(
                self.upsampled_optical_flow4_to_3(optical_flow4), out_img_conv3
            )
            out_img_deconv3 = crop_like(self.img_deconv3(concat4), out_img_conv3)
            concat3 = torch.cat((out_img_conv3, out_img_deconv3, optical_flow4_up), 1)
            
            # skip connection 4
            optical_flow3 = self.predict_optical_flow3(concat3)
            optical_flow3_up = self.upsampled_optical_flow3_to_2(optical_flow3)
            out_img_deconv2 = self.img_deconv2(concat3)
            concat2 = torch.cat((out_img_conv2, out_img_deconv2, optical_flow3_up), dim=1)
        elif self.use_skip_layers == 2:
            # skip connection 1
            optical_flow4 = self.predict_optical_flow6(optical_flow_in_feat)
            optical_flow4_up = self.upsampled_optical_flow4_to_3(optical_flow4)
            out_img_deconv3 = self.img_deconv5(optical_flow_in_feat)
            concat3 = torch.cat((out_img_conv4, out_img_deconv3, optical_flow4_up), 1)
            
            # skip connection 2
            optical_flow3 = self.predict_optical_flow3(concat3)
            optical_flow3_up = self.upsampled_optical_flow3_to_2(optical_flow3)
            out_img_deconv2 = self.img_deconv2(concat3)
            concat2 = torch.cat((out_img_conv2, out_img_deconv2, optical_flow3_up), dim=1)
            concat2 = self.upsample_conv(concat2)
        elif self.use_skip_layers == 0:
            out_deconv1 = self.deconv_out1(tiled_feat)
            out_deconv2 = self.deconv_out2(out_deconv1)
            out_deconv3 = self.deconv_out3(out_deconv2)

            future_frame_unmasked = self.conv_pred(out_deconv3)
            future_frame_mask = self.conv_mask(out_deconv3)
            future_frame_masks = self.softmax(self.conv_masks(out_deconv3))
            
        if self.use_skip_layers != 0:
            future_frame_unmasked = self.predict_optical_flow2(concat2)
            future_frame_mask = self.predict_optical_flow2_mask(concat2)
            future_frame_masks = self.predict_optical_flow2_masks(concat2)
            future_frame_masks = self.softmax(future_frame_masks)
        
        future_frame = future_frame_unmasked * torch.sigmoid(future_frame_mask)

        # return future_frame, future_frame_mask
        return future_frame_masks, future_frame
           

class TactileDecoder(nn.Module):
    def __init__(self, z_dim=1024, num_img=12, initailize_weights=True):
        super(TactileDecoder, self).__init__()

        self.num_img = num_img
        if self.num_img == 12:
            self.tac_decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(2 * z_dim), int(1.5 * z_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(int(1.5 * z_dim), int(1.25 * z_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(int(1.25 * z_dim),  z_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(z_dim, int(0.75 * z_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(int(0.75 * z_dim), 3*num_img), 
            )
        elif self.num_img == 8:
            self.tac_decoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(int(2 * z_dim), int(1.5 * z_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.4),
                nn.Linear(int(1.5 * z_dim), int(1.25 * z_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.2),
                nn.Linear(int(1.25 * z_dim), z_dim),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Dropout(0.4),
                nn.Linear(z_dim, int(0.75 * z_dim)),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(int(0.75 * z_dim), 3*num_img), 
            )
        if initailize_weights:
            init_weights(self.modules())
            
    def forward(self, img_encoded, tac_layers_results=None):

        return self.tac_decoder(img_encoded)
    
    
    
class Generate(nn.Module):
    def __init__(self, num_img, z_dim, batch_size, device, is_use_cdna=True, num_masks=8, data_mode='full_data'):
        super(Generate, self).__init__()
        self.batch_size = batch_size
        self.z_dim = z_dim
        self.num_img = num_img
        conv_error_stride = 7
        conv_error_out_channel = 3
        self.device = device
        self.is_use_cdna = is_use_cdna
        self.ablation = data_mode
        self.conv2d_1 = conv2d(9, 64, 3)
        self.convlstm1 = ConvLSTM(input_channels=64, hidden_channels=64, kernel_size=3, stride=2, first_flag=True, input_mode='frame', size_flag='up')
        self.convlstm2 = ConvLSTM(input_channels=64, hidden_channels=128, kernel_size=3, stride=2, size_flag='up', input_mode='frame')
        self.convlstm3 = ConvLSTM(input_channels=128, hidden_channels=128, kernel_size=3, input_mode='frame')
        self.convlstm4 = ConvLSTM(input_channels=128, hidden_channels=256, kernel_size=3, stride=2, size_flag='up', input_mode='frame')
        self.convlstm5 = ConvLSTM(input_channels=256+self.z_dim, hidden_channels=512, kernel_size=3, stride=2, size_flag='up', input_mode='frame')
        self.convlstm6 = ConvLSTM(input_channels=512, hidden_channels=128, kernel_size=3, stride=1, size_flag='up', input_mode='frame')
        self.conv2d_2 = conv2d(256, self.z_dim, 3)
        if self.ablation == 'full_data':
            self.conv2d_3 = conv2d(self.z_dim * self.num_img // 6 + (z_dim // 3) * 3 + (112 // conv_error_stride)**2 * conv_error_out_channel, self.z_dim, 3)
        elif self.ablation in ['zero_flex_tactile', 'zero_flex', 'zero_tactile']:
            self.conv2d_4 = conv2d(self.z_dim * self.num_img // 6 + (z_dim // 3) * 3 + (112 // conv_error_stride)**2 * conv_error_out_channel, self.z_dim, 3)
        self.fusion_conv = conv2d(self.z_dim + 256, 128, 3)
        self.conv_error = conv2d(6, conv_error_out_channel, kernel_size=9, stride=conv_error_stride)
        self.flatten = nn.Flatten()
        self.upsample = nn.UpsamplingNearest2d(64)
        self.batch_norm = nn.BatchNorm2d(128)
        # self.fc1 = nn.Linear(999, 2)
        self.cdna = CDNA(num_masks=num_masks, color_channels=3, kernel_size=[9, 9], z_dim=z_dim)
        self.future_frame_decoder = FutureFrameDecoder(use_skip_layers=2, num_masks=num_masks)
    def forward(self, last_frame, tactile, grasp_state, z_latten, error_all, is_zero_input=False):
        if not is_zero_input:
            inputs = torch.cat([last_frame, error_all], dim=1)
            # inputs = error_all
            
        else:
            inputs = torch.zeros([8, 9, 112, 112], dtype=torch.float32, device=self.device)
        out_conv1 = self.conv2d_1(inputs)
        hidden_state1, cell_state1 = self.convlstm1(out_conv1)
        hidden_state2, cell_state2 = self.convlstm2(hidden_state1, cell_state1)
        hidden_state3, cell_state3 = self.convlstm3(hidden_state2, cell_state2)
        hidden_state4, cell_state4 = self.convlstm4(hidden_state3, cell_state3)
        

        
        conv_error_result = self.conv_error(error_all)
        if self.ablation == 'full_data':
            multil_modal = torch.cat([duplicate(tactile.squeeze(1), self.z_dim // tactile.shape[-1], dim=-1), 
                                    self.flatten(conv_error_result),
                                    self.flatten(z_latten)], dim=-1).view(self.batch_size, -1, 1, 1).expand(-1, -1, 14, 14)
            modal_fused = self.conv2d_3(multil_modal)
        elif self.ablation in ['zero_flex_tactile', 'zero_flex', 'zero_tactile']:
            multil_modal = torch.cat([duplicate(tactile.squeeze(1), self.z_dim // tactile.shape[-1], dim=-1), 
                        self.flatten(conv_error_result),
                        self.flatten(z_latten)], dim=-1).view(self.batch_size, -1, 1, 1).expand(-1, -1, 14, 14)
            modal_fused = self.conv2d_4(multil_modal)
        cat_image_fused = torch.cat([hidden_state4, modal_fused], dim=1)
        
        # fused_result = self.fusion_conv(cat_image_fused)
        hidden_state5, cell_state5 = self.convlstm5(cat_image_fused, cell_state4)
        hidden_state6, cell_state6 = self.convlstm6(hidden_state5, cell_state5)
        out_hiddens = (
            hidden_state1,
            hidden_state2,
            hidden_state3,
            hidden_state4,
            hidden_state5,
        )

        out_states = (
            cell_state1,
            cell_state2,
            cell_state3,
            cell_state4,
            cell_state5,
            cell_state6
        )
        if self.is_use_cdna:
            cdna_input = self.flatten(hidden_state5)
            #cdna transform
            next_transformed, cdna_kerns = self.cdna(last_frame, cdna_input) 
            #skip connection
            masks, future_frame = self.future_frame_decoder(hidden_state6, out_hiddens)
            # next_transformed = [_+ future_frame for _ in next_transformed]
            mask_list = torch.split(masks, 1, dim=1)
            output = mask_list[0] * last_frame
            

            for frame, mask in zip(next_transformed, mask_list[1:]):
                output += frame * mask
        else:
            masks, future_frame = self.future_frame_decoder(last_frame, out_hiddens)
            mask_list = torch.split(masks, 1, dim=1)
            output = mask_list[0] * last_frame


        return output
        
RELU_SHIFT = 1e-10
class CDNA(nn.Module):
    def __init__(self, num_masks, color_channels, kernel_size, z_dim=128):
        super(CDNA, self).__init__()
        self.num_masks = num_masks 
        self.color_channels = color_channels
        self.kernel_size = kernel_size 
        self.cdna_params = nn.Linear(in_features=7*7*4*z_dim, out_features=self.kernel_size[0] * self.kernel_size[1] * self.color_channels) 


    def forward(self, prev_image, cdna_input):
        batch_size = cdna_input.shape[0]
        height = prev_image.shape[2]
        width = prev_image.shape[3]

        # Predict kernels using a linear function of the last hidden layer.
        cdna_kerns = self.cdna_params(cdna_input)
        
        cdna_kerns = cdna_kerns.view(batch_size, -1, self.kernel_size[0], self.kernel_size[1])
        
        cdna_kerns = torch.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = torch.sum(cdna_kerns, (1, 2, 3), keepdim=True)
        cdna_kerns /= norm_factor

        cdna_kerns = cdna_kerns.view(batch_size, self.color_channels, self.kernel_size[0], self.kernel_size[1])
        transformed = nn.functional.conv2d(prev_image, cdna_kerns, padding=(self.kernel_size[0] - 1) // 2, groups=1, stride=1) # --> [channle, batch_size, h, w]
        transformed = torch.split(transformed, 1, dim=1)
        
        return transformed, cdna_kerns
       
        
        
        
@staticmethod
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