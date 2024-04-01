
import torch.nn as nn
from models.models_utils import init_weights
from models.layers2 import CausalConv1D, Flatten, conv3d
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import torchvision.models as models
import numpy
from utils.models_utils import (duplicate,gaussian_parameters,product_of_experts,sample_gaussian)#device


class FlexEncoder(nn.Module):
    def __init__(self, z_dim, num_img=12, initailize_weights=True, device='cuda'):
        """
        应变力数据，使用因果卷积搭建模型
        """
        super(FlexEncoder, self).__init__()
        self.z_dim = z_dim
        self.num_img = num_img
        
        self.flex_causconv1 = CausalConv1D(1, 12, kernel_size=2, stride=2)
        self.flex_causconv2 = CausalConv1D(12, 24, kernel_size=2, stride=2)
        self.flex_causconv3 = CausalConv1D(24, 48, kernel_size=2, stride=2)
        self.flex_causconv4 = CausalConv1D(48, 96, kernel_size=2, stride=2)
        if self.num_img == 12:
            self.flex_causconv5 = CausalConv1D(96, self.z_dim//2, kernel_size=2, stride=1)
        elif self.num_img == 8:
            self.flex_causconv5 = CausalConv1D(96, self.z_dim, kernel_size=2, stride=1)
        self.flex_flatten = nn.Flatten()
        if initailize_weights:
            init_weights(self.modules(), device=device)

    def forward(self, flex):
        
        flex_causconv_result1, flex_layers_result1 = self.flex_causconv1(flex)
        flex_causconv_result2, flex_layers_result2 = self.flex_causconv2(flex_causconv_result1)
        flex_causconv_result3, flex_layers_result3 = self.flex_causconv3(flex_causconv_result2)
        flex_causconv_result4, flex_layers_result4 = self.flex_causconv4(flex_causconv_result3)
        flex_causconv_result5, flex_layers_result5 = self.flex_causconv5(flex_causconv_result4)
        flatten_result = self.flex_flatten(flex_causconv_result5)
        flex_layers_results = (
            flex_layers_result1,
            flex_layers_result2,
            flex_layers_result3,
            flex_layers_result4,
            flex_layers_result5
            )
            
        return flatten_result.unsqueeze(-1), flex_layers_results
        

class TactileEncoder(nn.Module):
    def __init__(self, z_dim, num_img=12, initailize_weights=True):
        """
        触觉数据，使用因果卷积搭建模型
        """
        super(TactileEncoder, self).__init__()
        self.z_dim = z_dim
        self.num_img = num_img

        self.tac_causconv1 = CausalConv1D(1, 12, kernel_size=2, stride=2)
        self.tac_causconv2 = CausalConv1D(12, 24, kernel_size=2, stride=2)
        self.tac_causconv3 = CausalConv1D(24, 48, kernel_size=2, stride=2)
        self.tac_causconv4 = CausalConv1D(48, 96, kernel_size=2, stride=2)
        if self.num_img == 12:
            self.tac_causconv5 = CausalConv1D(96, self.z_dim//2, kernel_size=2, stride=1)
        elif self.num_img == 8:
            self.tac_causconv5 = CausalConv1D(96, self.z_dim, kernel_size=2, stride=1)
            
        self.tac_flatten = nn.Flatten()

        if initailize_weights:
            init_weights(self.modules())

    def forward(self, tactile):
        # tactile_out = self.tactile_encoder(tactile).unsqueeze(2)
        leaky_result1, causconv1_result1 = self.tac_causconv1(tactile)
        leaky_result2, causconv1_result2 = self.tac_causconv2(leaky_result1)
        leaky_result3, causconv1_result3 = self.tac_causconv3(leaky_result2)
        leaky_result4, causconv1_result4 = self.tac_causconv4(leaky_result3)
        leaky_result5, causconv1_result5 = self.tac_causconv5(leaky_result4)
        flatten_result = self.tac_flatten(leaky_result5)
        
        tac_layers_results = (
            causconv1_result1,
            causconv1_result2,
            causconv1_result3,
            causconv1_result4,
            causconv1_result5,
            )
        
        
        return flatten_result.unsqueeze(-1), tac_layers_results


class ImageEncoder(nn.Module):
    def __init__(self, z_dim, model_flag, num_img=12, initailize_weights=True):
        """
        图片编码器，使用c3d进行卷积，结尾用lstm连接学习次序特征
        """
        super(ImageEncoder, self).__init__()
        self.z_dim = z_dim
        self.model_flag = model_flag
        self.num_img = num_img
        if self.model_flag == 'normal':
            self.img_conv1 = conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            
            self.img_conv2 = conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            
            self.img_conv3a = conv3d(128, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            # self.img_conv3b = conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1))
            
            # self.img_conv4a = conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            # # self.img_conv4b = conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            # self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            
            self.img_conv5a = conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.img_conv5b = conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
            
            self.img_conv6 = conv3d(256, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1), stride=(3, 3, 3))
            self.flatten = Flatten()
            self.dropout = nn.Dropout(p=0.3)

            self.fc1 = nn.Linear(6400, 3200)
            self.fc2 = nn.Linear(3200, 1024)
            self.fc = nn.Linear(1024, self.z_dim)
            
        elif self.model_flag == 'convlstm':
            
            # self.conv2d1 = nn.Conv2d(3, 64, 3, padding=1, stride=2)
            self.conv3d1 = conv3d(3, 64, (3, 3, 3), padding=(1, 1, 1), stride=(2, 2, 2), bias=True)
            # self.conv2d2 = nn.Conv2d(256, 128, 5, padding=3, stride=2)
            self.conv3d2 = conv3d(256, 128, (5, 5, 5), padding=(3, 3, 3), stride=(2, 2, 2))
            self.batch_norm = nn.BatchNorm3d(128)
            self.conv_lstm1 = ConvLSTM(input_channels=64, hidden_channels=64, num_layers=1, first_flag=True)
            self.conv_lstm2 = ConvLSTM(input_channels=64, hidden_channels=128, num_layers=1, stride=2, size_flag='up')
            self.conv_lstm3 = ConvLSTM(input_channels=128, hidden_channels=128, num_layers=1)
            self.conv_lstm4 = ConvLSTM(input_channels=128, hidden_channels=256, num_layers=1, stride=2, size_flag='up')
            self.flatten = nn.Flatten()
            self.dropout = nn.Dropout(0.3)
            if self.num_img == 12:
                self.fc = nn.Linear(18816, self.z_dim)

            elif self.num_img == 8:
                self.fc = nn.Linear(65536, self.z_dim)
                
        if initailize_weights:
            init_weights(self.modules())

    def forward(self, image):
        # image encoding layers
        if self.model_flag == 'normal':
            out_img_conv1 = self.img_conv1(image)
            out_img_pool1 = self.pool1(out_img_conv1)
            
            out_img_conv2 = self.img_conv2(out_img_pool1)
            out_img_pool2 = self.pool2(out_img_conv2)
            
            out_img_conv3b = self.img_conv3a(out_img_pool2)
            out_img_pool3 = self.pool3(out_img_conv3b)
        
            # out_img_conv4b = self.img_conv4a(out_img_pool3)
            # out_img_pool4 = self.pool4(out_img_conv4b)
            
            out_img_conv5b = self.img_conv5b(self.img_conv5a(out_img_pool3))
            out_img_pool5 = self.pool5(out_img_conv5b) # [8, 512, 2, 4, 4]

            img_out_convs = (
                out_img_pool1,
                out_img_pool1,
                out_img_pool2,
                out_img_pool3,
                out_img_pool5,
            )
            out_img_conv6 = F.leaky_relu(self.img_conv6(out_img_pool5))
            # image embedding parameters
            flattened = self.flatten(out_img_conv6) # [8, 16384]
            # img_out = self.img_encoder(out_img_conv5b).unsqueeze(2)
            dropout = self.dropout(flattened)

            out = F.relu(self.fc1(dropout))
            out = F.relu(self.fc2(self.dropout(out)))
            img_out = self.fc(out).unsqueeze(2)

            return img_out, img_out_convs
        elif self.model_flag=='convlstm':
            # conv_results = []
            # for seq in range(image.shape[2]):
            #     temp_conv = self.conv2d1(image[:, :, seq, :, :])
            #     conv_results.append(temp_conv)
            out_conv3d1 = self.conv3d1(image)
            out_h_state1, out_c_state1 = self.conv_lstm1(out_conv3d1)
            out_h_state2, out_c_state2 = self.conv_lstm2(out_h_state1, out_c_state1)
            out_h_state3, out_c_state3 = self.conv_lstm3(out_h_state2, out_c_state2)
            out_h_state4, out_c_state4 = self.conv_lstm4(out_h_state3, out_c_state3)
            '''
            将结果合成为序列
            '''
            # conv_results2 = []
            # for tens in out_h_state4:
            #     temp_conv2 = self.conv2d2(tens)
            #     conv_results2.append(temp_conv2)
            # cat_result = frame_to_stream(conv_results2)
            out_h_state4_stream = frame_to_stream(out_h_state4)
            out_conv3d2 = self.conv3d2(out_h_state4_stream)
            out_batch_norm = self.batch_norm(out_conv3d2)
            out_flatten = self.flatten(out_batch_norm)
            out_fc = self.fc(out_flatten)
            layers_results = (
                frame_to_stream(out_conv3d1),
                frame_to_stream(out_h_state1),
                frame_to_stream(out_h_state2),
                frame_to_stream(out_h_state3),
                out_h_state4_stream,
            )
            return out_fc.unsqueeze(-1), layers_results
        
@staticmethod
def frame_to_stream(in_list):
    temp_state = [temp.unsqueeze(2) for temp in in_list]
    cat_result = torch.cat(temp_state, dim=2)
    return cat_result
        
class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, in_b_n_h_w, kernel_size=3, stride=1, size_flag='remain'):
        super(ConvLSTMCell, self).__init__()
        padding = kernel_size // 2
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        try:
            b, c, n, h, w = in_b_n_h_w
            in_dim_num = 5
        except:
            b, n, w = in_b_n_h_w
            in_dim_num = 3
        self.size_flag = size_flag
        if in_dim_num == 5:
            self.wci = torch.zeros(b, hidden_channels, h, w, requires_grad=True, device='cuda')
            self.wcf = torch.zeros(b, hidden_channels, h, w, requires_grad=True, device='cuda')
            self.wco = torch.zeros(b, hidden_channels, h, w, requires_grad=True, device='cuda')
        elif in_dim_num == 3:
            self.wci = torch.zeros(b, hidden_channels, w, requires_grad=True, device='cuda')
            self.wcf = torch.zeros(b, hidden_channels, w, requires_grad=True, device='cuda')
            self.wco = torch.zeros(b, hidden_channels, w, requires_grad=True, device='cuda')
        if self.size_flag == 'remain':
            self.conv = nn.Conv2d(
                in_channels=input_channels + hidden_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif self.size_flag == 'up':
            self.conv = nn.Conv2d(
                in_channels=input_channels + input_channels,
                out_channels=4 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
    def forward(self, input_tensor, cur_state, first_flag):
        if first_flag:
            h_cur, c_cur = cur_state[-1]
        else:
            h_cur, c_cur = cur_state
        combined = torch.cat((input_tensor, h_cur), dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, combined_conv.size(1) // 4, dim=1)#[8, 128, 28, 28]
        if self.size_flag == 'up':
            c_cur = reshape_tensor(c_cur, cc_i)
            self.wci = reshape_tensor(self.wci, cc_i)
            self.wcf = reshape_tensor(self.wcf, cc_f)
            self.wco = reshape_tensor(self.wco, cc_o)
            
        i = torch.sigmoid(cc_i + self.wci[:, :, :c_cur.size(2), :c_cur.size(2)]*c_cur)
        f = torch.sigmoid(cc_f + self.wcf[:, :, :c_cur.size(2), :c_cur.size(2)]*c_cur)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        o = torch.sigmoid(cc_o + self.wco[:, :, :c_next.size(2), :c_cur.size(2)]*c_next)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

@staticmethod
def reshape_tensor(source_tensor, target_tensor):
    t_b, t_c, t_h, t_w = target_tensor.shape
    s_b, s_c, s_h, s_w = source_tensor.shape
    temp_tensor = source_tensor.view(t_b, -1, t_h, t_w)
    if temp_tensor.size(1) > t_c:
        temp_tensor = temp_tensor[:, :t_c, :, :]
    elif temp_tensor.size(1) / t_c == 2:
        temp_tensor = torch.cat((temp_tensor, temp_tensor), dim=1)
    else:
        temp_tensor = torch.cat((temp_tensor, temp_tensor), dim=1)
        temp_tensor = temp_tensor[:, :t_c, :, :]
    return temp_tensor

class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, in_b_n_h_w=(8, 3, 12, 112, 112), kernel_size=3, stride=1, padding=1, num_layers=1, first_flag=False, size_flag='remain', input_mode='stream'):
        super(ConvLSTM, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_layers = num_layers
        self.size_flag = size_flag
        self.input_mode = input_mode
        if self.stride == 2 or self.size_flag == 'up':
            self.stride = 2
            self.size_flag = 'up'
        try:
            b, c, n, h, w = in_b_n_h_w
            self.in_dim_num = 5
        except:
            b, n, w = in_b_n_h_w
            self.in_dim_num = 3
            
        self.first_flag = first_flag
        cell_list = []
        for i in range(self.num_layers):
            cur_input_channels = input_channels if i == 0 else hidden_channels
            cell_list.append(ConvLSTMCell(cur_input_channels, hidden_channels, in_b_n_h_w, kernel_size, stride, size_flag=self.size_flag))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, c_state=None):
        if self.input_mode == 'stream':
            in_b_c_h_w = list(input_tensor[-1].shape)
            if isinstance(input_tensor, torch.Tensor):
                seq_len = in_b_c_h_w[1]
            elif isinstance(input_tensor, list):
                seq_len = len(input_tensor)
        elif self.input_mode == 'frame':
            in_b_c_h_w = list(input_tensor.shape)
            seq_len = 1
        
        if c_state is None:
            c_state = self.init_hidden(input_tensor, in_b_c_h_w, in_mode_flag=self.input_mode)
        h_states = []
        c_states = []
        
        for seq in range(seq_len):
            if self.input_mode == 'stream':
                if isinstance(input_tensor, torch.Tensor):
                    temp_input = input_tensor[:, :, seq, :, :]
                elif isinstance(input_tensor, list):
                    temp_input = input_tensor[seq]
            elif self.input_mode == 'frame':
                temp_input = input_tensor
                
            for i in range(self.num_layers):
                if c_state is None or c_state == []:
                    cur_hidden = None
                # elif self.input_mode == 'stream':
                elif self.input_mode == 'frame' and not self.first_flag:
                    cur_hidden = c_state
                else:
                    cur_hidden = c_state[i]
                    
                h_state, cell_state = self.cell_list[i](temp_input, (temp_input, cur_hidden), self.first_flag)
                if self.input_mode == 'frame':
                    h_states, c_states = h_state, cell_state 
                elif self.input_mode == 'stream':
                    h_states.append(h_state)
                    c_states.append(cell_state)
        return h_states, c_states

    def init_hidden(self, input_tensor, in_b_c_h_w, device='cuda', in_mode_flag='stream'):
        hidden_states = []
        for i in range(self.num_layers):
            if i == 0:
                if len(in_b_c_h_w) == 5 : 
                    hidden_states.append((torch.zeros(in_b_c_h_w[0], in_b_c_h_w[1], in_b_c_h_w[3], in_b_c_h_w[4], device=device),
                                        torch.zeros(in_b_c_h_w[0], in_b_c_h_w[1], in_b_c_h_w[3], in_b_c_h_w[4], device=device)))
                elif len(in_b_c_h_w) == 4 and in_mode_flag == 'frame':
                    hidden_states.append((torch.zeros(*in_b_c_h_w, device=device),
                                        torch.zeros(*in_b_c_h_w, device=device)))
                elif len(in_b_c_h_w):
                    hidden_states.append((torch.zeros(input_tensor.shape[0], in_b_c_h_w[0], in_b_c_h_w[2], in_b_c_h_w[3], device=device),
                                        torch.zeros(input_tensor.shape[0], in_b_c_h_w[0], in_b_c_h_w[2], in_b_c_h_w[3], device=device)))
            else:
                hidden_states.append(None)
        return hidden_states

    
