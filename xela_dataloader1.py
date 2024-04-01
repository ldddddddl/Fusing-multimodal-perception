# -*- coding: utf-8 -*-

import os
import glob

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import h5py
import deepdish as dd
from PIL import Image
import csv
import numpy as np
from time import sleep
import cv2
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd
from torch.utils.data import DataLoader, random_split
import random
import shutil

# 读取一个excel表中的数据(一个表格)
# P = 1 / ( ((-1.19e-4) + R * 5e-8) * 102 ) 单位为 N 的方程
def readExcel(file):
    df = pd.read_excel(file, header = None)
    df = df.T
    #print(df)
    
    # 时间
    T = df.iloc[0, 1:]
    T = T.astype(float)
    T = np.array(T, dtype=np.float64)
    
    # 电阻值
    R = df.iloc[1, 1:]   #只读取一行，转置过
    R = R.astype(float)
    R = np.array(R, dtype = np.float64) # 这里的电阻单位是 Ω
    #print(R)
    
    # 压力值 
    #P = 1 / ((-1.19e-4) + R * 5e-8) # 单位是 g
    P = 1 / ( ((-1.19e-4) + R * 5e-8) * 102 ) # 单位是 N
    #print(P)
    
    # 返回触觉数据【一维，numpy形式】
    return P


def readExcel_flex(file):
    df = pd.read_excel(file, header = None)
    df = df.T
    #print(df)
    
    # 时间
    T = df.iloc[0, 1:]
    T = T.astype(float)
    T = np.array(T, dtype=np.float64)
    
    # 电阻值
    R = df.iloc[1, 1:]   #只读取一行，转置过
    R = R.astype(float)
    R = np.array(R, dtype = np.float64) # 这里的电阻单位是 Ω
    #print(R)
    
    # 压力值 
    #P = 1 / ((-1.19e-4) + R * 5e-8) # 单位是 g
    #P = 1 / ( ((-1.19e-4) + R * 5e-8) * 102 ) # 单位是 N
    #print(P)
    P=R
    # 返回应变数据【一维，numpy形式】
    return P

def labels2cat(label_encoder, list):
    return label_encoder.transform(list)

def labels2onehot(OneHotEncoder, label_encoder, list):
    return OneHotEncoder.transform(label_encoder.transform(list).reshape(-1, 1)).toarray()

def onehot2labels(label_encoder, y_onehot):
    return label_encoder.inverse_transform(np.where(y_onehot == 1)[1]).tolist()

def cat2labels(label_encoder, y_cat):
    return label_encoder.inverse_transform(y_cat).tolist()

#def load_data(path,clas,init,length,log):
def load_data(path,visual_seq_length,tactile_seq_length,flex_seq_length,slide_window_size):  #load_data(test_dir, 6, visual_seq_length, log)

    
    #caselist=os.listdir(path+"/"+clas+'/')
    caselist=os.listdir(path+'/')
    cases=[]
    for case in caselist:
        # print(case)
        # rowTemßp=[]
        '''
        pathTemp_visual=path+"/"+clas+'/'+case+'/'+'visual/'
        pathTemp_tactile=path+"/"+clas+'/'+case+'/'+'tactile/'
        pathTemp_flex=path+"/"+clas+'/'+case+'/'+'flex/'
        '''
        pathTemp_visual=path+"/"+case+'/'+'visual/'
        pathTemp_tactile=path+"/"+case+'/'+'tactile/'
        pathTemp_flex=path+"/"+case+'/'+'flex/'
        
        num_visual=len(os.listdir(pathTemp_visual))    #-1是减去.npy文件，得到视觉数据长度
        num_tactile=len(os.listdir(pathTemp_tactile))#同上
        num_flex=len(os.listdir(pathTemp_flex))
        YearMonthDay,Hour,Minutes,label=case.split("-")
        YearMonthDay_true = YearMonthDay[0:8]+'-'+ Hour + '-' + Minutes  + '-' + label 
        excelName = YearMonthDay +'-'+ Hour + '-' + Minutes  + '-' + label + '.xlsx'
        
        
        
        #pathTemp_tactile = path + "/" + clas + '/' + case + '/' + 'tactile/'  + YearMonthDay_true + '.xlsx'
        pathTemp_tactile = path + "/" +  case + '/' + 'tactile/'  + YearMonthDay_true + '.xlsx'
        tactile_data = readExcel(pathTemp_tactile)
        num_tactile = len(tactile_data) 
        
        #pathTemp_flex = path + "/" + clas + '/' + case + '/' + 'flex/'  + YearMonthDay_true + '.xlsx'
        pathTemp_flex = path + "/" + case + '/' + 'flex/'  + YearMonthDay_true + '.xlsx'
        flex_data = readExcel_flex(pathTemp_flex)
        num_flex = len(flex_data) 
        #width,force,label=case.split("_")
        # print(num_visual,num_tactile)
        # for i in range(init,num_visual-length,log):
        if num_visual > 16:
            init = (num_visual - visual_seq_length) // 2 + 2
        else:
            init = 0
        # for i in range(init, num_visual - visual_seq_length):
        # for i in range(init, num_visual - init):
        rowTemp=[]
        # print(i)
        rowTemp.append(YearMonthDay)
        rowTemp.append(Hour)
        rowTemp.append(Minutes)
        rowTemp.append(label)
        ##visual_seq_length+1  是为了后续读取下一帧动作图像
        for j in range(visual_seq_length):
            rowTemp.append(pathTemp_visual+str(init+j)+'.jpg')
        
        # for k in range(visual_seq_length):
        #     rowTemp.append(pathTemp_visual+str(i+k)+'.jpg')
        

        for n in range(tactile_seq_length):
            index = num_tactile - n - int(init/2) - 1 - (3*init) #【我这里是要倒着读取的】 
            rowTemp.append(tactile_data[index]) #添加一维触觉数据 
            # print(path+case,i+k)
        # cases.append(rowTemp) ### 为什么这里添加了一边数据后面又添加一遍
        # print(f'cases1:\n{cases}')
            # print(path+case,i+k)  

        for m in range(flex_seq_length):
            index = num_flex - m - int(init/2) - 1 - (3*init)#【我这里是要倒着读取的】 
            rowTemp.append(flex_data[index]) #添加一维拉伸数据 
        cases.append(rowTemp)  
        # print(f'only cases:\n{cases}')         
    return cases
            # writer_train.writerow(rowTemp)
    # csvFile_train.close()

#data-split
data_dir = './sorted data/graspingdata'
train_dir = './sorted data/traindata'
test_dir = './sorted data/testdata'
# clear
def clear_folder(folder_path): 
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

'''
方便测试，把重置数据集暂时屏蔽
'''            
test_flag = 1

if test_flag:
    clear_folder(train_dir)
    clear_folder(test_dir)

    # print("sssss")

    split_ratio = 0.8
    # print("dddd")
    sub_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    random.shuffle(sub_dirs)

    for sub_dir in sub_dirs[:int(len(sub_dirs) * split_ratio)]:
        shutil.copytree(os.path.join(data_dir, sub_dir), os.path.join(train_dir, sub_dir))
    for sub_dir in sub_dirs[int(len(sub_dirs) * split_ratio):]:
        shutil.copytree(os.path.join(data_dir, sub_dir), os.path.join(test_dir, sub_dir))
else:
    print('数据未重置')

def train_dataset(path,visual_seq_length,tactile_seq_length,flex_seq_length,slide_window_size,flag):
    train_dataset = load_data(train_dir, visual_seq_length, tactile_seq_length, flex_seq_length, slide_window_size)  ###数据集为'年月日，时分，标签，图片*6，触觉*3，形变*3'
    #test_dataset = load_data(test_dir, 6, visual_seq_length, log)
    if flag == 'train':
        dataset=train_dataset
    return dataset

def test_dataset(path,visual_seq_length,tactile_seq_length,flex_seq_length,slide_window_size,flag):    
    #train_dataset = load_data(train_dir, 6, visual_seq_length, log)
    test_dataset = load_data(test_dir, visual_seq_length, tactile_seq_length, flex_seq_length, slide_window_size) # 6-->5
    if flag == 'test':
        dataset=test_dataset
    return dataset

class MyDataset1(Dataset):
    def __init__(self, image_paths, visual_seq_length, tactile_seq_length,flex_seq_length,transform_v,transform_t,transform_f,slide_window_size,flag):
        self.image_paths = image_paths
        self.visual_seq_length = visual_seq_length
        self.tactile_seq_length = tactile_seq_length
        self.flex_seq_length = flex_seq_length
        self.transform_v = transform_v
        self.transform_t = transform_t
        self.transform_f = transform_f
        self.slide_window_size = slide_window_size
       # self.csvReader=csv.reader(open(image_paths))
        self.label=[]
        self.visual_sequence=[]
        self.tactile_sequence=[]
        self.flex_sequence=[]
        self.classes=['0','1']
        self.flag=flag
        self.dataset=train_dataset(self.image_paths,self.visual_seq_length,self.tactile_seq_length,self.flex_seq_length,self.slide_window_size,self.flag)
        # self.tactile_sequence_length=[]
        le = LabelEncoder()
        le.fit(self.classes) #去重，升序

# convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[3]))
            # self.tactile_sequence_length.append(int(item[-1]))
            visual = []
            tactile=[]
            flex=[]
            for i in range(self.visual_seq_length):#visual_seq_length + 1 为了后续提取下一帧动作特征
                visual.append(item[4+i])
            for j in range(self.tactile_seq_length):
                tactile.append(item[j+4+self.visual_seq_length]) 
            for k in range(self.flex_seq_length):
                flex.append(item[k+4+self.visual_seq_length+self.tactile_seq_length])
                
            self.visual_sequence.append(visual)
            self.tactile_sequence.append(tactile)
            self.flex_sequence.append(flex)
        self.label=labels2cat(le, self.label)
        #print(len(self.image_sequence))
    def __getitem__(self, index):
        '''
        dataloader加载数据集函数
        '''
        visuals = []
        tactiles=[]
        flexs=[]
        visuals_label = []
        tactile_label = []
        flex_label = []
        self.tactile_sequence = np.array(self.tactile_sequence) # 一维触觉数据 list形式转变为 numpy形式
        self.tactile_sequence = torch.from_numpy(self.tactile_sequence)
        self.flex_sequence = np.array(self.flex_sequence) # 一维拉伸数据 list形式转变为 numpy形式
        self.flex_sequence = torch.from_numpy(self.flex_sequence)
        ###training set
        for i in range(self.slide_window_size):
            visualTemp = Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)
            visuals.append(visualTemp.unsqueeze(1))
            
        for j in range(3 * self.slide_window_size):
            tactileTemp = self.tactile_sequence[index][j] # 一维 触觉 数据
            tactiles.append(tactileTemp.unsqueeze(0))
            
        for k in range(3 * self.slide_window_size):
            flexTemp = self.flex_sequence[index][k] # 一维 拉伸 数据
            flexs.append(flexTemp.unsqueeze(0))
            
        ###label
        for vis in range(self.slide_window_size, self.visual_seq_length, 1): # 输入图片为0--7， 标签图片为1--8
            visual_label_temp = Image.open(self.visual_sequence[index][vis])
            if self.transform_v:
                visual_label_temp = self.transform_v(visual_label_temp)
            visuals_label.append(visual_label_temp.unsqueeze(1))
            
        for tac in range(3 * self.slide_window_size, self.tactile_seq_length, 1):
            tactileTemp = self.tactile_sequence[index][tac] # 一维 触觉 数据
            tactile_label.append(tactileTemp.unsqueeze(0))
            
        for flex in range(3 * self.slide_window_size, self.flex_seq_length, 1):
            flexTemp = self.flex_sequence[index][flex] # 一维 拉伸 数据
            flex_label.append(flexTemp.unsqueeze(0))
                


        x_v = torch.cat(visuals,dim=1)
        # x_t = torch.from_numpy( np.array(tactiles).astype(float) )
        x_t = torch.from_numpy( np.array(tactiles).astype(float) )
        
        # x_f = torch.from_numpy( np.array(flexs).astype(float) )
        x_f = torch.from_numpy( np.array(flexs).astype(float) )
        
        y_grasping_state = torch.tensor(self.label[index], dtype=torch.long)
        
        y_visuals = torch.cat(visuals_label, dim=1)
        y_tactile = torch.from_numpy( np.array(tactile_label).astype(float) ).unsqueeze(0)
        y_flex = torch.from_numpy( np.array(flex_label).astype(float) ).unsqueeze(0)
        y_tac_pred = torch.cat([x_t.unsqueeze(0), y_tactile], dim=1)[:, 3:]
        # print(x_v.shape,x_t.shape,y)
        #
        return x_v,x_t,x_f, (y_visuals, y_tactile, y_flex, y_grasping_state, y_tac_pred)

    def __len__(self):
        return len(self.visual_sequence)

class MyDataset2(Dataset):
    def __init__(self, image_paths, visual_seq_length, tactile_seq_length,flex_seq_length,transform_v,transform_t,transform_f,slide_window_size,flag):
        self.image_paths = image_paths
        self.visual_seq_length = visual_seq_length
        self.tactile_seq_length = tactile_seq_length
        self.flex_seq_length = flex_seq_length
        self.transform_v = transform_v
        self.transform_t = transform_t
        self.transform_f = transform_f
        self.slide_window_size = slide_window_size
       # self.csvReader=csv.reader(open(image_paths))
        self.label=[]
        self.visual_sequence=[]
        self.tactile_sequence=[]
        self.flex_sequence=[]
        self.classes=['0','1']
        self.flag=flag
        self.dataset=test_dataset(self.image_paths,self.visual_seq_length,self.tactile_seq_length,self.flex_seq_length,self.slide_window_size,self.flag)
        # self.tactile_sequence_length=[]
        le = LabelEncoder()
        le.fit(self.classes)

# convert category -> 1-hot
        action_category = le.transform(self.classes).reshape(-1, 1)
        enc = OneHotEncoder()
        enc.fit(action_category)
        for item in self.dataset:
            self.label.append(str(item[3]))
            # self.tactile_sequence_length.append(int(item[-1]))
            visual = []
            tactile=[]
            flex=[]
            for i in range(self.visual_seq_length):
                visual.append(item[4+i])
                
            for j in range(self.tactile_seq_length):
                tactile.append(item[j+4+self.visual_seq_length]) 
                
            for k in range(self.flex_seq_length):
                flex.append(item[k+4+self.visual_seq_length+self.tactile_seq_length])
                
            self.visual_sequence.append(visual)
            self.tactile_sequence.append(tactile)
            self.flex_sequence.append(flex)
        self.label=labels2cat(le, self.label)
        #print(len(self.image_sequence))
    def __getitem__(self, index):

        visuals = []
        tactiles=[]
        flexs=[]
        visuals_label=[]
        tactile_label = []
        flex_label = []
        self.tactile_sequence = np.array(self.tactile_sequence) # 一维触觉数据 list形式转变为 numpy形式
        self.tactile_sequence = torch.from_numpy(self.tactile_sequence)
        self.flex_sequence = np.array(self.flex_sequence) # 一维拉伸数据 list形式转变为 numpy形式
        self.flex_sequence = torch.from_numpy(self.flex_sequence)
        ###train set
        for i in range(self.slide_window_size):
            visualTemp=Image.open(self.visual_sequence[index][i])
            if self.transform_v:
                visualTemp = self.transform_v(visualTemp)
            visuals.append(visualTemp.unsqueeze(1))
        
        for j in range(3 * self.slide_window_size):
            tactileTemp = self.tactile_sequence[index][j] # 一维 触觉 数据
            tactiles.append(tactileTemp.unsqueeze(0))
            
        for k in range(3 * self.slide_window_size):
            flexTemp = self.flex_sequence[index][k] # 一维 拉伸 数据
            flexs.append(flexTemp.unsqueeze(0))
            
        ###label
        for vis in range(self.slide_window_size, self.visual_seq_length, 1): # 输入图片为0--7， 标签图片为1--8
            visual_label_temp = Image.open(self.visual_sequence[index][vis])
            if self.transform_v:
                visual_label_temp = self.transform_v(visual_label_temp)
            visuals_label.append(visual_label_temp.unsqueeze(1))
            
        for tac in range(3 * self.slide_window_size, self.tactile_seq_length, 1):
            tactileTemp = self.tactile_sequence[index][tac] # 一维 触觉 数据
            tactile_label.append(tactileTemp.unsqueeze(0))
            
        for flex in range(3 * self.slide_window_size, self.flex_seq_length, 1):
            flexTemp = self.flex_sequence[index][flex] # 一维 拉伸 数据
            flex_label.append(flexTemp.unsqueeze(0))
                
                


        x_v = torch.cat(visuals,dim=1)
        x_t = torch.from_numpy( np.array(tactiles).astype(float) )
        x_f = torch.from_numpy( np.array(flexs).astype(float) )
        y_grasping_state = torch.tensor(self.label[index], dtype=torch.long)
        y_visuals = torch.cat(visuals_label, dim=1)
        y_tactile = torch.from_numpy( np.array(tactile_label).astype(float) ).unsqueeze(0)
        y_flex = torch.from_numpy( np.array(flex_label).astype(float) ).unsqueeze(0)
        y_tac_pred = torch.cat([x_t.unsqueeze(0), y_tactile], dim=1)[:, 3:]

        
        # print(x_v.shape,x_t.shape,y)
        return x_v,x_t,x_f, (y_visuals, y_tactile, y_flex, y_grasping_state, y_tac_pred)

    def __len__(self):
        return len(self.visual_sequence)
