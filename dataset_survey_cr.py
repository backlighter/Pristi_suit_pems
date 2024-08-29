import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
import torchcde
import os
from tqdm import tqdm
from utils import get_randmask, get_block_mask



class Survey_Dataset(Dataset):
    def __init__(self,true_data,ob_mask,gt_mask, c_data,val_start, test_start, eval_length=12, mode="train  ", missing_pattern='block',
                 is_interpolate=False, target_strategy='random', missing_ratio=None):
        """
        observed_mask,observed_data  - Data that can be seen throughout the process, 
                                       from which a portion will be intercepted for input and evaluation

        gt_mask  - The input part of the test set

        cond_mask - The input part of the train and valid set

        cut_length - The length of the truncation required to avoid repeating a evaluate for a given point in time.
        #避免重复给定时间点的求值所需的截断长度。
        coeffs - Results of linear interpolation
        """
        self.eval_length = eval_length
        self.is_interpolate = is_interpolate #是否插值
        self.target_strategy = target_strategy
        self.mode = mode
        self.missing_ratio = missing_ratio
        self.missing_pattern = missing_pattern

        # if mode == 'train':
        #     self.observed_mask = ob_mask
        #     self.gt_mask = gt_mask
        #     self.observed_data = c_data[:val_start]
        # elif mode == 'valid':
        #     # self.observed_mask = ob_mask[val_start: test_start]
        #     # self.gt_mask = gt_mask[val_start: test_start]
        #     self.observed_mask = ob_mask
        #     self.gt_mask = gt_mask
        #     self.observed_data = c_data[val_start: test_start]
        # elif mode == 'test':
        #     # The test set does not need to be constructed condition mask
        #     #测试集不需要构建条件Mask
        #     # and the remaining miss values are interpolated directly using our constructed data
        #     #这些保留的缺失值 直接使用我们构建的数据进行插值
        #     self.observed_mask = ob_mask
        #     self.gt_mask = gt_mask
        #     self.observed_data = true_data[test_start:]
        #ob_mask 对应原始数据缺失的mask
        #gt_mask 对应缺失掉百分之50之后的Mask
        #c_data  归一化后的原数据 噪声加到c_data上
        if mode == 'train':
            self.observed_mask = ob_mask[:val_start]
            self.gt_mask = gt_mask[:val_start]
            self.observed_data = c_data[:val_start]
        elif mode == 'valid':
            self.observed_mask = ob_mask[val_start: test_start]
            self.gt_mask = gt_mask[val_start: test_start]
            self.observed_data = c_data[val_start: test_start]
        elif mode == 'test':
            # The test set does not need to be constructed condition mask
            # and the remaining miss values are interpolated directly using our constructed data
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = true_data[test_start:]  
        elif mode == 'all':
            test_start=0  #使用全量数据
            self.observed_mask = ob_mask[test_start:]
            self.gt_mask = gt_mask[test_start:]
            self.observed_data = true_data[test_start:]
            
        self.current_length = len(self.observed_mask) - eval_length + 1
        self.use_index = np.arange(self.current_length)
        self.cut_length = [0] * len(self.use_index)

        if mode == 'test' or mode == 'all':

            index = 0
            self.current_length = 0
            self.use_index = []

            while index + eval_length <= len(self.observed_mask):
                self.use_index.append(index)
                self.current_length += 1
                index += eval_length
            
            self.cut_length = [0] * len(self.use_index)

            if len(self.observed_mask) % eval_length !=0:
                self.use_index.append(len(self.observed_mask) - eval_length)
                self.current_length += 1
                #self.cut_length.append()    
                self.cut_length.append(eval_length - (len(self.observed_mask) - self.use_index[-1]))
                self.cut_length[-1] = eval_length - (len(self.observed_mask) - index)

    def __getitem__(self, org_index):
        # 使用提供的索引从use_index中获取实际索引
        index = self.use_index[org_index]
        # 根据索引和预设的长度（eval_length），提取观测数据
        ob_data = self.observed_data[index: index + self.eval_length]  #根据index 和切片划分滑窗
        # 提取与观测数据对应的mask矩阵，表示数据中哪些是观测到的
        ob_mask = self.observed_mask[index: index + self.eval_length]  #ob_mask：原始数据的mask矩阵
        # 将观测数据的mask矩阵转换为torch.tensor并设置为float类型
        ob_mask_t = torch.tensor(ob_mask).float()
        # 提取输入数据的真实mask矩阵
        gt_mask = self.gt_mask[index: index + self.eval_length]   #输入数据的mask矩阵



        # 将真实mask矩阵转换为torch.tensor并设置为float32类型
        #cond_mask就是gt_mask
        cond_mask = torch.tensor(gt_mask).to(torch.float32)
        # if self.mode != 'train':
        #     cond_mask = torch.tensor(gt_mask).to(torch.float32)
        # else:
        #     if self.target_strategy != 'random':
        #         cond_mask = get_block_mask(ob_mask_t, target_strategy=self.target_strategy,min_seq=3, max_seq=12)
        #     else:
        #         cond_mask = get_randmask(ob_mask_t)
        # 准备返回的数据字典，包括观测数据、对应的mask矩阵等
        s = {
            "observed_data": ob_data,        # 模型的输入，原始数据 [12,432]
            "observed_mask": ob_mask,        # 原始数据的mask矩阵 [12,432]
            "gt_mask": gt_mask,              # 输入数据的真实mask矩阵 [12,432]
            "timepoints": np.arange(self.eval_length),# 时间点序列
            "cut_length": self.cut_length[0],# 数据切割长度
            "cond_mask": cond_mask.numpy()   # 条件mask矩阵，用于模型条件生成  cond_mask就是gt_mask
        }
        # 如果设置了插值标志，进行数据插值处理
        if self.is_interpolate:
            # 将观测数据转换为torch.tensor并设置为float64类型 tmp_data就是ob_data
            tmp_data = torch.tensor(ob_data).to(torch.float64)
            # 使用条件mask进行插值处理，未观测的数据位置用NaN填充 itp_data就是ob_data插值后
            itp_data = torch.where(cond_mask == 0, float('nan'), tmp_data).to(torch.float32)
            #使用线性插值
            itp_data = torchcde.linear_interpolation_coeffs(itp_data.permute(1, 0).unsqueeze(-1)).squeeze(-1).permute(1, 0)
            
            s["coeffs"] = itp_data.numpy()
        # 返回处理后的数据字典
        return s

    def __len__(self):
        return self.current_length

def cal_val_test_mask(true_data, miss_ratio, mode = "val"):

    new_mask=np.where(true_data,1,0) #获取原始mask
    if mode == "train":
        ori_mask = new_mask[:int(true_data.shape[0] * 0.6)]
        double_mask = new_mask[:int(true_data.shape[0] * 0.6)]
    elif mode=="val":
        ori_mask = new_mask[int(true_data.shape[0] * 0.6):int(true_data.shape[0] * 0.8)]
        double_mask = new_mask[int(true_data.shape[0] * 0.6):int(true_data.shape[0] * 0.8)]
    elif mode == "test":
        ori_mask = new_mask[int(true_data.shape[0] * 0.8):]
        double_mask = new_mask[int(true_data.shape[0] * 0.8):]
    ori_mask = ori_mask[:,:,np.newaxis]

    # test_part_mask=new_mask[test_index:,:] 
    # 手动构造test部分的缺失矩阵为test_part_mask
    for col in range(double_mask.shape[1]):  # 遍历每一列
        rows_with_1 = np.where(double_mask[:, col] == 1)[0]  # 找出当前列中值为1的行索引
        num_to_zero = int(len(rows_with_1) * miss_ratio)  # 计算miss_rate%的数量
        rows_to_zero = np.random.choice(rows_with_1, size=num_to_zero, replace=False)  # 随机选择这些行
        double_mask[rows_to_zero, col] = 0  # 将这些行中的1置为0
    
    
    double_mask=double_mask[:, :, np.newaxis]
    return ori_mask, double_mask
def get_dataloader(batch_size, device, val_len=0.2, test_len=0.2, missing_pattern='block',
                   is_interpolate=False, num_workers=4, target_strategy='random',data_prefix='',miss_type='SR-TR',miss_rate=0.1):
    """
    generate sample and split data for train,valid,test
    
    Parameters: 
    is_interpolate  - Whether to do linear interpolation
    target_strategy  - strategy of conditional mask
    
    Returns:
    Completed Tensor  -  numpy.array(Node, points_per_day, days)
    """
    
    true_datapath = os.path.join(data_prefix,f"true_data_{miss_type}_{miss_rate}_v2.npz")
    miss_datapath = os.path.join(data_prefix,f"miss_data_{miss_type}_{miss_rate}_v2.npz")

    # 两个mask矩阵，ob——mask对应原始数据的，gt——mask对应认为再次构造50%的缺失，1表示观测值的位置，0表示缺失值的位置
    ob_mask = np.load(true_datapath)['mask'].astype('uint8')[:,:,0]
    gt_mask = np.load(miss_datapath)['mask'].astype('uint8')[:,:,0]

    # 计算真实值的均值和方差，true_data代表所有的观测值
    true_data = np.load(true_datapath)['data'].astype(np.float32)[:, :, 0]
    if np.isnan(true_data).any(): #判断原矩阵中除0之外是否还有NaN 我印象中已经处理过了
        print("true_data contains NaN values.")
    true_data[np.isnan(true_data)] = 0#把NaN转成0

    train_mean, train_std= np.mean(true_data[ob_mask==1]), np.std(true_data[ob_mask==1])
    true_data = (true_data - train_mean)/train_std

    #这个地方经过归一化之后，0就不再代表着没有值了 因为是归一化之后的结果
    c_data = np.copy(true_data) * ob_mask  #用于模型输入，训练时，前向加噪加在它上面
    
    T,N = true_data.shape
    print(f'raw data shape:{ true_data.shape}')

    

    val_start = int((1 - val_len - test_len) * T)
    test_start = int((1 - test_len) * T)



    dataset = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="train", 
                              missing_pattern=missing_pattern,
                             is_interpolate=is_interpolate, target_strategy=target_strategy)
    train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    print(f'train dataset len:{dataset.__len__()}')


    # dataset_test = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="test", 
    #                                missing_pattern=missing_pattern,
    #                               is_interpolate=is_interpolate, target_strategy=target_strategy)

    dataset_test = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="test", 
                                   missing_pattern=missing_pattern,
                                  is_interpolate=is_interpolate, target_strategy=target_strategy)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print(f'test dataset len:{dataset_test.__len__()}')

    dataset_valid = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="valid", 
                                   missing_pattern=missing_pattern,
                                   is_interpolate=is_interpolate, target_strategy=target_strategy)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print(f'val dataset len:{dataset_valid.__len__()}')

    scaler = torch.tensor(train_std).to(device).float()
    mean_scaler = torch.tensor(train_mean).to(device).float()

    print(f'scaler: {scaler}')
    print(f'mean_scaler: {mean_scaler}')

    #all_data  使用全量数据  使用模型对全量数据进行插补
    dataset_all = Survey_Dataset(true_data,ob_mask,gt_mask, c_data,val_start, test_start,mode="all", 
                                   missing_pattern=missing_pattern,
                                  is_interpolate=is_interpolate, target_strategy=target_strategy)
    all_loader = DataLoader(dataset_all, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    print(f'all dataset len:{dataset_all.__len__()}')



    return train_loader, valid_loader, test_loader, scaler, mean_scaler,N,all_loader




