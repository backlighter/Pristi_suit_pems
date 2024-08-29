import pickle
# 禁用科学计数法
import torch
import numpy as np
torch.set_printoptions(sci_mode=False)



def evaluate(data):
    pre=data[0]
    true=data[1]
    mask=data[2]
    median_dim2 = torch.median(pre, dim=1).values
    # 第一维和第三维按顺序合并
    result_w = median_dim2.reshape(-1, 432, 1)  #插补后的
    result_w=result_w*416.38598+318.9039
    true=true*416.38598+318.9039
    result_w[result_w<0] = 0
    result_w = result_w.cpu().numpy()
    true_w=true.reshape(-1, 432, 1)
    mask_w=mask.reshape(-1, 432, 1)
    true_w=true_w.cpu().numpy()


    # mask_w #1 代表缺失值，0代表观测值
    mask_w=mask_w.cpu().numpy()  # 需要评测的数据
    result_w=result_w*mask_w

    mae = np.abs((true_w - result_w) * mask_w).sum() / mask_w.sum()
    rmse=np.sqrt((((true_w - result_w)* mask_w)**2).sum() / mask_w.sum())
    temp_mape = np.abs((true_w - result_w) / true_w) * mask_w
    mape = temp_mape.sum() / mask_w.sum() * 100
    print("rmse,mae,mape",mae,rmse,mape)
    

file_path="/data/LiYuxiang/AAAI/PriSTI/PriSTI-main/save/YunNan_SR-TR_0.9_20240331_232655/generated_outputs_nsample10.pk"
# 打开.pk文件以读取二进制数据
with open(file_path, 'rb') as file:

    # 使用pickle.load()函数加载数据
    data = pickle.load(file)
evaluate(data)