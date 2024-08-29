import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import logging
import time
import nni

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    foldername="",
):
    use_nni = False
    if 'nni' in config.keys():
        use_nni = config['nni']

    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    is_lr_decay = config["is_lr_decay"]
    if foldername != "":
        output_path = foldername + "/model.pth"
        logging.basicConfig(filename=foldername + '/train_model.log', level=logging.DEBUG)
    if is_lr_decay:
        p1 = int(0.75 * config["epochs"])
        p2 = int(0.9 * config["epochs"])
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[p1, p2], gamma=0.1
        )

    valid_epoch_interval = config["valid_epoch_interval"]
    best_valid_loss = 1e10

    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            logging.info("avg_epoch_loss:" + str(avg_loss / batch_no) + ", epoch:" + str(epoch_no))
            if is_lr_decay:
                lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0 and (epoch_no + 1) > config["epochs"] * 0.5:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
                    logging.info("valid_avg_epoch_loss"+str(avg_loss_valid / batch_no)+", epoch:"+str(epoch_no))
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
                logging.info("best loss is updated to "+str(avg_loss_valid / batch_no)+" at "+str(epoch_no))
                if foldername != "":
                    torch.save(model.state_dict(), foldername + "/tmp_model"+str(epoch_no)+".pth")
            if use_nni:
                nni.report_intermediate_result(avg_loss_valid)
    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="",dataset_name=None,miss_rate=None):
    with torch.no_grad():
        model.eval()
        mse_total = 0 #初始化总均方误差
        mae_total = 0 #初始化平均绝对误差
        mape_total = 0  #初始化总平均绝对百分百误差
        evalpoints_total = 0 #初始化总评估点数
        all_imputed_data = []
        
        # 初始化用于聚合输出的列表
        all_target = []  # 存储所有目标值
        all_observed_point = []  # 存储所有观测点
        all_observed_time = []  # 存储所有观测时间
        all_evalpoint = []  # 存储所有评估点
        all_generated_samples = []  # 存储所有生成的样本
        all_gen_samples=[]
        # nsample 控制生成的样本数量
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it: #根据batch-size计算
            for batch_no, test_batch in enumerate(it, start=1):  #
                import time

                # 记录开始时间
                start_time = time.time()
                output = model.evaluate(test_batch, nsample)#model生成对应的矩阵，并采样nsample次
                
                print("nsample的时间",nsample)
                #解包输出并调整维度
                """
                samples:模型输出的结果
                c_target：模型输入，二次mask的观测值
                eval_points：=1时要填补的点
                observed_points：只含有原本缺失的mask矩阵，=1的时候是有值的
                """
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # 调整样本维度  【B,10,12,432】
                print("samples",samples.shape)
                c_target = c_target.permute(0, 2, 1)  # 调整目标维度
                eval_points = eval_points.permute(0, 2, 1)  # 调整评估点维度
                observed_points = observed_points.permute(0, 2, 1)  # 调整观测点维度

                samples_median = samples.median(dim=1)
                
                
                
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples_median.values)
                all_gen_samples.append(samples)
                
                # original_mask = torch.where(torch.abs(c_target) < 1e-6, 1, 0)
                # eval_points=eval_points-original_mask
                # adjusted_median_values = samples_median.values * scaler + mean_scaler
                # c_target=c_target*scaler+mean_scaler
                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points) 
                ) * scaler
                mape_current = torch.divide(torch.abs((samples_median.values - c_target)*scaler)
                                            ,(c_target*scaler+mean_scaler)*((c_target*scaler+mean_scaler)>(1e-4)))\
                                    .nan_to_num(posinf=0,neginf=0,nan=0)*eval_points
                                     
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                mape_total += mape_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "mape_total": mape_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
                end_time = time.time()
                # 计算时间差
                time_difference = end_time - start_time

                # 输出结果
                print("执行时间为:", time_difference, "秒")
                #exit()
                logging.info("rmse_total={}".format(np.sqrt(mse_total / evalpoints_total)))
                logging.info("mae_total={}".format(mae_total / evalpoints_total))
                logging.info("mape_total={}".format(mape_total / evalpoints_total))
                logging.info("batch_no={}".format(batch_no))
                # 先尝试一下 拼成完整的值
                imputed_data=c_target*(1-eval_points)+samples_median.values*eval_points # 拼凑之后 完整的数据
                print("imputed_data.shape",imputed_data.shape)
                all_imputed_data.append(imputed_data)
    print("RMSE:", np.sqrt(mse_total / evalpoints_total))
    print("MAE:", mae_total / evalpoints_total)
    print("MAPE:",mape_total / evalpoints_total)
    
    all_gen_samples=torch.cat(all_gen_samples,dim=0)
    all_generated_samples=torch.cat(all_generated_samples,dim=0)
    all_imputed_data = torch.cat(all_imputed_data, dim=0)
    all_target = torch.cat(all_target, dim=0)
    all_evalpoint = torch.cat(all_evalpoint, dim=0)
    all_imputed_data = all_imputed_data * scaler + mean_scaler
    all_target = all_target * scaler + mean_scaler
    all_generated_samples=all_generated_samples*scaler+mean_scaler
    all_gen_samples=all_gen_samples*scaler+mean_scaler
    #MAE
    mse_final = (((all_imputed_data - all_target) * all_evalpoint) ** 2).sum().item()
    mae_final = torch.abs((all_imputed_data - all_target) * all_evalpoint).sum().item()
    eps = 1e-10  # 一个小常数，防止除零
    mape_gen=torch.abs((all_generated_samples-all_target)/(all_target+eps))*all_evalpoint
    all_evalpoint_sum=all_evalpoint.sum()
    mape_before = mape_gen.sum()/all_evalpoint_sum*100
    #CRPS
    CRPS = calc_quantile_CRPS(
                all_target, all_gen_samples, all_evalpoint, mean_scaler, scaler
            )
    
    
    mape_final = torch.sum(torch.abs((all_imputed_data - all_target)*all_evalpoint / (all_target + eps)))
    
    mape_final = torch.sum(torch.abs((all_imputed_data - all_target)/ (all_target + eps)))
    mape_after = torch.abs((all_imputed_data - all_target) / (all_target + eps)) *all_evalpoint
    all_evalpoint_sum=all_evalpoint.sum()
    mape_after = mape_after.sum()/all_evalpoint_sum*100
    
    
    rmse_final = np.sqrt(mse_final / evalpoints_total)
    mae_final = mae_final / evalpoints_total
    mape_final = mape_final / evalpoints_total

    print(f"Final CRPS: {CRPS:.6f}")
    print(f"Final RMSE: {rmse_final:.6f}")
    print(f"Final MAE: {mae_final:.6f}")
    print(f"Final before  MAPE: {mape_before:.6f}")
    print(f"Final  MAPE: {mape_final:.6f}")
    print(f"Final after MAPE: {mape_after:.6f}")
        
    import datetime
    if dataset_name:
        current_date = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"./result/{dataset_name}_{current_date}_CRPS.txt"
        with open(file_name, 'w') as f:
            f.write(f"miss_rate: {miss_rate:.6f}\n")
            f.write(f"Final CRPS: {CRPS:.6f}\n")
            f.write(f"Final RMSE: {rmse_final:.6f}\n")
            f.write(f"Final MAE: {mae_final:.6f}\n")
            f.write(f"Final MAPE: {mape_final:.6f}\n")
            f.write(f"Final before MAPE: {mape_before:.6f}\n")
            f.write(f"Final after MAPE: {mape_after:.6f}\n")
            f.write(f"Testing time: {time_difference:.6f}\n")           
                
            
    return mae_total / evalpoints_total,np.sqrt(mse_total / evalpoints_total),mape_total / evalpoints_total
def get_randmask(observed_mask, min_miss_ratio=0., max_miss_ratio=1.):
    rand_for_mask = torch.rand_like(observed_mask) * observed_mask
    rand_for_mask = rand_for_mask.reshape(-1)
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * (max_miss_ratio-min_miss_ratio) + min_miss_ratio
    num_observed = observed_mask.sum().item()
    num_masked = round(num_observed * sample_ratio)
    rand_for_mask[rand_for_mask.topk(num_masked).indices] = -1

    cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
    return cond_mask


def get_hist_mask(observed_mask, for_pattern_mask=None, target_strategy='hybrid'):
    if for_pattern_mask is None:
        for_pattern_mask = observed_mask
    if target_strategy == "hybrid":
        rand_mask = get_randmask(observed_mask)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.5:
        cond_mask = rand_mask
    else:  # draw another sample for histmask (i-1 corresponds to another sample)
        cond_mask = cond_mask * for_pattern_mask
    return cond_mask


def get_block_mask(observed_mask, target_strategy='block',min_seq = 12,max_seq = 24):
    rand_sensor_mask = torch.rand_like(observed_mask)
    randint = np.random.randint
    sample_ratio = np.random.rand()
    sample_ratio = sample_ratio * 0.15
    mask = rand_sensor_mask < sample_ratio
    
    for col in range(observed_mask.shape[1]):
        idxs = np.flatnonzero(mask[:, col])
        if not len(idxs):
            continue
        fault_len = min_seq
        if max_seq > min_seq:
            fault_len = fault_len + int(randint(max_seq - min_seq))
        idxs_ext = np.concatenate([np.arange(i, i + fault_len) for i in idxs])
        idxs = np.unique(idxs_ext)
        idxs = np.clip(idxs, 0, observed_mask.shape[0] - 1)
        mask[idxs, col] = True
    rand_base_mask = torch.rand_like(observed_mask) < 0.05
    reverse_mask = mask | rand_base_mask
    block_mask = 1 - reverse_mask.to(torch.float32)

    cond_mask = observed_mask.clone()
    mask_choice = np.random.rand()
    if target_strategy == "hybrid" and mask_choice > 0.7:
        cond_mask = get_randmask(observed_mask, 0., 1.)
    else:
        cond_mask = block_mask * cond_mask

    return cond_mask
