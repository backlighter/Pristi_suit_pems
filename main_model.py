import numpy as np
import torch
import torch.nn as nn
from diff_models import Guide_diff


class PriSTI(nn.Module):
    def __init__(self, target_dim, seq_len, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim #目标维度，即数据集中的特征数量
        self.seq_len = seq_len #时间序列的长度

        #从配置中读取模型参数
        self.emb_time_dim = config["model"]["timeemb"] #读取时间嵌入的维度 128
        self.emb_feature_dim = config["model"]["featureemb"]#特征嵌入的维度 16
        self.is_unconditional = config["model"]["is_unconditional"]#是否是无条件模型 parser.add_argument("--unconditional", action="store_true")
        self.target_strategy = config["model"]["target_strategy"] #目标策略 hybrid
        self.use_guide = config["model"]["use_guide"] #true  #这是个啥????

        self.cde_output_channels = config["diffusion"]["channels"] #diffusion 模型channels数目 ##这是个啥
        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim #特征嵌入和时间嵌入的拼接维度
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )   # 创建一个嵌入层，用于将目标维度的数据映射到特征嵌入空间

        config_diff = config["diffusion"] #读取扩散模型的配置
        config_diff["side_dim"] = self.emb_total_dim #设置侧信息的维度
        config_diff["device"] = device 
        self.device = device

        input_dim = 2 #输入维度
        self.diffmodel = Guide_diff(config_diff, input_dim, target_dim, self.use_guide)

        # parameters for diffusion models
        # 设置扩散模型的参数，如步骤数和β系数的调度策略
        self.num_steps = config_diff["num_steps"] #扩散步骤的总数
        if config_diff["schedule"] == "quad": #如果使用二次调度策略
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear": #如果使用线性调度策略
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta  # 计算α_hat系数，用于扩散过程中的噪声调整
        self.alpha = np.cumprod(self.alpha_hat) # 计算α系数的累积乘积，用于调整噪声
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1) # 将α系数转换为PyTorch张量，并调整形状


    def time_embedding(self, pos, d_model=128):
        # 时间嵌入函数，用于将时间位置信息转换为连续的表示
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)  # 初始化位置嵌入张量
        position = pos.unsqueeze(2)  # 增加一个维度以适应除法项的形状
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )  # 计算除法项，用于生成正弦和余弦波形
        pe[:, :, 0::2] = torch.sin(position * div_term)  # 填充偶数位置为正弦波形
        pe[:, :, 1::2] = torch.cos(position * div_term)  # 填充奇数位置为余弦波形
        return pe  # 返回位置嵌入张量


    def get_side_info(self, observed_tp, cond_mask):
        # 获取侧信息函数，用于结合时间嵌入和特征嵌入生成侧信息
        B, K, L = cond_mask.shape  # 从条件掩码中提取批大小、特征数和序列长度

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # 计算时间嵌入
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)  # 扩展时间嵌入的维度以匹配特征嵌入
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # 计算特征嵌入
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # 扩展特征嵌入的维度以匹配时间嵌入
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # 合并时间嵌入和特征嵌入作为侧信息
        side_info = side_info.permute(0, 3, 2, 1)  # 调整侧信息的维度顺序

        return side_info  # 返回侧信息

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train
    ):# 在验证阶段计算损失，遍历所有扩散步骤
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t # 计算所有时间步的损失
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps # 返回平均损失

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info, itp_info, is_train, set_t=-1): #diffusion model
        B, K, L = observed_data.shape  # 提取批大小、特征维数、序列长度
        if is_train != 1:  # 验证模式
            t = (torch.ones(B) * set_t).long().to(self.device)  # 设置所有样本的时间步为set_t
        else:  # 训练模式
            t = torch.randint(0, self.num_steps, [B]).to(self.device)  # 随机选择一个时间步
            
        current_alpha = self.alpha_torch[t]  # 获取当前α值
        noise = torch.randn_like(observed_data)  # 生成噪声
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise  # 生成噪声数据

        # 准备输入数据, 有观测值的地方是0，无观测值的地方是噪声
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)  
        if not self.use_guide:
            itp_info = cond_mask * observed_data  # 如果不使用引导信息，则使用观察到的数据
        predicted = self.diffmodel(total_input, side_info, t, itp_info, cond_mask)  # 使用扩散模型进行预测

        target_mask = observed_mask - cond_mask  # 计算目标掩码
        residual = (noise - predicted) * target_mask  # 计算残差
        num_eval = target_mask.sum()  # 计算需要评估的元素数量
        
        # 计算方差损失
        var_loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        
        # 计算均值损失
        mean_residual = residual.sum() / (num_eval if num_eval > 0 else 1)
        mean_loss = mean_residual ** 2
        
        # 合并损失 # 这个地方可以搜参
        # loss = 0.5 * var_loss + 0.5 * mean_loss  # 可以根据需要调整两者的权重，如：0.5 * var_loss + 0.5 * mean_loss
        loss = var_loss
        return loss


    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        # 准备输入到扩散模型的数据
        # combine the noisy_data and observed_data
        # print(self.is_unconditional, self.use_guide)
        
        if self.is_unconditional == True: # 如果是无条件模型，直接使用噪声数据
            total_input = noisy_data.unsqueeze(1)
        else:
            if not self.use_guide:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)
            else:
                total_input = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        return total_input

    # 数据插补方法
    def impute(self, observed_data, cond_mask, side_info, n_samples, itp_info): #Imputation process with PriSTI

        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):  #计算10次
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                # denosie phase
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    if not self.use_guide:
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                    else:
                        diff_input = ((1 - cond_mask) * current_sample).unsqueeze(1)  # (B,1,K,L)
                predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device), itp_info, cond_mask)

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()#插补出n条
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
            coeffs,
            cond_mask,
        ) = self.process_data(batch)
        """
        observed_data：模型的输入值
        observed_mask：只包含原本缺失的mask
        cond_mask: 二次mask后的mask矩阵
        """
        
        side_info = self.get_side_info(observed_tp, cond_mask)
        itp_info = None
        if self.use_guide:
            itp_info = coeffs.unsqueeze(1)  #线性插补的结果

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        output = loss_func(observed_data, cond_mask, observed_mask, side_info, itp_info, is_train)
        return output

    def evaluate(self, batch, n_samples):
        (
            observed_data, #
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            coeffs,
            _,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask  #目标值，需要评估

            side_info = self.get_side_info(observed_tp, cond_mask)
            itp_info = None
            if self.use_guide:
                itp_info = coeffs.unsqueeze(1)
            #考虑观测值 条件掩码 侧信息 对于每个缺失的数据点，生成插补值的数量  itp_info 是否使用引导信息
            samples = self.impute(observed_data, cond_mask, side_info, n_samples, itp_info)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class PriSTI_survey(PriSTI):
    def __init__(self, config, device, target_dim, seq_len=12):
        super(PriSTI_survey, self).__init__(target_dim, seq_len, config, device)
        self.config = config

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()#(B,K,L)
        observed_mask = batch["observed_mask"].to(self.device).float()#(B,K,L)
        observed_tp = batch["timepoints"].to(self.device).float()#(B,K) #时间点
        gt_mask = batch["gt_mask"].to(self.device).float()#(B,K,L)
        cut_length = batch["cut_length"].to(self.device).long()#用于切割的长度
        coeffs = None
        # 构造连续表示的线性插值系数
        if self.config['model']['use_guide']:
            coeffs = batch["coeffs"].to(self.device).float()
        cond_mask = batch["cond_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)  # [B, K, L]
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        cond_mask = cond_mask.permute(0, 2, 1)
        for_pattern_mask = observed_mask

        if self.config['model']['use_guide']:
            coeffs = coeffs.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            coeffs,
            cond_mask,
        )


