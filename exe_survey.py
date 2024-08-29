import argparse
import logging
import torch
import datetime
import json
import yaml
import os
import numpy as np
import nni
from dataset_survey_cr import get_dataloader
from main_model import PriSTI_survey
from utils import train, evaluate
import nni

def load_nni_parameter(config):
    params = nni.get_next_parameter()
    for key,value in params.items():
        for section in config.keys():
            if key in config[section].keys():
                config[section][key] = value

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def main(args):
    SEED = args.seed
    print("SEED",SEED)
    setup_seed(SEED)

    path = args.config
    with open(path, "r") as f:#加载配置文件
        config = yaml.safe_load(f)
    config['train']['nni'] = args.use_nni
    if config['train']['nni']:
        load_nni_parameter(config)

    config["model"]["is_unconditional"] = args.unconditional
    config["model"]["target_strategy"] = args.targetstrategy
    config["seed"] = ""
    
    print(json.dumps(config, indent=4))

    data_prefix = config['file']['data_prefix']
    miss_type = config['file']['miss_type']
    miss_rate = float(config['file']['miss_rate'])
    dataset = config['file']['dataset']

    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    foldername = (
        "./save/" + dataset+ '_'+ miss_type +'_'+str(miss_rate) + '_' + current_time + "/"
    )
    #新增了all_loader
    train_loader, valid_loader, test_loader, scaler, mean_scaler,node_num ,all_loader= get_dataloader(
        config["train"]["batch_size"], device=args.device, missing_pattern=args.missing_pattern,
        is_interpolate=config["model"]["use_guide"], num_workers=args.num_workers,
        target_strategy=args.targetstrategy,data_prefix=data_prefix,miss_type=miss_type,miss_rate=miss_rate
    )
    config["diffusion"]["node_num"] = node_num

    model = PriSTI_survey(config, args.device,target_dim = node_num).to(args.device)
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    print('model folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    if args.modelfolder == "":# 如果没有使用model
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
        )
    else:
        model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))

    logging.basicConfig(filename=foldername + '/test_model.log', level=logging.DEBUG)
    logging.info("model_name={}".format(args.modelfolder))
    dataset_name=dataset
    test_mae,_,_= evaluate(
        model,
        test_loader,
        nsample=args.nsample,
        scaler=scaler,
        mean_scaler=mean_scaler,
        foldername=foldername,
        dataset_name=dataset_name,
        miss_rate=miss_rate
        )
    # if args.modelfolder == "":
    #     test_mae,_,_,_ = evaluate(
    #     model,
    #     test_loader,
    #     nsample=args.nsample,
    #     scaler=scaler,
    #     mean_scaler=mean_scaler,
    #     foldername=foldername,
    #     )
    # else:# 这个地方把test_loader换成全量数据
    #     print("使用全量模型进行测试")
        # test_mae,_,_,_ = evaluate(
        # model,
        # all_loader,
        # nsample=args.nsample,
        # scaler=scaler,
        # mean_scaler=mean_scaler,
        # foldername=foldername,
        # )
    if config['train']['nni']:
        nni.report_final_result(test_mae)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PriSTI")
    parser.add_argument("--config", type=str, default="./config/pems08.yaml")
    parser.add_argument('--device', default='cuda:1', help='Device for Attack')
    parser.add_argument('--num_workers', type=int, default=4, help='Device for Attack')
    parser.add_argument("--modelfolder", type=str, default="")
    parser.add_argument(
        "--targetstrategy", type=str, default="hybrid", choices=["mix", "random", "block"]
    )
    parser.add_argument("--nsample", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconditional", action="store_true") 
    '''
    #这行代码的作用是定义了一个
    #--unconditional的命令行参数 ，
    #当用户在命令行中 指定--unconditional时 解析器会将其设置为True
    '''
    parser.add_argument("--missing_pattern", type=str, default="block")     # block|point
    parser.add_argument("--use_nni", action="store_true")
    args = parser.parse_args()
    print(args)

    main(args)
