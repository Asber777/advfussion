import os
import argparse
from typing import Dict
import numpy as np
import random
import torch
import torch.backends.cudnn as cudnn
from advfussion import logger
from advfussion.free.pure.Model import UNet
from advfussion.free.pure.Diffusion import GaussianDiffusionSampler
from tqdm import tqdm
from glob import glob
from robustbench import load_model
from torchvision.utils import save_image
from advfussion.script_util import save_args, add_dict_to_argparser
from advfussion.logger import log, configure

def seed_torch(seed=1029,cuda_deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # Speed-reproducibility tradeoff : https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def eval(modelConfig: Dict):
    seed_torch(modelConfig['seed'])
    configure(modelConfig['eval_path'], 
              log_suffix='pure_uncond')
    device = torch.device(modelConfig["device"])
    attack_model = load_model(
        model_name=modelConfig["name"], 
        model_dir=modelConfig['robustPath'],
        dataset='cifar10', 
        threat_model=modelConfig['threat_name']
        )
    attack_model = attack_model.to(device)
    pure_dir = os.path.join(modelConfig['eval_path'], 
                            'pure_uncond')
    os.makedirs(pure_dir, exist_ok=True)
    AS_after_pure = torch.tensor(0.0).to(device)
   
    with torch.no_grad():
        model = UNet(T=modelConfig["T"], 
                     ch=modelConfig["channel"], 
                     ch_mult=modelConfig["channel_mult"], 
                     attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], 
                     dropout=0.)
        ckpt = torch.load(modelConfig["test_load_weight"], 
                          map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, 
            modelConfig["beta_1"], 
            modelConfig["beta_T"], 
            modelConfig["T"]
            ).to(device)
    
        img_files = glob(os.path.join(
                        modelConfig['eval_path'],
                        modelConfig['image_dir_name'],
                        '*.pt'))
        for img_file in img_files:
            pt = torch.load(img_file, map_location=device)
            advx, labels = pt['xadv'], pt['y']
            xt = sampler.get_xt(x_0=(advx*2-1), t=modelConfig['start_T']-1)
            sampledImgs = sampler(xt, modelConfig["start_T"])
            pred_y = attack_model(sampledImgs * 0.5 + 0.5).argmax(dim=1)
            AS_after_pure += sum(labels!=pred_y)
            name = img_file.split('/')[-1].split('.')[0]
            log(f"AS after DiffPure:{AS_after_pure}, name:{name}")
            save_image(sampledImgs, os.path.join(pure_dir, f"{name}.png"), 
                nrow = 5, normalize=True, range=(-1, 1),)
            pt_path = os.path.join(f'{pure_dir}/{name}.pt')
            torch.save({'sample':sampledImgs*0.5+0.5, "x":advx, 
                        'y':labels, "predict":pred_y}, pt_path)
            
if __name__ == '__main__':
    modelConfig = {
        "batch_size": 50,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda",
        "test_load_weight": "/root/hhtpro/123/models/ddpm_free/DiffusionWeight.pt",
        'seed': 8 ,
        'name': "Rebuffi2021Fixing_70_16_cutmix_extra", 
        "robustPath":"/root/hhtpro/123/models",
        "threat_name":"L2", 
        "eval_path":"/root/hhtpro/123/result_of_cifar10_exp/GA/max_epsilon-160",  
        "start_T":100,
        "image_dir_name":'image',
        }
    parser = argparse.ArgumentParser() 
    add_dict_to_argparser(parser, modelConfig)
    args = parser.parse_args()
    modelConfig = vars(args)
    logger.configure(args.eval_path, log_suffix=f'pure_{args.start_T/ args.T}')
    save_args(logger.get_dir(), args)
    eval(modelConfig)