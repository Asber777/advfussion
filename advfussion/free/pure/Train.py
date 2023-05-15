
import os
from typing import Dict
import numpy as np
import random
import torch
from tqdm import tqdm
from robustbench import load_model
from torchvision.utils import save_image
import torch.backends.cudnn as cudnn
from advfussion.free.pure.Model import UNet
from advfussion.free.pure.Diffusion import GaussianDiffusionSampler
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
        for i in tqdm(range(0, 1000, modelConfig['batch_size'])):
            pt = torch.load(os.path.join(
                        modelConfig['eval_path'],'result', f"sampledImgs-{i}.pt"), 
                        map_location=device)
            advx, labels = pt['advx'], pt['y']
            xt = sampler.get_xt(x_0=advx, t=modelConfig['start_T']-1)
            sampledImgs = sampler(xt, modelConfig["start_T"])
            pred_y = attack_model(sampledImgs * 0.5 + 0.5).argmax(dim=1)
            AS_after_pure += sum(labels!=pred_y)
            log(f"AS after DiffPure:{AS_after_pure}")
            save_image(sampledImgs, os.path.join(pure_dir, f"{str(i)}.png"), 
                nrow = 5, normalize=True, range=(-1, 1),)
            pt_path = os.path.join(f'{pure_dir}/{str(i)}.pt')
            torch.save({'sample':sampledImgs, "x":advx, 
                        'y':labels, "predict":pred_y}, pt_path)