import os
from typing import Dict
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
import lpips 
from pytorch_msssim import ssim
from .DiffusionCondition import GaussianDiffusionSampler
from .ModelCondition import UNet
from robustbench import load_model
from advfussion.grad_cam import GradCamPlusPlus, get_last_conv_name
from .utils import add_border, init_seeds, _get_norm_batch
from advfussion.logger import log, configure

import warnings
warnings.filterwarnings("ignore")
NROW = 10

def get_dataloader(config):
    dataset = CIFAR10(
        root=config['cifar10path'], train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ]))
    dataloader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)
    return dataloader

def eval(config: Dict):
    init_seeds(config['seed'])
    configure(config['save_dir'], )
    device = torch.device(config["device"])
    dataloader = get_dataloader(config)
    attack_model = load_model(
        model_name=config["name"], 
        model_dir=config['robustPath'],
        dataset='cifar10', 
        threat_model=config['threat_name'])
    attack_model = attack_model.to(device)
    
    save_path = lambda fname: os.path.join(
        config['result_dir'], fname)
    save_img = lambda img, name: save_image(img, 
        save_path(name), nrow=NROW,  
        normalize=True, range=(-1, 1))
    
    model = UNet(
        T=config["T"], 
        num_labels=10, 
        ch=config["channel"], 
        ch_mult=config["channel_mult"],
        num_res_blocks=config["num_res_blocks"], 
        dropout=config["dropout"]).to(device)
    ckpt = torch.load(config["weight"], 
                      map_location=device)
    model.load_state_dict(ckpt)
    print("model load weight done.")
    model.eval()

    sampler = GaussianDiffusionSampler(
        model, config["beta_1"], config["beta_T"], 
        config["T"], w=config["w"]).to(device)

    num = 0
    AS_after_pure = torch.tensor(0.0).to(device)
    target_err_total = torch.tensor(0.0).to(device)
    L2_total = torch.tensor(0.0).to(device)
    lpips_total = torch.tensor(0.0).to(device)
    ssim_total = torch.tensor(0.0).to(device)

    layer_name = get_last_conv_name(attack_model)
    grad_cam = GradCamPlusPlus(attack_model, layer_name)

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(device)
    model_kwargs = {
            "mask":None, 
            "ts":config["ts"],
            "te":config['te'],
            'adver_scale':config['adver_scale'],
            'nb_iter_conf':config['nb_iter_conf'],
            # 'contrastive':config['contrastive'], 
            }
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            model_kwargs['mask'] = grad_cam(images * 0.5 + 0.5).unsqueeze(1)
            xt = sampler.get_xt(x_0=images, t=config['start_T']-1)
            sampledImgs = sampler(xt, labels, attack_model, 
                config["start_T"], kwargs=model_kwargs)
            pred_y = attack_model(sampledImgs * 0.5 + 0.5).argmax(dim=1)
            
            error_flag = labels!=pred_y
            target_err_total += sum(error_flag)
            if error_flag.sum() != 0:
                L2_total += _get_norm_batch((images[error_flag]-((sampledImgs+1)/2)[error_flag]), 2).sum()
                lpips_batch = loss_fn_alex.forward(images, sampledImgs)
                lpips_total += lpips_batch[error_flag].sum()
                ssim_batch = ssim((images+1)/2, (sampledImgs+1)/2, 
                    data_range=1., size_average=False)
                ssim_total += ssim_batch[error_flag].sum()
            num += len(labels)
            log(f"label:{labels.cpu().numpy()}")
            log(f"pred_y:{pred_y.cpu().numpy()}")
            log("subset-AS:", sum(error_flag)/len(labels))
            log(f'Robust Acc:{target_err_total}/{num}')
            log(f'Robust Acc After Pure:{AS_after_pure}/{num}')
            log(f'Avg L2 distance: {L2_total / target_err_total}')
            log(f'Avg LPIPS dis sum: {lpips_total/target_err_total}')
            log(f'Avg SSIM dis sum : {ssim_total/target_err_total}')

            torch.save(
                {
                "advx":sampledImgs, 
                "pred_y":pred_y,
                "y":labels,
                "x":images,
                }, 
                save_path(f"sampledImgs-{num}.pt"))
            flagsampledImgs = add_border(sampledImgs, error_flag)
            save_img(flagsampledImgs, f'advx{num}.png')
            save_img(images, f'originalpic{num}.png')
            
            if num >= config['sample_num']:
                break