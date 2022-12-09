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
from .utils import add_border, init_seeds
from .logger import log, set_logger

import warnings
warnings.filterwarnings("ignore")
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
    set_logger(config['save_path'], config['state'])
    device = torch.device(config["device"])
    dataloader = get_dataloader(config)
    attack_model = load_model(
        model_name=config["name"], 
        model_dir=config['robustPath'],
        dataset='cifar10', threat_model=config['threat_name'])
    attack_model = attack_model.to(device)

    model = UNet(
        T=config["T"], 
        num_labels=10, 
        ch=config["channel"], 
        ch_mult=config["channel_mult"],
        num_res_blocks=config["num_res_blocks"], 
        dropout=config["dropout"]).to(device)
    ckpt = torch.load(os.path.join(
        config["save_dir"], 
        config["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    print("model load weight done.")
    model.eval()

    sampler = GaussianDiffusionSampler(
        model, config["beta_1"], config["beta_T"], 
        config["T"], w=config["w"]).to(device)

    num = 0
    AS_after_pure = torch.tensor(0.0).to(device)
    natural_err_total = torch.tensor(0.0).to(device)
    target_err_total = torch.tensor(0.0).to(device)
    Lp_total = torch.tensor(0.0).to(device)
    lpips_total = torch.tensor(0.0).to(device)
    ssim_total = torch.tensor(0.0).to(device)
    save_path = lambda fname: os.path.join(
        config['save_path'], fname)
    save_img = lambda img, name: save_image(img, 
        save_path(name), nrow=config["nrow"],  
        normalize=True, range=(-1, 1))
    layer_name = get_last_conv_name(attack_model)
    grad_cam = GradCamPlusPlus(attack_model, layer_name)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        mask = grad_cam(images).unsqueeze(1)
        model_kwargs = {
            "mask":mask, 
            "modelConfig":config, 
            "path": config['save_path'], 
            "images":images, 
            "lpips": None
            }
        with torch.no_grad():
            xt = sampler.get_xt(x_0=images, t=config['start_T']-1)
            sampledImgs, acc_cure = sampler(xt, labels+1, attack_model, 
                config["start_T"], kwargs=model_kwargs, 
                show = config['save_intermediate_result'])
            if config['save_intermediate_result']: log(acc_cure)

            pred_y = attack_model(sampledImgs * 0.5 + 0.5).argmax(dim=1)
            target_err_total += sum(labels!=pred_y)
            log(f"label:{labels.cpu().numpy()}")
            log(f"pred_y:{pred_y.cpu().numpy()}")
            log("subset-AS:", sum(labels!=pred_y)/len(labels))
            flagsampledImgs = add_border(sampledImgs, labels!=pred_y)
            
            save_img(flagsampledImgs, f'advx{num}.png')
            save_img(images, f'originalpic{num}.png')
            torch.save({"sampledImgs":sampledImgs, "labels":labels }, 
                save_path(f"sampledImgs-{num}.pt"))

        if config['measure']:
            with torch.no_grad():
                pred_y = attack_model(images * 0.5 + 0.5).argmax(dim=1)
                natural_err_total += sum(labels!=pred_y)
                Lp_total += ((images-sampledImgs) ** 2).sum(dim=(1, 2, 3)).sqrt().sum()

                loss_fn_alex = lpips.LPIPS(net='alex')
                loss_fn_alex = loss_fn_alex.to(device)

                lpips_batch = loss_fn_alex.forward(images, sampledImgs)
                lpips_total += lpips_batch.sum()
                ssim_batch = ssim((images+1)/2, (sampledImgs+1)/2, 
                    data_range=1., size_average=False)
                ssim_total += ssim_batch.sum()
                log(f"SSIM_BATCH = {ssim_batch.mean().cpu().item()}")
                log(f"LPIPS_BATCH = {lpips_batch.mean().cpu().item()}")

        num += len(labels)
        if config['measure']:
            log(f'Nature Error total:{natural_err_total}/{num}')
            log(f'Target Error total:{target_err_total}/{num}')
            log(f'Target Error total After Pure:{AS_after_pure}/{num}')
            log(f'Avg L2 distance: {Lp_total / num}')
            log(f'Avg LPIPS dis sum: {lpips_total/target_err_total}')
            log(f'Avg SSIM dis sum : {ssim_total/target_err_total}')
        if num >= config['sample_num']:
            break



from .pure.Model import UNet as PureUNet
from .pure.Diffusion import GaussianDiffusionSampler as PureSammpler
from tqdm import tqdm
def pure(config, pconfig):
    device = torch.device(config["device"])
    with torch.no_grad():
        model = PureUNet(
            T=pconfig["T"], 
            ch=pconfig["channel"], 
            ch_mult=pconfig["channel_mult"], 
            attn=pconfig["attn"],
            num_res_blocks=pconfig["num_res_blocks"], 
            dropout=0.)
        ckpt = torch.load(pconfig["test_load_weight"], map_location=device)
        model.load_state_dict(ckpt)
        model.eval()
        Puresampler = PureSammpler(
            model, pconfig["beta_1"], pconfig["beta_T"], pconfig["T"]).to(device)

    for num in tqdm(range(config['sample_num']/config['batch_size'])):
        sampledImgs = torch.load(os.path.join(
                    config['save_path'], f"sampledImgs-{num}.pt"), map_location=device)
        xt = Puresampler.get_xt(x_0=sampledImgs['sampledImgs'], t=config['start_T']-1)
        x = Puresampler(xt, config["start_T"])
        torch.save(x, os.path.join(config['save_path'],
            f"sampledImgs-pure-{num}.pt"))
            
        # pred_y = attack_model(x * 0.5 + 0.5).argmax(dim=1)
        # log(f"pred_y after pure:{pred_y}")
        # log("AS after DiffPure:", sum(labels!=pred_y)/len(labels))
        # AS_after_pure += sum(labels!=pred_y)
        # save_image(add_border(x, labels!=pred_y), 
        #     os.path.join(config['save_path'], f'advx_after_pure-{num}.png'), 
        #     nrow=config["nrow"],  normalize=True, range=(-1, 1))