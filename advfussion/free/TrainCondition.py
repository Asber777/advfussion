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
from .Model import UNet as DDPMUNet
from .ModelCondition import UNet
from robustbench import load_model
from advfussion.grad_cam import GradCamPlusPlus, get_last_conv_name
from .utils import add_border, init_seeds
from .logger import log, configure

import warnings
warnings.filterwarnings("ignore")

def eval(modelConfig: Dict, PureConfig: Dict=None):
    init_seeds(modelConfig['seed'])
    configure(modelConfig['save_path'], modelConfig['state'])
    device = torch.device(modelConfig["device"])
    # load dataset
    dataset = CIFAR10(
        root=modelConfig['cifar10path'], train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)

    attack_model = None
    if modelConfig["name"] != None:
        attack_model = load_model(
            model_name=modelConfig["name"], 
            model_dir=modelConfig['robustPath'],
            dataset='cifar10', threat_model=modelConfig['threat_name'])
        attack_model = attack_model.to(device)

    model = UNet(
        T=modelConfig["T"], 
        num_labels=10, 
        ch=modelConfig["channel"], 
        ch_mult=modelConfig["channel_mult"],
        num_res_blocks=modelConfig["num_res_blocks"], 
        dropout=modelConfig["dropout"]).to(device)
    ckpt = torch.load(os.path.join(
        modelConfig["save_dir"], 
        modelConfig["test_load_weight"]), map_location=device)
    model.load_state_dict(ckpt)
    print("model load weight done.")
    model.eval()

    sampler = GaussianDiffusionSampler(
        model, modelConfig["beta_1"], modelConfig["beta_T"], 
        modelConfig["T"], w=modelConfig["w"]).to(device)
    
    if modelConfig['Pure'] or modelConfig['state'] == 'pure':
        with torch.no_grad():
            model = DDPMUNet(
                T=PureConfig["T"], 
                ch=PureConfig["channel"], 
                ch_mult=PureConfig["channel_mult"], 
                attn=PureConfig["attn"],
                num_res_blocks=PureConfig["num_res_blocks"], 
                dropout=0.)
            ckpt = torch.load(PureConfig["test_load_weight"], map_location=device)
            model.load_state_dict(ckpt)
            model.eval()
            Puresampler = GaussianDiffusionSampler(
                model, PureConfig["beta_1"], PureConfig["beta_T"], PureConfig["T"]).to(device)

    print("plz check dir: ", modelConfig['save_path'])
    num = 0
    AS_after_pure = torch.tensor(0.0).to(device)
    natural_err_total = torch.tensor(0.0).to(device)
    target_err_total = torch.tensor(0.0).to(device)
    Lp_total = torch.tensor(0.0).to(device)
    lpips_total = torch.tensor(0.0).to(device)
    ssim_total = torch.tensor(0.0).to(device)

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        mask = None
        if attack_model is not None and modelConfig['state']=='eval':
            layer_name = get_last_conv_name(attack_model)
            grad_cam = GradCamPlusPlus(attack_model, layer_name)
            mask = grad_cam(images).unsqueeze(1)
        model_kwargs = {
            "mask":mask, 
            "modelConfig":modelConfig, 
            "path": modelConfig['save_path'], 
            "images":images, 
            "lpips": None
            }
        if modelConfig['state'] == 'eval':
            with torch.no_grad():
                xt = sampler.get_xt(x_0=images, t=modelConfig['start_T']-1)
                if modelConfig['save_intermediate_result']:
                    save_image(xt, os.path.join(modelConfig['save_path'], f'x_startT-{num}.png'), 
                        nrow=modelConfig["nrow"],  normalize=True, range=(-1, 1))
                
                sampledImgs, acc_cure = sampler(xt, labels+1, attack_model, 
                    modelConfig["start_T"], kwargs=model_kwargs, 
                    show = modelConfig['save_intermediate_result'])
                if modelConfig['save_intermediate_result']: 
                    print(acc_cure)
                if attack_model is not None:
                    pred_y = attack_model(sampledImgs * 0.5 + 0.5).argmax(dim=1)
                    target_err_total += sum(labels!=pred_y)
                    log(f"label:{labels.cpu().numpy()}")
                    log(f"pred_y:{pred_y.cpu().numpy()}")
                    log("subset-AS:", sum(labels!=pred_y)/len(labels))
                    flagsampledImgs = add_border(sampledImgs, labels!=pred_y)
                    save_image(flagsampledImgs, os.path.join(modelConfig['save_path'], f'advx{num}.png'), 
                        nrow=modelConfig["nrow"],  normalize=True, range=(-1, 1))
                    save_image(images, os.path.join(modelConfig['save_path'], f'originalpic{num}.png'), 
                        nrow=modelConfig["nrow"],  normalize=True, range=(-1, 1))
                    torch.save({"sampledImgs":sampledImgs, "labels":labels }, os.path.join(
                        modelConfig['save_path'], f"sampledImgs-{num}.pt"))
        else:
            sampledImgs = torch.load(os.path.join(
                        modelConfig['save_path'], f"sampledImgs-{num}.pt"), map_location=device)
            assert len(sampledImgs) == modelConfig['batch_size']
        # Puring
        if modelConfig['measure'] or modelConfig['state'] == 'pure':
            with torch.no_grad():
                if modelConfig['state'] == 'pure':
                    pred_y = attack_model(sampledImgs * 0.5 + 0.5).argmax(dim=1)
                    target_err_total += sum(labels!=pred_y)
                    log(f"label:{labels.cpu().numpy()}")
                    log(f"pred_y:{pred_y.cpu().numpy()}")
                    log("subset-AS:", sum(labels!=pred_y)/len(labels))

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
                if modelConfig['start_T'] == modelConfig['T']:
                    log("NOTE:sampledImgs are Generated from Gassian")
                log(f"SSIM_BATCH = {ssim_batch.mean().cpu().item()}")
                log(f"LPIPS_BATCH = {lpips_batch.mean().cpu().item()}")
                if modelConfig['Pure']:
                    log(f"----------------Pure Test at {num}-----------------")
                    xt = sampler.get_xt(x_0=sampledImgs, t=modelConfig['start_T']-1)
                    sampledImgs_afterDiffPure = Puresampler(xt, modelConfig["start_T"])
                    pred_y = attack_model(sampledImgs_afterDiffPure * 0.5 + 0.5).argmax(dim=1)
                    log(f"pred_y after pure:{pred_y}")
                    log("AS after DiffPure:", sum(labels!=pred_y)/len(labels))
                    AS_after_pure += sum(labels!=pred_y)
                    save_image(add_border(sampledImgs_afterDiffPure, labels!=pred_y), 
                        os.path.join(modelConfig['save_path'], f'advx_after_pure-{num}.png'), 
                        nrow=modelConfig["nrow"],  normalize=True, range=(-1, 1))
        
        num += len(labels)
        if modelConfig['measure'] or modelConfig['state'] == 'pure':
            log(f'Nature Error total:{natural_err_total}/{num}')
            log(f'Target Error total:{target_err_total}/{num}')
            log(f'Target Error total After Pure:{AS_after_pure}/{num}')
            log(f'Avg L2 distance: {Lp_total / num}')
            log(f'Avg LPIPS dis sum: {lpips_total/target_err_total}')
            log(f'Avg SSIM dis sum : {ssim_total/target_err_total}')
        elif modelConfig['state'] == 'eval':
            log(f'Target Error total:{target_err_total}/{num}')
        if num >= modelConfig['sample_num']:
            break
