
import os
import math
import lpips
import shutil
# import time
import cv2
import numpy as np
from skimage import io
import torch as th
import os.path as osp
import torch.distributed as dist
from torchvision import utils
from torch import clamp
# from torchvision import transforms
# from pytorch_msssim import ssim, ms_ssim
from robustbench.utils import load_model
# from torchvision.utils import make_grid
from advfussion.data_model.my_loader import MyCustomDataset
# from guided_diffusion.resizer import Resizer
from advfussion import dist_util, logger
from advfussion.script_util import save_args, create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults, diffuson_pgd, seed_torch, DT, add_border
from advfussion.grad_cam import GradCamPlusPlus, get_last_conv_name
from advfussion.myargs import create_argparser


def main():
    args = create_argparser().parse_args()
    args.describe = 'Ablation-CAM'
    args.start_t = 20
    args.stop_count = 60
    dist_util.setup_dist()
    dir = osp.join(args.result_dir, args.describe, DT(),)
    th.cuda.empty_cache()
    logger.configure(dir)
    result_dir = osp.join(logger.get_dir(), 'result')
    os.makedirs(result_dir, exist_ok=True)
    save_args(logger.get_dir(), args)

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16: model.convert_to_fp16()
    model.eval()
    
    data = MyCustomDataset(img_path=args.ImageNetpath)
    attack_loader = th.utils.data.DataLoader(dataset=data,
                                                batch_size=args.batch_size,
                                                shuffle=False,num_workers=2, pin_memory=True)
    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev()).eval()
    layer_name = get_last_conv_name(attack_model)
    grad_cam = GradCamPlusPlus(attack_model, layer_name)

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(dist_util.dev())
    logger.log("creating samples...")
    seed_torch(args.seed)
    for i, (img, label, img_name) in enumerate(attack_loader):
        img, label = img.to(dist_util.dev()), label.to(dist_util.dev())
        mask = grad_cam(img).unsqueeze(1) if args.use_cam else None
        x = img.clone().detach()*2-1
        model_kwargs = {"guide_x":x, "y":label, "mask":mask, }
        for j in range(0, 3):
            def cond_fn(x, t, y=None, guide_x=None, guide_x_t=None, 
                    mean=None, log_variance=None,  pred_xstart=None, 
                    mask=None, threshold=None, early_stop=True, **kwargs):
                time = int(t[0].detach().cpu()) # using this variable in add_scalar will GO WRONG!
                if args.use_adver and args.range_t2_s <=time <= args.range_t2_e:
                    maks = mask.detach().clone()
                    eps = th.exp(0.5 * log_variance)
                    delta = th.zeros_like(x)
                    with th.enable_grad():
                        delta.requires_grad_()
                        for _ in range(args.nb_iter_conf):
                            tmpx = pred_xstart.detach().clone() + delta  # range from -1~1
                            attack_logits = attack_model(th.clamp((tmpx+1)/2.,0.,1.)) 
                            if early_stop:
                                target = y
                                sign = th.where(attack_logits.argmax(dim=1)==y, 1, 0)
                            else:
                                target = th.where(attack_logits.argmax(dim=1)==y, y, attack_logits.argmin(dim=1))
                                sign = th.where(attack_logits.argmax(dim=1)==y, 1, -1)
                            selected = sign * attack_logits[range(len(attack_logits)), target.view(-1)] 
                            loss = -selected.sum()
                            loss.backward()
                            grad_ = delta.grad.data.detach().clone()
                            delta.data += grad_  * args.adver_scale  *(1-maks)**args.mask_p
                            delta.data = clamp(delta.data, -eps, eps)
                            delta.grad.data.zero_()
                        # delta.grad.data.zero_()
                        th.save(delta, os.path.join(result_dir, f"delta{i}-{j}-{time}.pt"))
                    mean = mean.float() + delta.data.float() 
                return mean
            sample = diffusion.p_sample_loop(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if args.use_adver else None,
                start_t=args.start_t if args.use_half else None,
                device=dist_util.dev(),
                progress=True,
            )
            sample = th.clamp(sample,-1.,1.)
            out_path = os.path.join(logger.get_dir(), f"result{i}-{j}.png")
            pt_path = os.path.join(result_dir, f"result{i}-{j}.pt")
            utils.save_image(th.cat([sample, x], dim=0), out_path, 
                nrow=len(sample), normalize=True, range=(-1, 1),)
            th.save({'sample':sample, "x":x, 'y':label,"heatmap":mask}, pt_path)
        if i > 10:
            break
    dist.barrier()
    logger.log("sampling complete")
    grad_cam.remove_handlers()

if __name__ == "__main__":
    main()
