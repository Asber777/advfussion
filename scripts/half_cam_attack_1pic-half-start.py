
import os
import math
import lpips
import shutil
# import time
import torch as th
import os.path as osp
import torch.distributed as dist
from torchvision import utils
from torch import clamp
# from torchvision import transforms
# from pytorch_msssim import ssim, ms_ssim
from robustbench.utils import load_model
# from torchvision.utils import make_grid
from guided_diffusion.data_model.my_loader import MyCustomDataset
# from guided_diffusion.resizer import Resizer
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import save_args, create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults, diffuson_pgd, seed_torch, DT, add_border
from guided_diffusion.grad_cam import GradCamPlusPlus, get_last_conv_name
from guided_diffusion.myargs import create_argparser

'''
冗余代码 只是为了获取某张图片攻击时候的详情
'''


def main():
    args = create_argparser().parse_args()
    args.describe = 'Half-Start'
    dist_util.setup_dist()
    dir = osp.join(args.result_dir, args.describe, DT(),)
    logger.configure(dir)
    result_dir = osp.join(logger.get_dir(), 'result')
    os.makedirs(result_dir, exist_ok=True)
    save_args(logger.get_dir(), args)
    shutil.copy(os.path.realpath(__file__), logger.get_dir())
    shutil.copy('/root/hhtpro/123/guided-diffusion/guided_diffusion/gaussian_diffusion.py', logger.get_dir())

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16: model.convert_to_fp16()
    model.eval()
    
    # data = OnePicDataset("/root/hhtpro/123/GA-Attack-main/data/images", '44.png')
    # args.batch_size = 1
    data = MyCustomDataset(img_path="/root/hhtpro/123/GA-Attack-main/data/images")
    attack_loader = th.utils.data.DataLoader(dataset=data,
                                                batch_size=args.batch_size,
                                                shuffle=False,num_workers=2, pin_memory=True)
    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev()).eval()
    layer_name = get_last_conv_name(attack_model)
    grad_cam = GradCamPlusPlus(attack_model, layer_name)

    def cond_fn(x, t, y=None, guide_x=None, guide_x_t=None, 
            mean=None, log_variance=None,  pred_xstart=None, 
            mask=None, threshold=None, early_stop=True, **kwargs):
        '''
        x: x_{t+1}
        mean: x_{t}
        guide_x: pgd_x0
        guide_x_t: pgd_x0_t
        mean = mean.float() +  kwargs['variance'] *  gradient.float() # kwargs['variance'] 感觉可以变成常量?
        '''
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
                    # delta.data = clamp(pred_xstart.data + delta.data, -1, +1) - pred_xstart.data
                    delta.grad.data.zero_()
                # delta.grad.data.zero_()
            mean = mean.float() + delta.data.float() 
            # out_path = os.path.join(logger.get_dir(), f"delta{str(time).zfill(5)}.png")
            # utils.save_image(delta*10, out_path, nrow=args.batch_size, 
            #     normalize=True, range=(-0.5, 0.5),)
        return mean

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
        # import cv2
        # import numpy as np
        # from skimage import io
        # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        # heatmap = np.float32(heatmap) / 255
        # heatmap = heatmap[..., ::-1]  # gbr to rgb
        # io.imsave(os.path.join(result_dir, 'heatmap.jpg'), (heatmap * 255).astype(np.uint8))
        x = img.clone().detach()*2-1
        if args.use_adver:
            x = diffuson_pgd(x, label, attack_model, nb_iter=args.nb_iter,)
        model_kwargs = {
            "guide_x":x, "y":label, "mask":mask,# "resizer":resizers,
        }
        for j in range(25, args.timestep_respacing[0]+1,25):
            sample = diffusion.p_sample_loop(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn if args.use_adver else None,
                start_t=j if args.use_half else None,
                device=dist_util.dev(),
                progress=True,
            )
            sample = th.clamp(sample,-1.,1.)
            at_predict = attack_model((sample+1)/2).argmax(dim=-1)
            logger.log(f'original label of sample: {label.cpu().numpy()},')
            logger.log(f'predict of attack_model: {at_predict.cpu().numpy()}, ')
            out_path = os.path.join(logger.get_dir(), f"result{j}.png")
            pt_path = os.path.join(result_dir, f"result{j}.pt")
            utils.save_image(th.cat([sample, x], dim=0), out_path, 
                nrow=len(sample), normalize=True, range=(-1, 1),)
            th.save({'sample':sample, "x":x, 'y':label, "predict":at_predict}, pt_path)
        break
    dist.barrier()
    logger.log("sampling complete")
    grad_cam.remove_handlers()

if __name__ == "__main__":
    main()
