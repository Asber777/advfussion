
import os
import math
import lpips
import shutil
import time
import torch as th
import os.path as osp
import torch.distributed as dist
from torchvision import utils
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim
from robustbench.utils import load_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from guided_diffusion.data_model.my_loader import MyCustomDataset
from guided_diffusion.resizer import Resizer
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import save_args, create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults, diffuson_pgd, seed_torch, DT
from guided_diffusion.grad_cam import GradCamPlusPlus, get_last_conv_name
from guided_diffusion.myargs import create_argparser



def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    dir = osp.join(args.result_dir, args.describe, DT(),)
    logger.configure(dir)
    save_args(logger.get_dir(), args)
    shutil.copy(os.path.realpath(__file__), logger.get_dir())

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16: model.convert_to_fp16()
    model.eval()
    
    data = MyCustomDataset(img_path="/root/hhtpro/123/GA-Attack-main/data/images")
    sampler = th.utils.data.distributed.DistributedSampler(data)
    attack_loader = th.utils.data.DataLoader(dataset=data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=2, pin_memory=True)

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev()).eval()
    layer_name = get_last_conv_name(attack_model)
    grad_cam = GradCamPlusPlus(attack_model, layer_name)

    resizers = None
    if args.use_ilvr:
        assert math.log(args.down_N, 2).is_integer()
        shape = (args.batch_size, 3, args.image_size, args.image_size)
        shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
        down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
        up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
        resizers = (down, up) # 现在把ilvr放在gaussian diffusion文件里执行

    def cond_fn(x, t, y=None, guide_x=None, guide_x_t=None, 
            mean=None, log_variance=None,  pred_xstart=None, mask=None, threshold=None, **kwargs):
        '''
        x: x_{t+1}
        mean: x_{t}
        guide_x: pgd_x0
        guide_x_t: pgd_x0_t
        mean = mean.float() +  kwargs['variance'] *  gradient.float() # kwargs['variance'] 感觉可以变成常量?
        '''
        time = int(t[0].detach().cpu()) # using this variable in add_scalar will GO WRONG!
        if args.use_adver and time < args.range_t2:
            maks = model_kwargs['mask'].detach().clone()
            delta = th.zeros_like(x)
            with th.enable_grad():
                delta.requires_grad_()
                tmpx = pred_xstart.detach().clone() + delta *(1-maks) # range from -1~1

                attack_logits = attack_model(th.clamp((tmpx+1)/2.,0.,1.)) 
                target = th.where(attack_logits.argmax(dim=1)==y, y, attack_logits.argmin(dim=1))
                sign = th.where(attack_logits.argmax(dim=1)==y, 1, -1)
                selected = sign * attack_logits[range(len(attack_logits)), target.view(-1)] 
                loss = -selected.sum() * args.adver_scale 
                loss.backward()
                grad_sign = delta.grad.data.detach().clone() *(1-maks)
                # delta.grad.data.zero_()
            mean = mean.float() + grad_sign.float()
            out_path = os.path.join(logger.get_dir(), f"grad_sign{str(time).zfill(5)}.png")
            utils.save_image(grad_sign, out_path, nrow=args.batch_size, 
            normalize=True, range=(-1, 1),)
        return mean

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(dist_util.dev())

    logger.log("creating samples...")
    count = 0
    natural_err_total = th.tensor(0.0).cuda()
    # pgd_err_total = th.tensor(0.0).cuda() # source model 's acc
    target_err_total = th.tensor(0.0).cuda()
    eps_total = th.tensor(0.0).cuda()
    quality_level = th.tensor(0.0).cuda()
    lpips_total = th.tensor(0.0).cuda()
    ssim_total = th.tensor(0.0).cuda()
    reward_fn = lambda x: 1.0 / (x)

    seed_torch(args.seed)
    time_start = time.time()
    for i, (img, label, img_name) in enumerate(attack_loader):
        count += len(img)
        img, label = img.to(dist_util.dev()), label.to(dist_util.dev())
        mask = grad_cam(img).unsqueeze(1) if args.use_cam else None
        # get natural_err_total
        with th.no_grad():
            err = (attack_model(img).data.max(1)[1] != label.data).float().sum()
            natural_err_total += err
        x = img.clone().detach()*2-1
        if args.use_adver:
            x = diffuson_pgd(x, label, attack_model, nb_iter=args.nb_iter,)
        # begin sample
        model_kwargs = {
            "guide_x":x, "y":label, "mask":mask, "resizer":resizers,
            "range_t1":args.range_t1, "threshold":args.threshold
        }
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
        budget = th.abs((sample+1)/2 - img).reshape(len(sample), -1).max(dim = -1)[0]

        time_end = time.time()

        at_predict = attack_model((sample+1)/2).argmax(dim=-1)
        err_mask = (at_predict.data != label.data)
        target_err_total += err_mask.float().sum()

        distance_batch = budget[err_mask]
        lpips_batch = loss_fn_alex.forward(x, sample)[err_mask]
        ssim_batch = ssim(img, (sample+1)/2, data_range=1., size_average=False)[err_mask]
        eps_total += distance_batch.sum()
        batch_score = reward_fn(distance_batch)
        quality_level += batch_score.sum() 
        lpips_total += lpips_batch.sum()
        ssim_total += ssim_batch.sum()

        logger.log(f"created {count} samples")
        logger.log(f'original label of sample: {label.cpu().numpy()},')
        logger.log(f'predict of attack_model: {at_predict.cpu().numpy()}, ')
        logger.log(f'time cost:{time_end-time_start} s')
        logger.log(f'Nature Error total:{natural_err_total}/{count}')
        # logger.log(f'Source Success total:{pgd_err_total}')
        logger.log(f'Target Success total:{target_err_total}/{count}')
        logger.log(f'Avg distance of successfully transferred: {eps_total / target_err_total}')
        logger.log(f'Avg perturbation reward: {quality_level / target_err_total}')
        logger.log(f'Avg LPIPS dis: {lpips_total / target_err_total}')
        logger.log(f'Avg SSIM dis: {ssim_total / target_err_total}')

        out_path = os.path.join(logger.get_dir(), f"{str(count).zfill(5)}.png")
        utils.save_image(th.cat([sample, x], dim=0), out_path, nrow=args.batch_size, 
            normalize=True, range=(-1, 1),)
        if count>= args.stop_count:
            break

    dist.barrier()
    logger.log("sampling complete")
    grad_cam.remove_handlers()

if __name__ == "__main__":
    main()
