import argparse
import os
import shutil
import os.path as osp
import torch.distributed as dist
import lpips
import torch.nn.functional as F
from robustbench.utils import load_model
from guided_diffusion.resizer import Resizer
from pytorch_msssim import ssim, ms_ssim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import *
# from guided_diffusion.image_datasets import load_data
from torchvision import utils
import math
from time import time

'''
加上了cond_fn来引导和原图长得像
'''

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    
    dir_path = osp.join(args.result_dir, args.describe, DT(),)
    logger.configure(dir_path)
    
    save_args(logger.get_dir(), args)
    shutil.copy(os.path.realpath(__file__), logger.get_dir())
    
    # Create DDPM Components
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16: model.convert_to_fp16()
    model.eval()

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu"))
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16: classifier.convert_to_fp16()
    classifier.eval()

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev())

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(dist_util.dev())

    data = load_imagenet_batch(args.batch_size, '/root/hhtpro/123/imagenet')
    writer = SummaryWriter(logger.get_dir())

    map_i_s = get_idex2name_map(INDEX2NAME_MAP_PATH)
    mapname = lambda predict: [map_i_s[i] for i in predict.cpu().numpy()]

    assert math.log(args.down_N, 2).is_integer()
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    count = 0
    attack_succ_sum = 0
    attack_clas_succ_sum = 0
    time_sum = 0
    lpips_sum = 0
    ssim_sum = 0
    seed_torch(args.seed)

    def back_fn(x, guide_x=None,  time=None):
        '''
        x is original varia out["sample"]
        y is original label
        guide_x is what we wish it looks like
        guide_y is what we wish attack model predict is 
        time is time.
        '''
        assert guide_x is not None
        assert time is not None
        if resizers is not None:
            down, up = resizers
            if time > args.range_t1: 
                x = x - up(down(x)) + up(down(guide_x))
        return x

    def cond_fn(x, t, y=None, guide_x=None, guide_y=None, mean=None, guide_x_t=None, **kwargs):
        time = int(t[0].detach().cpu()) # using this variable in add_scalar will GO WRONG!
        mean = back_fn(mean, guide_x_t, time)

        with th.enable_grad():
            x_in = mean.detach().requires_grad_(True)# "mean" used to be "x" 

            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)] 

            loss = selected.sum() * args.generate_scale

            writer.add_scalar(f"pic{count}/lossofclassifier",selected.sum(),time)

            # # range from 0~1 but x_in maybe not
            ssim_val = ssim((guide_x_t+1)/2, (x_in+1)/2, 
                data_range=1, size_average=True)
            loss += ssim_val * args.ssim_scale
            writer.add_scalar(f"pic{count}/ssim",ssim_val, time)

            # # range from -1~1 but x_in maybe not
            lpips_val = loss_fn_alex.forward(x_in, guide_x_t).sum()
            loss -=  lpips_val * args.lpips_scale
            writer.add_scalar(f"pic{count}/lpips",lpips_val, time)

            if time < args.range_t2:
                attack_logits = attack_model(th.clamp((x_in+1)/2.,0.,1.)) # 所以这里输入也是不对的... 
                # print(time, th.topk(attack_logits, dim=-1, k=3))
                attack_log_probs = F.log_softmax(attack_logits, dim=-1)
                selected = attack_log_probs[range(len(attack_logits)), y.view(-1)] 
                loss -= selected.sum() * args.adver_scale
                writer.add_scalar(f"pic{count}/lossofattack",selected.sum(),time)

            gradient = th.autograd.grad(loss, x_in)[0]

        mean = mean.float() +  kwargs['variance'] *  gradient.float() # kwargs['variance'] 感觉可以变成常量?

        # PGD
        # if time < args.range_t2:

        return mean

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        # if you want to perturb here change y 
        time = int(t[0].detach().cpu())
        if time > args.range_t3:
            y = (y + 3) %1000
        return model(x, t, y if args.class_cond else None)


    logger.log("creating samples...")
    while count * args.batch_size < args.num_samples:
        model_kwargs = {}
        x, y = next(data)
        x, y = x.to(dist_util.dev()), y.to(dist_util.dev())
        model_kwargs["guide_x"] = x
        model_kwargs['y'] = y
        model_kwargs['guide_y'] = (y.detach().clone()+30)%1000
        begin_time = time()
        sample = diffusion.p_sample_loop(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            # back_fn=back_fn,
            device=dist_util.dev(),
        )
        end_time = time()
        time_sum += end_time - begin_time
        sample = th.clamp(sample,-1.,1.)

        num = (count+1) * args.batch_size
        logger.log(f"created {num} samples")

        at_predict = attack_model((sample+1)/2).argmax(dim=-1)
        attack_succ_sum += sum(at_predict != y)

        cl_predict = classifier(sample, th.zeros_like(y)).argmax(dim=-1)
        attack_clas_succ_sum += sum(cl_predict != y)

        logger.log(f'original label of sample: {y.cpu().numpy()}, {[map_i_s[i] for i in y.cpu().numpy()]}')
        logger.log(f'predict of attack_model: {at_predict.cpu().numpy()}, {mapname(at_predict)}')
        logger.log(f'predict of classifier: {cl_predict.cpu().numpy()}, {mapname(cl_predict)}')
        logger.log(f'fool attck_model: {attack_succ_sum/num}; fool classifier: {attack_clas_succ_sum/num}')
        ssim_val = ssim((sample+1)/2, (x+1)/2, data_range=1, size_average=True)
        ssim_sum += ssim_val
        # # range from -1~1 but x_in maybe not
        lpips_val = loss_fn_alex.forward(x, sample).sum()/ args.batch_size
        lpips_sum += lpips_val
        logger.log(f'ssim of sample: {ssim_val}')
        logger.log(f'lpips of sample: {lpips_val}')
        logger.log(f'cost time: {end_time - begin_time}s')
        
        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(),
                                    f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )
        count += 1

    dist.barrier()
    logger.log(f"sampling complete, avg_time is {time_sum/(count*args.batch_size)}")
    logger.log(f"sampling complete, avg_lpips is {lpips_sum/(count)}")
    logger.log(f"sampling complete, avg_ssim is {ssim_val/(count)}")

def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result',
        describe="ilvr",
        clip_denoised=True,
        num_samples=100,
        batch_size=4,
        down_N=8,
        range_t1=600,
        range_t2=1000,
        range_t3=1000,
        model_path="guided-diffusion/models/256x256_diffusion.pt",
        classifier_path="guided-diffusion/models/256x256_classifier.pt",
        attack_model_name= "Standard_R50", # "Standard_R50", #"Salman2020Do_50_2"
        attack_model_type='Linf',
        generate_scale=1.0,
        adver_scale=10,
        ssim_scale=10,
        lpips_scale=50,#1000,
        seed=777,
    ) 
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    model_flags = dict(
        use_ddim=False,
        timestep_respacing=[100],
        image_size=256,
        attention_resolutions="32,16,8",
        class_cond=True, 
        diffusion_steps=1000,
        learn_sigma=True, 
        dropout=0.0,
        noise_schedule="linear",
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown=True,
        use_fp16=True,
        use_scale_shift_norm=True,
        )
    defaults.update(model_flags)
    parser = argparse.ArgumentParser() 
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()



