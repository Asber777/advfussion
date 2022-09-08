import argparse
import os
import shutil
import os.path as osp
import torch.distributed as dist
import lpips
from torch import clamp
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
from guided_diffusion.grad_cam import GradCamPlusPlus, get_last_conv_name
import math

def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result',
        describe="ilvr_nocond",
        clip_denoised=True,
        num_samples=5,
        batch_size=5,
        down_N=8,
        range_t1=500,
        range_t2=1000,
        model_path="guided-diffusion/models/256x256_diffusion.pt",
        classifier_path="guided-diffusion/models/256x256_classifier.pt",
        attack_model_name= "Salman2020Do_50_2", #"Standard_R50", #"Salman2020Do_50_2"
        attack_model_type='Linf',
        generate_scale=1.0,
        adver_scale=0.1,
        ssim_scale=0,
        lpips_scale=0, 
        seed=777,
        start_t=60,  # must <= max(timestep_respacing) ? currently
        nb_iter=10,
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

    classifier = None
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     dist_util.load_state_dict(args.classifier_path, map_location="cpu"))
    # classifier.to(dist_util.dev())
    # if args.classifier_use_fp16: classifier.convert_to_fp16()
    # classifier.eval()

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev())
    layer_name = get_last_conv_name(attack_model)
    grad_cam = GradCamPlusPlus(attack_model, layer_name)

    # loss_fn_alex = lpips.LPIPS(net='alex')
    # loss_fn_alex = loss_fn_alex.to(dist_util.dev())

    data = load_imagenet_batch(args.batch_size, '/root/hhtpro/123/imagenet')
    # writer = SummaryWriter(logger.get_dir())

    map_i_s = get_idex2name_map(INDEX2NAME_MAP_PATH)
    mapname = lambda predict: [map_i_s[i] for i in predict.cpu().numpy()]
    # get_grid = lambda pic: make_grid(pic.detach().clone(), args.batch_size, normalize=True)

    # assert math.log(args.down_N, 2).is_integer()
    # shape = (args.batch_size, 3, args.image_size, args.image_size)
    # shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    # down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    # up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    # resizers = (down, up)

    def cond_fn(x, t, y=None, guide_x=None, guide_x_t=None, mean=None, variance=None,  pred_xstart=None, **kwargs):
        time = int(t[0].detach().cpu()) # using this variable in add_scalar will GO WRONG!
        
        # ILVR: 
        # if time > args.ranget1 and guide_x_t is not None and resizers is not None:
        #     down, up = resizers
        #     if time > args.range_t1: 
        #         mean = mean - up(down(mean)) + up(down(guide_x_t))

        # with th.enable_grad():
        #     x_in = mean.detach().requires_grad_(True)# "mean" used to be "x" 

        #     logits = classifier(x_in, t)
        #     log_probs = F.log_softmax(logits, dim=-1)
        #     selected = log_probs[range(len(logits)), y.view(-1)] 

        #     loss = selected.sum() * args.generate_scale

            # # range from 0~1 but x_in maybe not
            # loss += ssim((guide_x+1)/2, (x_in+1)/2, 
            #     data_range=1, size_average=True) * args.ssim_scale

            # # range from -1~1 but x_in maybe not
            # loss -= loss_fn_alex.forward(x_in, guide_x).sum() * args.lpips_scale

            # gradient = th.autograd.grad(loss, x_in)[0]

        # mean = mean.float() +  kwargs['variance'] *  gradient.float() # kwargs['variance'] 感觉可以变成常量?

        if time < args.range_t2:
            delta = th.zeros_like(x)
            with th.enable_grad():
                delta.requires_grad_()
                tmpx = pred_xstart.detach().clone() + delta # range from -1~1
                attack_logits = attack_model(th.clamp((tmpx+1)/2.,0.,1.)) 
                # attack_log_probs = F.log_softmax(attack_logits, dim=-1)
                selected = attack_logits[range(len(attack_logits)), y.view(-1)] 
                loss = -selected.sum() * args.adver_scale 
                loss.backward()
                grad_sign = delta.grad.data.detach().clone()
                # delta.grad.data.zero_()
            mean = mean.float() + grad_sign.float()

        return mean

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)


    logger.log("creating samples...")
    count = 0
    attack_acc = 0
    attack_clas_acc = 0
    seed_torch(args.seed)
    while count * args.batch_size < args.num_samples:
        x, y = next(data)     
        x, y = x.to(dist_util.dev()), y.to(dist_util.dev())
        mask = grad_cam((x+1)/2.)
        pgd_x = diffuson_pgd(x, y, attack_model, nb_iter=args.nb_iter,)
        model_kwargs = {"guide_x":pgd_x, "y":y}
        sample = diffusion.p_sample_loop(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            start_t=args.start_t,
            mask=mask,
        )
        sample = th.clamp(sample,-1.,1.)

        num = (count+1) * args.batch_size
        logger.log(f"created {num} samples")
        logger.log(
            f'original label of sample: {y.cpu().numpy()},'+
            f'{[map_i_s[i] for i in y.cpu().numpy()]}')

        if attack_model is not None:
            at_predict = attack_model((sample+1)/2).argmax(dim=-1)
            attack_acc += sum(at_predict != y)
            logger.log(
                f'predict of attack_model: {at_predict.cpu().numpy()}, '+
                f'{mapname(at_predict)}')
            logger.log(f'fool attck_model: {attack_acc/num};')

        if classifier is not None:
            cl_predict = classifier(sample, th.zeros_like(y)).argmax(dim=-1)
            attack_clas_acc += sum(cl_predict != y)
            logger.log(
                f'predict of classifier: {cl_predict.cpu().numpy()}, '+
                f'{mapname(cl_predict)}')
            logger.log(f' fool classifier: {attack_clas_acc/num}')

        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(), 
                f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(sample[i].unsqueeze(0), 
                out_path, nrow=1, normalize=True, range=(-1, 1),)
        count += 1
    
    dist.barrier()
    logger.log("sampling complete")
    grad_cam.remove_handlers()

if __name__ == "__main__":
    main()
