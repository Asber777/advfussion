import argparse
import os
import shutil
import numpy as np
import torch as th
import os.path as osp
import datetime
import torch.distributed as dist
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim, ms_ssim
from robustbench.utils import load_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torch.nn.functional import one_hot
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES, 
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    load_imagenet_batch, 
    get_idex2name_map, 
    save_args, 
    arr2pic_save, 
    diffuson_pgd, 
    seed_torch,
    get_steps_scale_map, 
)

TARGET_MULT = 10000.0
hidden_index = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
# GUIDE_Y = [583, 445, 703, 546, 254]
INDEX2NAME_MAP_PATH = "/root/hhtpro/123/guided-diffusion/scripts/image_label_map.txt"
DT = lambda :datetime.datetime.now().strftime("adv-%Y-%m-%d-%H-%M-%S-%f")

def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result', # where to store experiments
        describe="robustacc",
        clip_denoised=True,
        num_samples=100,
        batch_size=5,
        model_path="/root/hhtpro/123/256x256_diffusion.pt",
        classifier_path="/root/hhtpro/123/256x256_classifier.pt",
        splitT=300, 
        generate_scale=10.0,
        guide_scale=100.0, 
        multi_guide=[20,10,2,1,1,1,1,1,1,1], 
        hidden_scale=0.0,
        lpips_scale=10.0, 
        ssim_scale=10.0, 
        use_lpips=True,
        use_mse=False,
        use_ssim=True, 
        use_pgd=False, 
        get_hidden=True,
        get_middle=True, 
        guide_as_generate=False, 
        attack_model_name= "Salman2020Do_50_2", #"Standard_R50",
        attack_model_type='Linf',
        target_attack=True, 
        target=[123,234,345,456,567],
        sample_noguide = False, 
        seed=666, 
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    # MODEL_FLAGS
    model_flags = dict(
        timestep_respacing=[25,25,10,3,3,1,1,1,1,1],  # [30,30,5,5,5,4,4,4,4,4] "ddim25"
        use_ddim=True,
        # timestep_respacing = [125],
        # use_ddim = False, 
        image_size=256, 
        attention_resolutions="32,16,8",
        class_cond=True, 
        diffusion_steps=1000,
        learn_sigma = True, 
        noise_schedule='linear', 
        num_channels=256,
        num_head_channels=64,
        num_res_blocks=2,
        resblock_updown = True, 
        use_fp16 = True,
        use_scale_shift_norm = True
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

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16: model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16: classifier.convert_to_fp16()
    classifier.eval()

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev())

    loss_fn_alex = lpips.LPIPS(net='alex')
    loss_fn_alex = loss_fn_alex.to(dist_util.dev())

    data_generator = load_imagenet_batch(args.batch_size, '/root/hhtpro/123/imagenet')
    writer = SummaryWriter(logger.get_dir())
    # modify conf_fn and model_fn to get adv
    map_i_s = get_idex2name_map(INDEX2NAME_MAP_PATH)
    mapname = lambda predict: [map_i_s[i] for i in predict.cpu().numpy()]
    index, g = 0, 0
    get_grid = lambda pic: make_grid(pic.detach().clone(), args.batch_size, normalize=True)
    scale_map = get_steps_scale_map(args.diffusion_steps, args.timestep_respacing, args.multi_guide)
    def cond_fn(x, t, y=None, guide_x=None, guide_y=None, **kwargs):
        nonlocal index, g
        assert y is not None
        assert guide_x is not None
        if args.target_attack: 
            guide_y = th.from_numpy(np.array(guide_y)).to(dist_util.dev())
        time = int(t[0].detach().cpu()) # using this variable in add_scalar will GO WRONG!
        with th.enable_grad():
            generate_loss = 0
            hidden_loss = 0
            lpips_loss = 0
            ssim_loss = 0
            adv_loss = 0
            loss = 0

            x_in = x.detach().requires_grad_(True)
            if args.use_lpips == True:
                # input need to be range from [-1, 1]
                lpips_loss += loss_fn_alex.forward(x_in, guide_x).sum()
            if args.use_ssim == True: 
                # ssim need input range from [0, 1]
                ssim_loss = ssim((guide_x+1)/2, (x_in+1)/2, data_range=1, size_average=True)
            if args.use_mse == True:
                _, guide_hidden = classifier(guide_x, t, \
                    args.get_hidden, args.get_middle, hidden_index)
                _, hidden = classifier(x_in, t, \
                    args.get_hidden, args.get_middle, hidden_index)
                for h, gh in zip(hidden, guide_hidden):
                    hidden_loss -= ((h - gh)**2).mean()
            
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)] 
            generate_loss = selected.sum()

            attack_logits = attack_model(((x_in+1)/2.))
            attack_log_probs = F.log_softmax(attack_logits, dim=-1)
            
            if not args.target_attack:
                if time <= args.splitT:
                    # optimize generate_term and guide_term when t < splitT together
                    # selected = log_probs[range(len(logits)), y.view(-1)] 
                    # generate_loss = selected.sum()
                    y_onehot = one_hot(y, NUM_CLASSES) 
                    attack_selected = ((1.0 - y_onehot) * log_probs 
                    - (y_onehot * TARGET_MULT)).max(1)[0]
                    adv_loss = attack_selected.sum()
            else: 
                # 最初的版本 只对classifier进行攻击
                if time > args.splitT: 
                    pass
                    # selected = log_probs[range(len(logits)), y.view(-1)] 
                    # generate_loss = selected.sum()
                else: 
                    # adv_loss = log_probs[range(len(logits)), guide_y.view(-1)].sum()
                    attack_selected = attack_log_probs[range(len(logits)), guide_y.view(-1)]
                    adv_loss = attack_selected.sum()
                    for i, v in enumerate(attack_selected.detach().cpu().numpy()): 
                        writer.add_scalar(f'group{g}/guidey{i}', v, index)

                    # adv_loss = attack_log_probs[range(len(logits)), guide_y.view(-1)].sum()
            
            loss += generate_loss* args.generate_scale
            if args.multi_guide is not None:
                loss += adv_loss * args.guide_scale * scale_map[time]
            else:
                loss += adv_loss * args.guide_scale
            loss += ssim_loss * args.ssim_scale
            loss += hidden_loss * args.hidden_scale
            loss -= lpips_loss * args.lpips_scale
            gradient = th.autograd.grad(loss, x_in)[0]

        # logger.log(f"{time}\n max of logits:{th.topk(logits, 3, 1)}; \
        #      max of attack_logits:{th.topk(attack_logits, 3, 1)} \n")
        for i, v in enumerate(selected.detach().cpu().numpy()): 
            writer.add_scalar(f'group{g}/y{i}', v, index)
        writer.add_histogram(f'group{g}/classifier_logits', log_probs, index)
        writer.add_histogram(f'group{g}/attack_logits', attack_log_probs, index)
        writer.add_scalar(f'group{g}/t', time, index)
        writer.add_scalar(f'group{g}/generate_loss', generate_loss, index)
        writer.add_scalar(f'group{g}/adv_loss', adv_loss, index)
        writer.add_scalar(f'group{g}/ssim_loss', ssim_loss, index)
        writer.add_scalar(f'group{g}/lpips_loss', lpips_loss, index)
        writer.add_image(f'group{g}/gradient', get_grid(gradient), index)
        pic = kwargs['mean'].float() + kwargs['variance'] * gradient.float()
        writer.add_image(f'group{g}/x', get_grid(pic), index)
        index += 40 if args.timestep_respacing == 'ddim25' else 1
        if args.use_pgd and t[0] < 10:
            x = pic * 0.5 + 0.5
            advx = diffuson_pgd(x, guide_y, )
            new_mean = (advx - 0.5) * 2.0
            return new_mean
        return gradient
    def fake_cond_fn(x, t, y=None, guide_x=None, guide_y=None, **kwargs):
        return 0

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...", DT())

    attack_acc = 0
    attack_clas_acc = 0
    sampled_num = 0
    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
    while sampled_num < args.num_samples:
        guide_x, y = data_generator.__next__()
        guide_x, y = guide_x.to(dist_util.dev()), y.to(dist_util.dev())
        writer.add_image(f'group{g}/guide_x', get_grid(guide_x), index)
        model_kwargs = {"y": y} # what label pic should looks like
        model_kwargs["guide_x"] = guide_x  # what pic should looks like
        model_kwargs["guide_y"] = None if not args.target_attack else args.target
        seed_torch(args.seed)
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        writer.add_image(f'group{g}/x', get_grid(sample), index)
        at_predict = attack_model(((sample+1)/2.)).argmax(dim=-1)
        attack_acc += sum(at_predict != y)
        cl_predict = classifier(sample, th.zeros_like(y)).argmax(dim=-1)
        attack_clas_acc += sum(cl_predict != y)
        logger.log(f'y: {y.cpu().numpy()}, {[map_i_s[i] for i in y.cpu().numpy()]}')
        logger.log(f'predict of attack_model: {at_predict.cpu().numpy()}, {mapname(at_predict)}')
        logger.log(f'predict of classifier: {cl_predict.cpu().numpy()}, {mapname(cl_predict)}')
        if dist.get_rank() == 0:
            arr = sample.detach().cpu().numpy()
            oriimages = guide_x.detach().cpu().numpy() 
            label_arr = y.detach().cpu().numpy() 
            predict_arr =  cl_predict.detach().cpu().numpy() 
            arr2pic_save(arr, logger.get_dir(), 5, f"result{g}.jpg")
            arr2pic_save(oriimages, logger.get_dir(), 5, f"original{g}.jpg")
            shape_str = "x".join([str(x) for x in arr.shape])
            out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_{g}.npz")
            logger.log(f"saving to {out_path}")
            np.savez(out_path, arr, oriimages, label_arr, predict_arr)
        sampled_num += args.batch_size
        logger.log(f"attack success {attack_acc} / {sampled_num}")
        logger.log(f"attack  classifier success {attack_clas_acc} / {sampled_num}")
        g += 1
        index = 0
        if args.sample_noguide:
            seed_torch(args.seed)
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                device=dist_util.dev(),
            )
            writer.add_image(f'group{g}/noguid_x', get_grid(sample), index)
    writer.close()
    logger.log("sampling complete", DT())
    dist.barrier()
    logger.log("complete")

if __name__ == "__main__":
    main()
