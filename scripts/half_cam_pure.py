
import os
import torch as th
import argparse
import os.path as osp
from torchvision import utils
import torch.distributed as dist
from robustbench.utils import load_model
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults,seed_torch
from guided_diffusion.script_util import model_and_diffusion_defaults, \
    classifier_defaults, add_dict_to_argparser

''' 
check PURE ACC for advDDPM
'''
    
def create_argparser():
    defaults = dict(
        describe="makeitsiilar-early_stop", # mask_ilvr_half_attack
        # num_samples=5,
        batch_size=5,
        stop_count=1000,
        use_adver=False,
        range_t2_e=200,
        range_t2_s=0,
        attack_model_name= "Standard_R50", #"Standard_R50", #"Salman2020Do_50_2"
        attack_model_type='Linf',
        adver_scale=0.4, #3
        # PGD at begin
        nb_iter=30,
        nb_iter_conf=25, #1
        seed=666,
        # half setting
        use_half=True,
        start_t=100, #100,  # must <= max(timestep_respacing) ? currently
        # CAM setting
        use_cam=True,
        threshold=0.5,
        # change_pre_time=20,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    model_flags = dict(
        use_ddim=False,
        timestep_respacing=[250],
        class_cond=False, 
        diffusion_steps=1000,
        )
    unchange_flags = dict(
        result_dir='../../result',
        clip_denoised=True,
        image_size=256,
        model_path="guided-diffusion/models/256x256_diffusion_uncond.pt",
        attention_resolutions="32,16,8",
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
    defaults.update(unchange_flags)
    parser = argparse.ArgumentParser() 
    add_dict_to_argparser(parser, defaults)
    return parser
device_idx = "cuda:0"
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
    print(th.cuda.device_count())
    a = th.randn(19)
    a.to(th.device('cuda:2'))
    device = th.device(device_idx)
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    dir = osp.join(args.result_dir, args.describe, args.attack_model_name,)
    logger.configure(dir, log_suffix='pure_uncond')
    pure_dir = osp.join(logger.get_dir(), 'pure_uncond')
    os.makedirs(pure_dir, exist_ok=True)
    # load args

    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(device)
    if args.use_fp16: model.convert_to_fp16()
    model.eval()

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(device).eval()

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("creating samples...")
    target_err_total_after_pure = th.tensor(0.0).to(device)
    seed_torch(args.seed)
    with th.no_grad():
        for count in range(5,1001, 5):
            d = th.load(os.path.join(logger.get_dir(), f'result/{str(count).zfill(5)}.pt'))
            advx, x, y, = d['sample'].to(device), d['x'].to(device), d['y'].to(device)
            model_kwargs = {"guide_x":advx, "y":y, "mask":None,}
            sample = diffusion.p_sample_loop(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                start_t=args.start_t,
                device=device,
                progress=True,
            )
            sample = th.clamp(sample,-1.,1.)
            at_predict = attack_model((sample+1)/2).argmax(dim=-1)
            err_mask = (at_predict.data != y.data)
            target_err_total_after_pure += err_mask.float().sum()
            logger.log(f"target_err_total_after_pure:{target_err_total_after_pure}/{count}")
            utils.save_image(sample, os.path.join(pure_dir, f"{str(count).zfill(5)}.png"), 
            nrow=args.batch_size, normalize=True, range=(-1, 1),)
    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()
