
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
from guided_diffusion.myargs import create_argparser
''' 
check PURE ACC for advDDPM
'''
    
device_idx = "cuda:0"
def main():
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
    print(th.cuda.device_count())
    a = th.randn(19)
    a.to(th.device('cuda:2'))
    device = th.device(device_idx)
    args = create_argparser().parse_args()
    args.use_adver=False
    args.class_cond=False
    dist_util.setup_dist()
    dir = osp.join(args.result_dir, args.describe, args.pure_name,)
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
