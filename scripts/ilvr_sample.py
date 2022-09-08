import argparse
import os

import os.path as osp
import torch.distributed as dist

from guided_diffusion import dist_util, logger
from guided_diffusion.resizer import Resizer
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
    load_imagenet_batch, 
)
# from guided_diffusion.image_datasets import load_data
from torchvision import utils
import math


def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()
    # dir = osp.join(args.result_dir, args.describe, DT(),)
    logger.configure(dir=args.save_dir)

    logger.log("creating model...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("creating resizers...")
    assert math.log(args.down_N, 2).is_integer()

    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    logger.log("loading data...")
    data = load_imagenet_batch(args.batch_size, '/root/hhtpro/123/imagenet')

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)


    def back_fn(x, y=None, guide_x=None, guide_y=None, time=None):
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
            if time > args.range_t: 
                x = x - up(down(x)) + up(down(guide_x))
                # x = clamp(x, -1, 1)  
        return x

    logger.log("creating samples...")
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs = {}
        x, y = next(data)
        x, y = x.to(dist_util.dev()), y.to(dist_util.dev())
        model_kwargs["guide_x"] = x
        model_kwargs['y'] = y
        model_kwargs['guide_y'] = y
        
        sample = diffusion.p_sample_loop(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            back_fn=back_fn,
            device=dist_util.dev(),
        )
        
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
        logger.log(f"created {count * args.batch_size} samples")

    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=40,
        batch_size=4,
        down_N=8,
        range_t=60,
        use_ddim=False,
        model_path="guided-diffusion/models/256x256_diffusion.pt",
        save_dir="guided-diffusion/output",
        # save_latents=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser() 
    defaults.update(
        dict(
            attention_resolutions="32,16,8", 
            diffusion_steps=1000,
            dropout=0.0,
            image_size=256,
            learn_sigma=True,
            noise_schedule="linear",
            num_channels=256,
            num_head_channels=64,
            num_res_blocks=2,
            resblock_updown=True,
            use_fp16=True,
            use_scale_shift_norm=True,
            timestep_respacing=[100],
            class_cond=True, 
        )
    )
    '''
    dict(
    attention_resolutions="16", # 
    diffusion_steps=1000,
    dropout=0.0,
    image_size=256,
    learn_sigma=True,
    noise_schedule="linear",
    num_channels=128,
    num_head_channels=64,
    num_res_blocks=1,
    resblock_updown=True,
    use_fp16=False,
    use_scale_shift_norm=True,
    timestep_respacing=[100],
    class_cond=False, 
    )
    '''
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()



