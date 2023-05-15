
import os
import torch as th
import os.path as osp
from torchvision import utils
import torch.distributed as dist
import torch.nn.functional as F
from robustbench.utils import load_model
from advfussion import dist_util, logger
from advfussion.script_util import create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults,seed_torch
from advfussion.myargs import create_argparser
os.environ['CUDA_VISIBLE_DEVICES']='0,1'
uncondition_ddpm_path = "/root/hhtpro/123/models/guide_ddpm/256x256_diffusion_uncond.pt"


def main():
    args = create_argparser().parse_args()
    args.model_path = uncondition_ddpm_path
    args.class_cond = False
    args.batch_size = 1
    dist_util.setup_dist()
    device = dist_util.dev()
    Pure_name = f"pure_Ts:{args.start_t}"
    logger.configure(args.dir, log_suffix=Pure_name)
    logger.log(f"using {th.cuda.device_count()} GPU.")
    pure_dir = osp.join(logger.get_dir(), Pure_name)
    os.makedirs(pure_dir, exist_ok=True)

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
        return model(x, t, None)

    logger.log("creating samples...")
    target_err_total_after_pure = th.tensor(0.0).to(device)
    seed_torch(args.seed)
    with th.no_grad():
        for i in range(1000):
            advx = th.load(os.path.join(logger.get_dir(), f'image/{str(i)}.pt'))
            advx = advx.to(device).unsqueeze(0)
            advx = F.interpolate(advx, size=args.image_size, mode="nearest")
            y = th.tensor([i]).to(device)
            pred = attack_model(advx).argmax(dim=-1)
            model_kwargs = {"guide_x":advx.detach().clone()*2-1, "mask":None,}
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
            logger.log(f"target_err_total_after_pure:{target_err_total_after_pure}/{i+1}")
            utils.save_image(sample, os.path.join(pure_dir, f"{str(i)}.png"), 
                nrow = 5, normalize=True, range=(-1, 1),)
            pt_path = os.path.join(f'{pure_dir}/{str(i)}.pt')
            th.save({'sample':sample, "predict_before_pure":pred, 'y':y, "predict":at_predict}, pt_path)
    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    main()
