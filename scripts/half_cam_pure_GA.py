
import os
import torch as th
import os.path as osp
import torch.distributed as dist
import torch.nn.functional as F
from robustbench.utils import load_model
from tqdm import tqdm
from advfussion import dist_util, logger
from advfussion.script_util import create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults,seed_torch
from advfussion.myargs import create_argparser
print("current cuda count:", th.cuda.device_count())
device_idx = "cuda:1"
def main(dir, a, b):
    device = th.device(device_idx)
    args = create_argparser().parse_args()
    args.attack_model_name = a
    args.attack_model_type = b
    args.batch_size = 1
    dist_util.setup_dist()
    logger.configure(dir, log_suffix='pure')
    pure_dir = osp.join(logger.get_dir(), 'pure')
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
        for i in tqdm(range(1000)):
            advx = th.load(os.path.join(logger.get_dir(), f'image/{str(i)}.pt'))
            advx = advx.to(device)
            advx = advx.unsqueeze(0)
            advx = F.interpolate(advx, size=args.image_size, mode="nearest")
            y = th.tensor([i]).to(device)
            model_kwargs = {"guide_x":advx.detach().clone()*2-1, "y":y, "mask":None,}
            sample = diffusion.p_sample_loop(
                model_fn,
                (1, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=None,
                start_t=args.start_t,
                device=device,
                progress=True,
            )
            sample = th.clamp(sample,-1.,1.)
            at_predict = attack_model((sample+1)/2).argmax(dim=-1)
            th.save({"sample":sample, "at_predict":at_predict}, os.path.join(pure_dir, f"{str(i)}.pt"))
            err_mask = (at_predict.data != y.data)
            target_err_total_after_pure += err_mask.float().sum()
            logger.log(f"target_err_total_after_pure:{target_err_total_after_pure}/{i+1}")
    dist.barrier()
    logger.log("sampling complete")

if __name__ == "__main__":
    # attack_model_name = 'Engstrom2019Robustness'
    attack_model_name = 'Salman2020Do_50_2'
    # attack_model_name = 'Standard_R50' # Engstrom2019Robustness
    # attack_model_name = 'Salman2020Do_R50'
    attack_model_type = 'Linf'
    OUTPUT_DIR=None
    assert OUTPUT_DIR is not None, "plz assign the dir of GA-attack"
    main(OUTPUT_DIR, attack_model_name, attack_model_type)
