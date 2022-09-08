import argparse
import os
import math
import shutil
import numpy as np
import torch as th
import os.path as osp
import torch.distributed as dist
import lpips
from torch import clamp
from robustbench.utils import load_model
from guided_diffusion.resizer import Resizer
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import *

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

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu"))
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16: classifier.convert_to_fp16()
    classifier.eval()

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(dist_util.dev())

    # loss_fn_alex = lpips.LPIPS(net='alex')
    # loss_fn_alex = loss_fn_alex.to(dist_util.dev())

    data_generator = load_imagenet_batch(args.batch_size, '/root/hhtpro/123/imagenet')
    writer = SummaryWriter(logger.get_dir())

    map_i_s = get_idex2name_map(INDEX2NAME_MAP_PATH)
    mapname = lambda predict: [map_i_s[i] for i in predict.cpu().numpy()]
    get_grid = lambda pic: make_grid(pic.detach().clone(), args.batch_size, normalize=True)
    index, g = 0, 0
    
    assert math.log(args.down_N, 2).is_integer()
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
    down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
    up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
    resizers = (down, up)

    def model_fn(x, t, y=None, **kwargs):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    def back_fn(x, y, guide_x, guide_y, time, **kwargs):
        assert y is not None
        assert guide_x is not None
        if resizers is not None:
            down, up = resizers
            if time > args.range_t: 
                x = x - up(down(x)) + up(down(guide_x))
                x = clamp(x, -1, 1)
        if args.fgm_fn and time < args.fgm_t:
            if args.target_attack:
                x += diffuson_pgd(x, guide_y, attack_model, args.nb_iter, args.eps_iter, 
                clip_min=-1., clip_max=1., target=True)
            else:
                x += diffuson_pgd(x, y, attack_model, args.nb_iter, args.eps_iter,
                clip_min=-1., clip_max=1.,) 
        return x

    logger.log("sampling...", DT())

    attack_acc = 0
    attack_clas_acc = 0
    sampled_num = 0

    sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )

    seed_torch(args.seed)
    while sampled_num < args.num_samples:
        guide_x, y = data_generator.__next__()
        guide_x, y = guide_x.to(dist_util.dev()), y.to(dist_util.dev())
        writer.add_image(f'group{g}/guide_x', get_grid(guide_x), index)

        model_kwargs = {"y": y} # what label pic should looks like
        model_kwargs["guide_x"] = guide_x  # what pic should looks like
        if not args.target_attack:
            model_kwargs["guide_y"] = None 
        else:
            model_kwargs["guide_y"] = th.from_numpy(np.array(args.target)).to(dist_util.dev())
     
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=None,
            device=dist_util.dev(),
            back_fn=back_fn, 
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
    writer.close()
    logger.log("sampling complete", DT())
    dist.barrier()
    logger.log("complete")
def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result', # where to store experiments
        describe="robustacc",
        clip_denoised=True,
        num_samples=5,
        batch_size=5,
        model_path="/root/hhtpro/123/256x256_diffusion.pt",
        classifier_path="/root/hhtpro/123/256x256_classifier.pt",
        attack_model_name= "Standard_R50", #"Salman2020Do_50_2", #"Standard_R50",
        attack_model_type='Linf',
        target_attack=True, 
        target=[123,234,345,456,567],
        seed=666, 
        range_t = 0,
        down_N = 4, 
        fgm_fn = True,  # 是否进行fgsm 
        fgm_t = 500,
        nb_iter = 1, 
        eps_iter = 1./255, 
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    # MODEL_FLAGS
    model_flags = dict(
        use_ddim=False, # which is need to be False in ilvr? 
        timestep_respacing = 100,
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
if __name__ == "__main__":
    main()
