import argparse
import os
import shutil
import os.path as osp
import torch.distributed as dist
import lpips
from torch import clamp
import torch.nn.functional as F
from robustbench.utils import load_model
from guided_diffusion.data_model.architectures import get_architecture
from guided_diffusion.data_model.my_loader import MyCustomDataset
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
        describe="checkx0", # mask_ilvr_half_attack
        num_samples=5,
        batch_size=5,
        # ilvr 
        use_ilvr=False,
        down_N=8,
        range_t1=80, # 这个是在0~timestep_respacing之间 决定什么时候ilvr
        # adver
        use_adver=False,
        range_t2=1000, # 这个是在0~diffusion_steps之间 决定什么时候攻击
        attack_model_name="InceptionResnetV2",  # "Salman2020Do_50_2", #"Standard_R50", #"Salman2020Do_50_2"
        adver_scale=1.8,
        seed=666,
        # half
        use_half=False,
        start_t=None,#100,  # must <= max(timestep_respacing) ? currently
        # PGD at begin
        nb_iter=20,
        # CAM
        use_cam=False,
        threshold=0.5,
    ) 
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    model_flags = dict(
        use_ddim=False,
        timestep_respacing=[100],
        class_cond=True, 
        diffusion_steps=1000,
        )
    unchange_flags = dict(
        result_dir='/root/hhtpro/123/result',
        clip_denoised=True,
        image_size=256,
        model_path="guided-diffusion/models/256x256_diffusion.pt",
        classifier_path="guided-diffusion/models/256x256_classifier.pt",
        attack_model_type='Linf',
        # 
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
    
    # data = load_imagenet_batch(args.batch_size, '/root/hhtpro/123/imagenet')
    # writer = SummaryWriter(logger.get_dir())
    data = MyCustomDataset(img_path="/root/hhtpro/123/GA-Attack-main/data")
    if args.distributed:
        sampler = th.utils.data.distributed.DistributedSampler(data)
    else:
        sampler = th.utils.data.SequentialSampler(data)
    attack_loader = th.utils.data.DataLoader(dataset=data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=8, pin_memory=True)

    if args.use_adver:
        attack_model = get_architecture(model_name=args.attack_model_name)
        # attack_model = load_model(model_name=args.attack_model_name, 
        #     dataset='imagenet', threat_model=args.attack_model_type)
        attack_model = attack_model.to(dist_util.dev()).eval()
    if args.use_cam:
        layer_name = get_last_conv_name(attack_model)
        grad_cam = GradCamPlusPlus(attack_model, layer_name)

    map_i_s = get_idex2name_map(INDEX2NAME_MAP_PATH)
    mapname = lambda predict: [map_i_s[i] for i in predict.cpu().numpy()]

    resizers = None
    if args.use_ilvr:
        assert math.log(args.down_N, 2).is_integer()
        shape = (args.batch_size, 3, args.image_size, args.image_size)
        shape_d = (args.batch_size, 3, int(args.image_size / args.down_N), int(args.image_size / args.down_N))
        down = Resizer(shape, 1 / args.down_N).to(next(model.parameters()).device)
        up = Resizer(shape_d, args.down_N).to(next(model.parameters()).device)
        resizers = (down, up) # 现在把ilvr放在gaussian diffusion文件里执行

    def cond_fn(x, t, y=None, guide_x=None, guide_x_t=None, 
            mean=None, variance=None,  pred_xstart=None, mask=None, threshold=None, **kwargs):
        '''
        x: x_{t+1}
        mean: x_{t}
        guide_x: pgd_x0
        guide_x_t: pgd_x0_t
        mean = mean.float() +  kwargs['variance'] *  gradient.float() # kwargs['variance'] 感觉可以变成常量?
        '''
        time = int(t[0].detach().cpu()) # using this variable in add_scalar will GO WRONG!
        out_path = os.path.join(logger.get_dir(), 
            f"pred_xstart{str(time)}.png")
        utils.save_image(pred_xstart[0].unsqueeze(0), 
            out_path, nrow=1, normalize=True, range=(-1, 1),)
        
        if args.use_adver and time < args.range_t2:
            maks = th.where(model_kwargs['mask'] > model_kwargs['threshold'], 1.,0.)
            delta = th.zeros_like(x)
            with th.enable_grad():
                delta.requires_grad_()
                tmpx = pred_xstart.detach().clone() + delta *(1-maks) # range from -1~1
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
    seed_torch(args.seed)
    # while count * args.batch_size < args.num_samples:
    for (img, label, img_name) in attack_loader:
        x, y = img.to(dist_util.dev()), label.to(dist_util.dev())
        mask = grad_cam((x+1)/2.).unsqueeze(1) if args.use_cam else None
        x = diffuson_pgd(x, y, attack_model, nb_iter=args.nb_iter,) if args.use_adver else x
        model_kwargs = {"guide_x":x, "y":y, "mask":mask, 
            "resizer":resizers,'range_t1':args.range_t1, "threshold":args.threshold}
        sample = diffusion.p_sample_loop(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
            start_t=args.start_t if args.use_half else None,
            progress=True,
        )
        sample = th.clamp(sample,-1.,1.)

        num = (count+1) * args.batch_size
        logger.log(f"created {num} samples")
        logger.log(
            f'original label of sample: {y.cpu().numpy()},'+
            f'{[map_i_s[i] for i in y.cpu().numpy()]}')

        if args.use_adver:
            at_predict = attack_model((sample+1)/2).argmax(dim=-1)
            attack_acc += sum(at_predict != y)
            logger.log(
                f'predict of attack_model: {at_predict.cpu().numpy()}, '+
                f'{mapname(at_predict)}')
            logger.log(f'fool attck_model: {attack_acc/num};')

        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(), 
                f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(sample[i].unsqueeze(0), 
                out_path, nrow=1, normalize=True, range=(-1, 1),)
        count += 1
    
    dist.barrier()
    logger.log("sampling complete")
    if args.use_cam: 
        grad_cam.remove_handlers()

if __name__ == "__main__":
    main()
