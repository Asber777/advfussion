
import os
import time
import argparse
import torch 
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
import torch.nn as nn
import torch as th
import os.path as osp
import torch.distributed as dist
import functools
from robustbench.utils import load_model
from advfussion.data_model.my_loader import MyCustomDataset
from advfussion.bpda_eot.bpda_eot_attack import BPDA_EOT_Attack
from advfussion import dist_util, logger
from advfussion.script_util import create_model_and_diffusion, \
    args_to_dict, model_and_diffusion_defaults, seed_torch, classifier_defaults, \
    add_dict_to_argparser, DT
from advfussion.bpda_utils import get_accuracy

class ResNet_Adv_Model(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        # image classifier
        self.resnet = model.to(device)

    def purify(self, x):
        return x

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


def create_argparser():
    defaults = dict(
        describe="BPDA",
        batch_size=5,
        stop_count=1000,
        attack_model_name= "Salman2020Do_50_2", #"Standard_R50", #"Salman2020Do_50_2"
        attack_model_type='Linf',
        seed=666,
        # half setting
        use_half=True,
        start_t=100, #100,  # must <= max(timestep_rdespacing) ? currently
        # CAM setting
        use_cam=True,
        threshold=0.5,
        device='cuda:1',
        adv_eps=4/255,
        eot_defense_reps=20, # EOT sen
        eot_attack_reps=15, # EOT at first
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
        classifier_path="guided-diffusion/models/256x256_classifier.pt",
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
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
    print(torch.cuda.device_count())
    a = torch.randn(19)
    a.to(torch.device('cuda:2'))
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
    seed_torch(args.seed)
    device = th.device(args.device)
    dir = osp.join(args.result_dir, args.describe, DT(),)
    dist_util.setup_dist()
    logger.configure(dir, log_suffix='BPDA_PURE')
    pure_dir = osp.join(logger.get_dir(), 'pure')
    os.makedirs(pure_dir, exist_ok=True)

    # model, diffusion = create_model_and_diffusion(
    #     **args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu"))
    # model.to(device)
    # if args.use_fp16: model.convert_to_fp16()
    # model.eval()

    # def model_fn(x, t, **kwargs):
    #     return model(x, t, )

    data = MyCustomDataset(img_path="../data/images")
    sampler = th.utils.data.distributed.DistributedSampler(data)
    attack_loader = th.utils.data.DataLoader(dataset=data,
                                                batch_size=args.batch_size,
                                                shuffle=False,
                                                sampler=sampler, num_workers=2, pin_memory=True)

    attack_model = load_model(model_name=args.attack_model_name, 
        dataset='imagenet', threat_model=args.attack_model_type)
    attack_model = attack_model.to(device).eval()

    # purefun = functools.partial(diffusion.p_sample_loop, model=model_fn,  
    #     clip_denoised=args.clip_denoised,
    #     start_t=args.start_t,
    #     device=device,
    #     progress=True,
    #     )
    # ngpus = torch.cuda.device_count()
    resnet_bpda = ResNet_Adv_Model(attack_model, device)
    # SED_model = SDE_Adv_Model(attack_model, device, purefun)
    # if ngpus > 1: 
    #     resnet_bpda = torch.nn.DataParallel(resnet_bpda)
    #     SED_model = torch.nn.DataParallel(SED_model)

    count = 0
    res_NA, res_RA, pure_NA, pure_RA = 0,0,0,0
    for i, (img, label, img_name) in enumerate(attack_loader):
        count += len(img)
        img, label = img.to(device), label.to(device)
        start_time = time.time()
        init_acc = get_accuracy(resnet_bpda, img, label, bs=args.batch_size, device=device)
        
        logger.log('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))
        adversary_resnet = BPDA_EOT_Attack(resnet_bpda, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
                                        eot_attack_reps=args.eot_attack_reps)
        start_time = time.time()
        class_batch, ims_adv_batch = adversary_resnet.attack_batch(img, label)
        init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
        robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
        res_NA += float(class_batch[0, :].sum())
        res_RA += float(class_batch[-1, :].sum()) 
        logger.log('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc, time.time() - start_time))
        torch.save([ims_adv_batch, label], os.path.join(logger.get_dir(), f'x_adv_resnet_sd{i}.pt'))
        # # EOT
        # init_acc = get_accuracy(SED_model, img, label, bs=args.batch_size, device=device)
        
        # logger.log('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))
        # adversary_sde = BPDA_EOT_Attack(SED_model, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
        #                                 eot_attack_reps=args.eot_attack_reps)
        # start_time = time.time()
        # class_batch, ims_adv_batch = adversary_sde.attack_batch(img, label)
        # init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
        # robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
        # pure_NA += float(class_batch[0, :].sum())
        # pure_RA += float(class_batch[-1, :].sum())
        # logger.log('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc, time.time() - start_time))
        # torch.save([ims_adv_batch, label], os.path.join(logger.get_dir(), f'x_adv_sde_sd{i}.pt'))
        logger.log('res_NA:{}, res_RA:{}, pure_NA:{}, pure_RA:{}'.format(res_NA, res_RA, pure_NA, pure_RA))
    dist.barrier()
    logger.log("sampling complete")


if __name__ == "__main__":
    main()
