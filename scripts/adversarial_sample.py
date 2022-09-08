"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import shutil
import json
import numpy as np
import torch as th
import os.path as osp
import datetime
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import transforms

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

'''
This file can generate new adversarial images that look like y but are classified as 
guide_y which is asign by us. 
Use following command to do so:
SAMPLE_FLAGS="--batch_size 5 --num_samples 5 --timestep_respacing 250"
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True \
--noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True \
--use_fp16 True --use_scale_shift_norm True"
python guided-diffusion/scripts/adversarial_sample.py $MODEL_FLAGS --classifier_path 64x64_classifier.pt --classifier_depth 4 \
--model_path 64x64_diffusion.pt $SAMPLE_FLAGS  --classifier_scale 1.0 --adv_scale 0.0 --describe "adv_generate"
'''

def main():
    args = create_argparser().parse_args()
    dist_util.setup_dist()

    dir = osp.join(args.result_dir, args.describe,
        datetime.datetime.now().strftime("adv-%Y-%m-%d-%H-%M-%S-%f"),
    )
    logger.configure(dir)

    logger.log("creating model and diffusion...")
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

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    # modify conf_fn and model_fn to get adv, conf_fn

        # get guide picture and check batchsize
    if args.guide_exp:
        guide_path = osp.join(args.result_dir, args.guide_exp, "samples_5x64x64x3.npz")
        guide_np = np.load(guide_path)
        guide_x_np = guide_np['arr_0']
        guide_y_np = guide_np['arr_1']
        assert args.batch_size <= len(guide_y_np)
        guide_x = th.from_numpy(guide_x_np[:args.batch_size]).to(dist_util.dev())
        guide_x = guide_x.permute(0, 3, 1, 2)
        guide_x = ((guide_x/127.5) -1.).clamp(-1., 1.) # to float32?
        guide_y = th.from_numpy(guide_y_np[:args.batch_size]).to(dist_util.dev())
    else:
        guide_x_np = None
        guide_y_np = np.array([1, 2, 3, 4, 5])
        guide_y = th.from_numpy(guide_y_np[:args.batch_size]).to(dist_util.dev())

    def cond_fn(x, t, y=None,):
        assert y is not None
        with th.enable_grad():
            # hidden_loss = 0
            x_in = x.detach().requires_grad_(True)
            # guide_logits, guide_hidden = classifier(guide_x, t, args.get_hidden, args.get_middle)
            # logits, hidden = classifier(x_in, t, args.get_hidden, args.get_middle)
            logits = classifier(x_in, t)
            # for h, gh in zip(hidden, guide_hidden):
            #     hidden_loss += (h * gh).mean()
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), guide_y.view(-1)]
            loss = selected.sum() * args.classifier_scale
            # t = t[0].cpu().detach().item()
            # logger.log("hidden_loss_{}: {}".format(t, hidden_loss.detach()))
            # logger.log("guide_loss_{}: {}".format(t, selected.sum().detach()))
            # loss = hidden_loss * args.hidden_scale
            return th.autograd.grad(loss, x_in)[0]

    def model_fn(x, t, y=None,):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_predict = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )
        log_probs = classifier(sample, th.zeros_like(guide_y))
        predict = log_probs.argmax(dim=-1)

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        gathered_predicts = [th.zeros_like(predict) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_predicts, predict)
        all_predict.extend([predict.cpu().numpy() for predict in gathered_predicts])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    predict_arr = np.concatenate(all_predict, axis=0)
    predict_arr = predict_arr[:args.num_samples]
    
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        map_i_s = get_index_name_map()
        for i, (y_, p_) in enumerate(zip(label_arr, predict_arr)):
            g_y = guide_y_np[i%args.batch_size]
            logger.log(f"label_{i}:{y_}, guide_y_{i%args.batch_size}:{g_y}, predict{i}:{p_}")
            logger.log(f"{map_i_s[y_]} ; {map_i_s[g_y]}; {map_i_s[p_]}")
        if guide_x_np:
            np.savez(out_path, arr, guide_x_np, label_arr, guide_y_np, predict_arr)
        else:
            np.savez(out_path, arr, label_arr, guide_y_np, predict_arr)
        # save argparser json 
        args_path = os.path.join(logger.get_dir(), f"exp.json")
        info_json = json.dumps(vars(args), sort_keys=False, indent=4, separators=(' ', ':'))
        with open(args_path, 'w') as f:
            f.write(info_json)
        # copy code in case some result need to check it's historical implementation.
        shutil.copy('/root/hhtpro/123/guided-diffusion/scripts/adversarial_sample.py', logger.get_dir())
        show_pic_n = 5
        picture = arr[:show_pic_n]
        picture = th.from_numpy(picture)
        picture = picture.permute(0, 3, 1, 2)
        row_picture = th.cat([pic for pic in picture], 2)
        unloader = transforms.ToPILImage()
        unloader(row_picture).save(osp.join(logger.get_dir(), "result.jpg"))
    
    dist.barrier()
    logger.log("sampling complete")


    # 读取引导图, 引导图的数量>=batch_size
    # 每次生产都是用过相同的引导图. 

def create_argparser():
    defaults = dict(
        result_dir='/root/hhtpro/123/result',
        guide_exp="",
        clip_denoised=True,
        num_samples=5,
        batch_size=5,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        hidden_scale=1.0,
        adv_scale=0.0,
        get_hidden=True,
        get_middle=True, 
        describe="default_desc"
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

def get_index_name_map():
    result_dict = {}
    with open("/root/hhtpro/123/guided-diffusion/scripts/image_label_map.txt") as fp:
        sample = fp.readlines()
        for line in sample:
            sample_ = line.split('\t',maxsplit=1)
            result_dict[int(sample_[0])]=sample_[1].split('\n')[0]
    return result_dict

if __name__ == "__main__":
    main()
