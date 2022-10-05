import argparse
from guided_diffusion.script_util import model_and_diffusion_defaults, \
    classifier_defaults, add_dict_to_argparser
    
def create_argparser():
    defaults = dict(
        describe="checkx0", # mask_ilvr_half_attack
        num_samples=5,
        batch_size=5,
        # ilvr setting
        use_ilvr=True,
        down_N=8,
        range_t1=35, # 这个是在0~timestep_respacing之间 决定什么时候ilvr
        # adver setting
        use_adver=False,
        range_t2=1000, # 这个是在0~diffusion_steps之间 决定什么时候攻击
        attack_model_name= "Salman2020Do_50_2", #"Standard_R50", #"Salman2020Do_50_2"
        adver_scale=8,
        seed=666,
        # half setting
        use_half=True,
        start_t=50,#100,  # must <= max(timestep_respacing) ? currently
        # PGD at begin
        nb_iter=20,
        # CAM setting
        use_cam=True,
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