import argparse
from advfussion.script_util import model_and_diffusion_defaults, \
    classifier_defaults, add_dict_to_argparser
    
def create_argparser():
    defaults = dict(
        describe="TEST", # mask_ilvr_half_attack
        batch_size=5,
        stop_count=1000,
        # adver setting
        use_adver=True,
        range_t2_e=200, 
        range_t2_s=0,
        attack_model_name= "Engstrom2019Robustness", #"Standard_R50", #"Salman2020Do_50_2"
        adver_scale=0.4, #3
        # PGD at begin
        # nb_iter=30,
        nb_iter_conf=25, #1
        # half setting
        use_half=True,
        start_t=100, #100,  # must <= max(timestep_respacing) ? currently
        # CAM setting
        use_cam=True,
        mask_p=1,
        device='cuda',
        pure_name = "",
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    model_flags = dict(
        use_ddim=False,
        timestep_respacing=[250],
        class_cond=True, 
        diffusion_steps=1000,
        )
    unchange_flags = dict(
        ImageNetpath = "./data/images",
        seed=666,
        result_dir='./result',
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