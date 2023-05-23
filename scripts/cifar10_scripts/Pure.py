from advfussion.free.pure.Train import eval
from advfussion.script_util import save_args, add_dict_to_argparser
from advfussion import logger
import argparse

if __name__ == '__main__':
    modelConfig = {
        "batch_size": 50,
        "T": 1000,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda",
        "test_load_weight": "/root/hhtpro/123/models/ddpm_free/DiffusionWeight.pt",
        'seed': 8 ,
        'name': "Rebuffi2021Fixing_70_16_cutmix_extra", 
        "robustPath":"/root/hhtpro/123/models",
        "threat_name":"L2", 
        "eval_path":"/root/hhtpro/123/result_of_cifar10_exp/ICCV/r_rho-0.1",  
        "start_T":100,
        }
    parser = argparse.ArgumentParser() 
    add_dict_to_argparser(parser, modelConfig)
    args = parser.parse_args()
    modelConfig = vars(args)
    logger.configure(args.eval_path, log_suffix=f'pure_{args.start_T/ args.T}')
    save_args(logger.get_dir(), args)
    eval(modelConfig)