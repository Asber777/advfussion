import os 
from advfussion.free.TrainCondition import eval
from advfussion.script_util import save_args, add_dict_to_argparser
from advfussion import logger
import argparse

EXP_PATH = "/root/hhtpro/123/result_of_cifar10_exp"
def main():
    AdvDiffuserConfig = {
        "T": 500, # total time of timestep, it should be 500 when condition, 1000 when uncondition
        "channel": 128, # plz remain unchange when using our checkpoint
        "channel_mult": [1, 2, 2, 2],# plz remain unchange when using our checkpoint
        "num_res_blocks": 2,# plz remain unchange when using our checkpoint
        "dropout": 0.15,# plz remain unchange when using our checkpoint
        "beta_1": 1e-4,# plz remain unchange when using our checkpoint
        "beta_T": 0.028,# plz remain unchange when using our checkpoint
        "img_size": 32,# cifar10 size
        "w": 1.8, 
        "weight": "/root/hhtpro/123/models/ddpm_free/DiffusionConditionWeight.pt", 
    }
    AttackParameterConfig = {
        'useCAM': True, 
        "adver_scale":0.8, 
        "nb_iter_conf":1,
        "ts":0, 
        "te":50, 
        "start_T":50, 
        'contrastive':False,
    }
    Config = {
        "sample_num": 1000,  
        "batch_size": 50, 
        "save_dir": "ICCV",
        "device": "cuda", # assign your device
        'seed':8, 
        "robustPath":"/root/hhtpro/123/models", 
        'cifar10path' : '/root/hhtpro/123/CIFAR10', 
    }
    AttackedModelConfig = {
        "name": "Rebuffi2021Fixing_70_16_cutmix_extra",
        "threat_name":"L2", 
    }
    Config.update(AdvDiffuserConfig)
    Config.update(AttackParameterConfig)
    Config.update(AttackedModelConfig)
    parser = argparse.ArgumentParser() 
    add_dict_to_argparser(parser, Config)
    args = parser.parse_args()
    Config = vars(args)
    Config['save_dir'] = os.path.join(EXP_PATH, Config['save_dir'])
    Config['result_dir'] = os.path.join(Config['save_dir'], 'result')
    os.makedirs(Config['result_dir'], exist_ok=True)
    logger.configure(Config['save_dir'], 
                     log_suffix=f'pure_{Config["start_T"]/Config["T"]}')
    save_args(logger.get_dir(), args)
    eval(Config)


if __name__ == '__main__':
    main()
