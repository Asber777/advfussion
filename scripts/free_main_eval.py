
import datetime
import os 
import json
import datetime
from advfussion.free.TrainCondition import eval
'''
        # Linf setting
        # "adver_scale":1/255,
        # "nb_iter_conf":3,
        # "eps":3/255,
        # "threat_name":"Linf", 
'''
if __name__ == '__main__':
    config = {
        "sample_num": 1000,  # total number of images to generate
        "batch_size": 50, # batch size of once generation
        "device": "cuda", # assign your device
        "save_dir": "./", # where to load weight of condition diffusion
        "test_load_weight": "DiffusionConditionWeight.pt", # the name of weight of condition diffusion
        "sampled_dir": "./Appendix", # I think it might be not used
        "nrow": 10, # how much pictures in a row when using method <save_image>
        'seed':8,  # random seed
        # attack 
        "start_T":100, # Half-start time
        "name": "Standard", #"Rebuffi2021Fixing_70_16_cutmix_extra",#"Standard", # What model is used for attacked
        "robustPath":"./models", # where to load the attacked model 
        'cifar10path' : '/root/datasets', # where to load CIFAR10 data

        "adver_scale":0.15, # hyper1 
        "nb_iter_conf":1, # hyper2
        "eps":0.15, # hyper1
        "threat_name":"L2", #
        "ts":0, # hyper3 : the time to end adversarial guidance, usually to be 0
        "te":100, # hyper3 : the time to start adversarial guidance
        "perturbxp":False, # NOTE: if change predicted image, which is abandoned
        'lpips_scale':0.01, # scale when changing predicted image, which is abandoned
        "perturbxt":True, # if use adversarial guidance
        # NOTE: Attacker: plz check advfussion-cifar10/DiffusionFreeGuidence/lafeat.py - Method <attack_batch> for more info
        'Attacker':'lafeat',  # 'lafeat', 'pgd', 'label' 
        'measure': False, # if calculate Nature Error\Target Error\Avg L2\Avg LPIPSAvg SSIM after generation. 
        'Pure':False,  # if calculate acc after diffpure 
        'useCAM': True, # if use CAM when doing adversarial guidance. 
        'save_intermediate_result': False, 
        'time':'',# use when state == 'pure' ,other wise is assign to current time 

        "state": "", # or "ablation", "draw_pure_cur", "get_y"
        "T": 500, # total time of timestep, it should be 500 when condition, 1000 when uncondition
        "channel": 128, # plz remain unchange when using our checkpoint
        "channel_mult": [1, 2, 2, 2],# plz remain unchange when using our checkpoint
        "num_res_blocks": 2,# plz remain unchange when using our checkpoint
        "dropout": 0.15,# plz remain unchange when using our checkpoint
        "beta_1": 1e-4,# plz remain unchange when using our checkpoint
        "beta_T": 0.028,# plz remain unchange when using our checkpoint
        "img_size": 32,# cifar10 size
        "w": 1.8,  # classifier-free guidance's parameter, decideing how much the guidance is  
    }
    config['state'] = 'eval'
    config['time'] = datetime.datetime.now().strftime("%m-%d-%H-%M")
    config['save_path'] = config["sampled_dir"] + config['time']
    os.makedirs(config['save_path'], exist_ok=True)
    args_path = os.path.join(config['save_path'], f"exp.json")
    info_json = json.dumps(config, sort_keys=False, indent=4, separators=(' ', ':'))
    with open(args_path, 'w') as f:
        f.write(info_json)
    eval(config)