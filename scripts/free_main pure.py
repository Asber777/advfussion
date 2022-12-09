
import datetime
import os 
import json
import datetime
from advfussion.free.TrainCondition import pure

if __name__ == '__main__':
    modelConfig = {
        "state": "eval", # or "ablation", "draw_pure_cur", "get_y"
        "sample_num": 1000,  # total number of images to generate
        "batch_size": 50, # batch size of once generation
        "T": 500, # total time of timestep, it should be 500 when condition, 1000 when uncondition
        "channel": 128, # plz remain unchange when using our checkpoint
        "channel_mult": [1, 2, 2, 2],# plz remain unchange when using our checkpoint
        "num_res_blocks": 2,# plz remain unchange when using our checkpoint
        "dropout": 0.15,# plz remain unchange when using our checkpoint
        "lr": 1e-4, # which is not used here
        "multiplier": 2.5,  # which is not used here
        "beta_1": 1e-4,# plz remain unchange when using our checkpoint
        "beta_T": 0.028,# plz remain unchange when using our checkpoint
        "img_size": 32,# cifar10 size
        "grad_clip": 1., # which is not used when not training
        "device": "cuda", # assign your device
        "w": 1.8,  # classifier-free guidance's parameter, decideing how much the guidance is  
        "save_dir": "./", # where to load weight of condition diffusion
        "training_load_weight": None, # which is not used when not training
        "test_load_weight": "DiffusionConditionWeight.pt", # the name of weight of condition diffusion
        "sampled_dir": "./Appendix", # I think it might be not used
        "nrow": 10, # how much pictures in a row when using method <save_image>
        'seed':8,  # random seed
        # attack 
        "start_T":100, # Half-start time
        "name": "Standard", #"Rebuffi2021Fixing_70_16_cutmix_extra",#"Standard", # What model is used for attacked
        "robustPath":"./models", # where to load the attacked model 
        'cifar10path' : '/root/datasets', # where to load CIFAR10 data

        # Linf setting
        # "adver_scale":1/255,
        # "nb_iter_conf":3,
        # "eps":3/255,
        # "threat_name":"Linf", 

        "adver_scale":0.15, # hyper1 
        "nb_iter_conf":1, # hyper2
        "eps":0.15, # hyper1
        "threat_name":"L2", #
        "ts":0, # hyper3 : the time to end adversarial guidance, usually to be 0
        "te":100, # hyper3 : the time to start adversarial guidance
        "perturbxp":False, # if change predicted image, which is abandoned
        'lpips_scale':0.01, #  # scale when changing predicted image, which is abandoned
        "perturbxt":True, # if use adversarial guidance
        # Attacker: plz check advfussion-cifar10/DiffusionFreeGuidence/lafeat.py - Method <attack_batch> for more info
        'Attacker':'lafeat',  # 'lafeat', 'pgd', 'label' 
        'measure': True, # if calculate Nature Error\Target Error\Avg L2\Avg LPIPSAvg SSIM after generation. 
        'Pure':False,  # if calculate acc after diffpure 
        'useCAM': True, # if use CAM when doing adversarial guidance. 
        'save_intermediate_result': False, 
        'time':'',# use when state == 'pure' ,other wise is assign to current time 
    }
    save_arg = [ # 
        "state", "name","start_T","threat_name", 
        "perturbxt", "perturbxp", "eps", 
        "adver_scale", "nb_iter_conf",
        "te", "ts", 'Attacker', 'useCAM', 'time'
    ]
    PureConfig = { #  hyperparametr of Diffpure
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
        "test_load_weight": "./Checkpoints/DiffusionWeight.pt", 
        "start_T":200,
        }
    if modelConfig['state'] == 'eval':
        modelConfig['time'] = datetime.datetime.now().strftime("%m-%d-%H-%M")
        config_str = "/".join([f"{k}:{modelConfig[k]}"for k in save_arg])
        modelConfig['save_path'] = modelConfig["sampled_dir"] +'/'+ config_str
        os.makedirs(modelConfig['save_path'], exist_ok=True)
        args_path = os.path.join(modelConfig['save_path'], f"exp.json")
        info_json = json.dumps(modelConfig, sort_keys=False, indent=4, separators=(' ', ':'))
        with open(args_path, 'w') as f:
            f.write(info_json)
    elif modelConfig['state'] == 'pure':
        assert modelConfig['time'] is not None
        config_str = "state:eval/"+"/".join([f"{k}:{modelConfig[k]}"for k in save_arg[1:]])
        modelConfig['save_path'] = modelConfig["sampled_dir"] +'/'+ config_str
        assert os.path.isdir(modelConfig['save_path'])
    eval(modelConfig)