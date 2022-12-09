# guided-diffusion

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for generate adversarial exampels.

# install as package

```
pip install git+https://github.com/RobustBench/robustbench.git
pip install -e .
```

# Download pre-trained models

## pre-trained model of ImageNet
Here are the download links for each model checkpoint:

 * 256x256 classifier: [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
 * 256x256 diffusion: [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
 * 256x256 diffusion (not class conditional): [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)
set "model_path" in advfussion/myargs.py to where you put weight.

## pre-trained models of CIFAR-10
The link of diffusion weight is "https://drive.google.com/file/d/1B9jC_x72TxvcYWn8nOrUToIp10tPufSU/view?usp=sharing"
The link of diffusion condition weight is "https://drive.google.com/file/d/1vtEbQEC_DZftDeuCgX8hg9a0oWCirji1/view?usp=sharing"
set "save_dir" in script/free_main to where you put weight. 

# Imagenet eval at GA-attack
You could download the data/images dataset from [google drive](https://drive.google.com/file/d/1M7Xc7guRKk_YuLoDf-xVv45HX3nh4r_-/view?usp=sharing) (140M) 
and put them at "/data/images"

We assume that you have downloaded the relevant model checkpoints and Dataset.
# RUN ADVDDPM
To sample from conditional DDPM in 256*256 image, you can use the `half_cam_attack.py` scripts.
Then we will generate 1000 samples with batch size 5. Feel free to change the hyper parameter values.
```
python scripts/half_cam_attack.py --adver_scale 0.4 --range_t2_e 200 --start_t 100
```
plz make sure that "start_t" is smaller than "timestep_respacing"(250)


* You can run MainCondition.py to get Unrestricted Adversarial Examples on CIFAR-10.