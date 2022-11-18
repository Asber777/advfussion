# guided-diffusion

This repository is based on [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with modifications for generate adversarial exampels.

# Download pre-trained models and Image

Here are the download links for each model checkpoint:

 * 256x256 classifier: [256x256_classifier.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt)
 * 256x256 diffusion: [256x256_diffusion.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt)
 * 256x256 diffusion (not class conditional): [256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt)

you can download them and put them in folder "/models".

You could download the data/images dataset from [google drive](https://drive.google.com/file/d/1M7Xc7guRKk_YuLoDf-xVv45HX3nh4r_-/view?usp=sharing) (140M) 
and put them at "/data/images"


We assume that you have downloaded the relevant model checkpoints into a folder called `models/`.

To sample from conditional DDPM in 256*256 image, you can use the `half_cam_attack.py` scripts.
Then we will generate 1000 samples with batch size 5. Feel free to change the hyper parameter values.
```
python scripts/half_cam_attack.py --adver_scale 0.4 --range_t2_e 200 --start_t 100
```
plz make sure that "start_t" is smaller than "timestep_respacing"(250)
