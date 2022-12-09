import torch 
import torch.nn as nn
from torch import clamp
import numpy as np
import random
import torch.backends.cudnn as cudnn

def init_seeds(seed=8, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

def diffuson_pgd(x, y, attack_model, nb_iter=1, eps=8./255, eps_iter=2./255, 
                clip_min=-1.0, clip_max=1.0, target=False, output_delta=False):
    x, y = x.detach().clone(), y.detach().clone()
    delta = torch.zeros_like(x)
    # delta = nn.Parameter(delta)
    CE = nn.CrossEntropyLoss(reduction="sum")
    with torch.enable_grad():
        delta.requires_grad_()
        for _ in range(nb_iter):
            outputs = attack_model((x + delta+1)/2)
            # if sum(mask) == 0: break
            loss = CE(outputs, y)
            loss.backward()
            # first limit delta in [-eps,eps] then limit data in [clip_min,clip_max](inf_ord)
            grad_sign = delta.grad.data.sign()
            grad_sign *= eps_iter
            delta.data += grad_sign
            delta.data = clamp(delta.data, -eps, eps)
            delta.data = clamp(x.data + delta.data, clip_min, clip_max) - x.data
            delta.grad.data.zero_()
    return delta.data if output_delta else x + delta.data

def add_border(img, flag, width=1, R=1, G=0, B=0):
    img = img.detach().clone()
    img[flag, 0, :width, :] = R
    img[flag, 1, :width, :] = G
    img[flag, 2, :width, :] = B
    img[flag, 0, -width:, :] = R
    img[flag, 1, -width:, :] = G
    img[flag, 2, -width:, :] = B
    img[flag, 0, :, :width] = R
    img[flag, 1, :, :width] = G
    img[flag, 2, :, :width] = B
    img[flag, 0, :, -width:] = R
    img[flag, 1, :, -width:] = G
    img[flag, 2, :, -width:] = B
    img[~flag, 0, :width, :] = 0
    img[~flag, 1, :width, :] = 0
    img[~flag, 2, :width, :] = 1
    img[~flag, 0, -width:, :] = 0
    img[~flag, 1, -width:, :] = 0
    img[~flag, 2, -width:, :] = 1
    img[~flag, 0, :, :width] = 0
    img[~flag, 1, :, :width] = 0
    img[~flag, 2, :, :width] = 1
    img[~flag, 0, :, -width:] = 0
    img[~flag, 1, :, -width:] = 0
    img[~flag, 2, :, -width:] = 1
    return img

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.utils import save_image
    dataset = CIFAR10(
        root='./CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), 
        ]))
    dataloader = DataLoader(
        dataset, batch_size=10, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)
    flag = torch.zeros([10]).bool()
    flag[1] = True
    flag[3] = True
    flag[7] = True
    for img, y in dataloader:
        break
    save_image(add_border(img, flag), 'addborder.png', 
        nrow=1,  normalize=True, range=(-1, 1))
