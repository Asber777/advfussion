import math
import torch
from torch import nn, Tensor
from typing import Any, Optional
from guided_diffusion.lafeat_util import StepSizeSchedule, ThreatModel, clip, tslice
import guided_diffusion.logger as logger

def grad_schedule(
    epsilon: float, schedule: float, grad: Tensor, 
    step_size: Optional[float] = None, 
    step_size_schedule: StepSizeSchedule = 'constant',
    mask = 1, threat: ThreatModel = 'Linf', 
) -> Tensor:
    if step_size_schedule == 'constant':
        s = step_size
    elif step_size_schedule == 'linear':
        s = 2 * step_size * (1 - schedule)
    elif step_size_schedule == 'cosine':
        s = step_size * (1 + math.cos(schedule * math.pi))
    else:
        raise ValueError(
            f'Unknown step size schedule {step_size_schedule!r}.')
    if threat == 'Linf':
        return s * grad.sign() * mask
    else:
        return s * grad /((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)  * mask

def update_func(
    threat: ThreatModel, epsilon: float, schedule: float,
    xai: Tensor, xai2: Tensor, xgi: Tensor, xw: Tensor, 
    momentum: float = 0.75, step_size: Optional[float] = None, 
    step_size_schedule: StepSizeSchedule = 'constant',
    mask = 1,
) -> Tensor:
    xan = xai + grad_schedule(epsilon, schedule, xgi, step_size, step_size_schedule, mask, threat)
    xan = clip(xan, xw, threat, epsilon) 
    mmt = momentum if schedule > 0 else 0
    xai = xai + (1 - mmt) * (xan - xai) + mmt * (xai - xai2)
    return clip(xai, xw, threat, epsilon)

    
def LafeatLoss(outputs: Tensor, labels: Tensor, num_classes=1000, temperature=1.0, reduction='none'):
    batch_size = outputs.shape[0]
    onehot = nn.functional.one_hot(labels, num_classes)
    top1 = outputs[range(batch_size), labels]
    top2 = torch.max((1.0 - onehot) * outputs, dim=1).values
    scale = (top1 - top2).detach().unsqueeze(1).clamp_(min=0.1)
    outputs = outputs / scale / temperature
    return nn.functional.cross_entropy(outputs, labels, reduction=reduction) 

def attack_batch(
    images: Tensor, labels: Tensor, 
    iterations: int, model:Any,
    loss:str, 
    step_size: Optional[float] = None,  
    epsilon: float = 8/255, 
    momentum: float = 0,
    threat_name: ThreatModel = 'Linf',
    step_size_schedule: StepSizeSchedule = 'constant',
    returndelta = True,
    early_stop = True,
    resume = False,
    mask = 1, 
    use_lpips=False,
    lpips_fn = None,
    originalx = None,
    lpips_scale = 0, 
    ) -> Tensor:
    batch_size = images.shape[0]
    xi = images.detach().clone()
    xa, xai = images.detach().clone(), images.detach().clone()
    xai2 = xai.detach().clone() # used for mmt 
    li = labels.detach().clone()
    success = torch.zeros(batch_size, device=xa.device).bool()
    with torch.enable_grad():
        for i in range(iterations):
            xai.requires_grad_()
            oi = model(xai)
            if i == 0:
                logger.log(f"before attack, acc={sum(oi.max(1)[1] == li)/batch_size}")
            if not early_stop:
                loss = LafeatLoss(oi, li, reduction='none').mean()
                xgi = torch.autograd.grad(loss, xai)[0]
            else:
                si = (oi.max(1)[1] != li).detach()
                xa[~success] = xai.detach()
                success[~success] = si
                loss = LafeatLoss(oi, li, reduction='none').mean()
                # loss = nn.functional.cross_entropy(oi, li, reduction='none').mean()
                oi, li, xai2, xi = tslice((oi, li, xai2, xi), ~si)
                xgi = torch.autograd.grad(loss, xai)[0]
                xgi, xai = tslice((xgi, xai), ~si)
                mask = tslice([mask], ~si)[0] if type(mask) != int else 1
                if sum(si) == batch_size:
                    logger.log(f"early stop at iter{i}")
                    break
            xai2, xai = xai.detach(), update_func(
                threat_name, epsilon, i / iterations, xai, xai2, xgi, xi, 
                momentum, step_size, step_size_schedule, mask
                ).detach()
    if early_stop:
        xa[~success] = xai.detach()
    else:
        xa = xai.detach()
    if resume:
        oi = model(xa)
        logger.log(f"after attack, acc={sum(oi.max(1)[1] == labels)/batch_size}")
    if returndelta:
        delta = xa - images.detach().clone()
        return delta
    return xa

def cond_fn(x, y, model, var, mask=None, 
        adver_scale=1, nb_iter_conf = 1):
    maks = mask.detach().clone() if mask != None else 0
    eps = torch.sqrt(var) * 3
    delta = torch.zeros_like(x)
    with torch.enable_grad():
        delta.requires_grad_()
        for _ in range(nb_iter_conf):
            tmpx = x.detach().clone() + delta  # range from -1~1
            attack_logits = model(torch.clamp((tmpx+1)/2.,0.,1.)) 
            target = torch.where(attack_logits.argmax(dim=1)==y, y, attack_logits.argmin(dim=1))
            sign = torch.where(attack_logits.argmax(dim=1)==y, 1, -1)
            selected = sign * attack_logits[range(len(attack_logits)), target.view(-1)] 
            loss = -selected.sum()
            loss.backward()
            grad_ = delta.grad.data.detach().clone()
            delta.data += grad_  * adver_scale *(1-maks)
            delta.data = torch.clamp(delta.data, -eps, eps)
            delta.grad.data.zero_() 
    return delta.detach().clone()


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from torchvision.datasets import CIFAR10
    from torchvision.utils import save_image
    from robustbench import load_model
    # TODO: 检查L2这边 不同eps 不同eps_iter的影响 为什么
    modelConfig = {
        'device': 'cuda:0',
        'name': 'Rebuffi2021Fixing_70_16_cutmix_extra', #'Rebuffi2021Fixing_70_16_cutmix_extra',
        'robustPath':"/root/hhtpro/123/models",
    }
    device = torch.device(modelConfig["device"])
    dataset = CIFAR10(
        root='/root/hhtpro/123/DenoisingDiffusionProbabilityModel-ddpm-/CIFAR10', train=False, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=30, shuffle=True,
        num_workers=4, drop_last=True, pin_memory=True)
    attack_model = load_model(
        model_name=modelConfig["name"], 
        model_dir=modelConfig['robustPath'],
        dataset='cifar10', threat_model='L2')
    attack_model = attack_model.to(device)
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        pred = attack_model(images).argmax(dim=1)
        print("ACC:", sum(labels==pred)/len(labels))
        # attacker = LafeatAttack(20, attack_model, 'Linf', True, True, 2/255, 8/255)
        returnkwarg = attack_batch(images, labels, 20, attack_model,'lafeat', 0.2, 0.5,threat_name='L2',
            step_size_schedule='constant') 
        advx = images + returnkwarg['delta']
        pred = attack_model(advx).argmax(dim=1)
        print("ACC after attack:", sum(labels==pred)/len(labels))
        save_image(images, 'x.png', nrow=8,  normalize=True, range=(0, 1))
        save_image(advx, 'advx.png', nrow=8,  normalize=True, range=(0, 1))
        break