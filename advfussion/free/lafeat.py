import math
import torch
from torch import nn, Tensor
from typing import Any, Optional
from .lafeat_util import StepSizeSchedule, ThreatModel, clip, tslice
from . import logger

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

    
def LafeatLoss(outputs: Tensor, labels: Tensor, num_classes=10, temperature=1.0, reduction='none'):
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
    **kwargs
    ) -> Tensor:
    if loss == 'lafeat':
        return attack_batch_lafeat(
            images=images, 
            labels=labels,
            iterations=iterations,
            model=model, 
            step_size= step_size,
            **kwargs)
    else:
        return attack_batch_pgd(
            images=images, 
            labels=labels,
            iterations=iterations,
            model=model, 
            step_size = step_size,
            **kwargs)

def attack_batch_pgd(
    images: Tensor, labels: Tensor, 
    iterations: int, model:Any,
    step_size: Optional[float] = None,  
    early_stop = True,
    resume = False,
    mask = 1, 
    ) -> Tensor:
    x = images.detach().clone()
    y = labels.detach().clone()
    batch_size = images.shape[0]
    returnkwarg = {"init_acc":0, "acc_after":-1, "flag":None, 'delta':None, "init_logit":None, "logit_after": None}
    delta = torch.zeros_like(x)
    with torch.enable_grad():
        delta.requires_grad_()
        for i in range(iterations):
            attack_logits = model(torch.clamp(x + delta,0.,1.)) 
            if i == 0:
                logger.log(f"before attack, acc={sum(attack_logits.max(1)[1] == y)/batch_size}")
                if resume:  
                    returnkwarg['init_acc'] = sum(attack_logits.max(1)[1] == y)/batch_size
                    returnkwarg['init_logit'] = attack_logits[range(batch_size), labels]
            if early_stop:
                target = y
                sign = torch.where(attack_logits.argmax(dim=1)==y, 1, 0)
            else:
                target = torch.where(attack_logits.argmax(dim=1)==y, y, attack_logits.argmin(dim=1))
                sign = torch.where(attack_logits.argmax(dim=1)==y, 1, -1)
            selected = sign * attack_logits[range(len(attack_logits)), target.view(-1)] 
            loss = -selected.sum()
            loss.backward()
            grad_ = delta.grad.data.detach().clone()
            delta.data += grad_  * step_size  *(1-mask)
            delta.grad.data.zero_()
    if resume:
        attack_logits = model(torch.clamp(x + delta,0.,1.)) 
        flag = attack_logits.max(1)[1] == labels
        returnkwarg['acc_after'] = sum(flag)/batch_size 
        returnkwarg['logit_after'] = attack_logits[range(batch_size), labels]
        logger.log(f"after attack, acc={sum(flag)/batch_size}")
        returnkwarg['flag'] = flag
    returnkwarg['delta'] = delta
    return returnkwarg

def attack_batch_lafeat(
    images: Tensor, labels: Tensor, 
    iterations: int, model:Any,
    step_size: Optional[float] = None,  
    epsilon: float = 8/255, 
    momentum: float = 0.75,
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
    returnkwarg = {"init_acc":0, "acc_after":-1, "flag":None, 'delta':None, "init_logit":None, "logit_after": None}
    with torch.enable_grad():
        for i in range(iterations):
            xai.requires_grad_()
            oi = model(xai)
            if i == 0:
                logger.log(f"before attack, acc={sum(oi.max(1)[1] == li)/batch_size}")
                if resume:  
                    returnkwarg['init_acc'] = sum(oi.max(1)[1] == li)/batch_size
                    returnkwarg['init_logit'] = oi[range(batch_size), labels]
            if not early_stop:
                loss = LafeatLoss(oi, li, reduction='none').mean()
                xgi = torch.autograd.grad(loss, xai)[0]
            else:
                if success.all(): 
                    logger.log(f"early stop at iter{i}")
                    break
                si = (oi.max(1)[1] != li).detach()
                xa[~success] = xai.detach()
                success[~success] = si
                loss = LafeatLoss(oi, li, reduction='none').mean()
                oi, li, xai2, xi = tslice((oi, li, xai2, xi), ~si)
                xgi = torch.autograd.grad(loss, xai)[0]
                xgi, xai = tslice((xgi, xai), ~si)
                mask = tslice([mask], ~si)[0] if type(mask) != int else 1
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
        flag = oi.max(1)[1] == labels
        returnkwarg['acc_after'] = sum(flag)/batch_size 
        returnkwarg['logit_after'] = oi[range(batch_size), labels]
        logger.log(f"after attack, acc={sum(flag)/batch_size}")
        returnkwarg['flag'] = flag
    if returndelta:
        returnkwarg['delta'] = xa - images.detach().clone()
        return returnkwarg
    returnkwarg['delta'] = xa
    return returnkwarg

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