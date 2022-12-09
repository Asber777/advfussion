import torch
from torch import nn, Tensor
from typing import Literal, Optional, Iterable, Tuple

ThreatModel = Literal['Linf', 'L2']

def clip(x: Tensor, x0: Tensor, threat: str, eps: float) -> Tensor:
    if threat == 'Linf':
        eta = torch.clamp(x - x0, -eps, eps)
        return torch.clamp(x0 + eta, 0.0, 1.0)
    elif threat == 'L2':
        norm = ((x - x0) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()
        return  torch.clamp(x0 + (x - x0) / 
            (norm+ 1e-12) * torch.min(eps * torch.ones_like(x), norm.detach()), 0.0, 1.0)
    raise ValueError(f'Unrecognized threat mode {threat!r}.')

def tslice(
    ts: Iterable[Optional[Tensor]], indices: Tensor
) -> Tuple[Optional[Tensor], ...]:
    return tuple(t[indices] if t is not None else None for t in ts)

StepSizeSchedule = Literal['constant', 'linear', 'cosine']

class BaseAttack:
    early_stop: bool
    resume: bool

    def __init__(
        self, early_stop: bool = True, resume: bool = False
    ) -> None:
        super().__init__()
        self.early_stop = early_stop
        self.resume = resume

    def attack_batch(
        self,
        indices: Tensor, images: Tensor, labels: Tensor,
        adv_images: Optional[Tensor] = None
    ) -> Tensor:
        raise NotImplementedError

    def loss_func(
        self, 
        indices: Tensor, outputs: Tensor, labels: Tensor,
        images: Tensor, adv_images: Tensor
    ) -> Tensor:
        return nn.functional.cross_entropy(outputs, labels, reduction='none')
