
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
 

class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T):
        super().__init__()

        self.model = model
        self.T = T

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    def forward(self, x_0, labels):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w = 0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod, [1, 0], value=1)[:T]
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

        self.register_buffer('sqrt_recip_alphas_cumprod', np.sqrt(1.0 / alphas_cumprod))# sqrt ( 1/ {(bar αt)^2} )
        self.register_buffer('sqrt_recipm1_alphas_cumprod', np.sqrt(1.0 / alphas_cumprod - 1)) # sqrt( 1/ {(bar αt)^2 -1} )
        
        self.register_buffer('posterior_mean_coef1', 
            self.betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer('posterior_mean_coef2', 
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

    def get_xt(self, x_0, t):
        t = torch.ones(size=(x_0.shape[0], ), device=x_0.device, dtype = torch.int64) * int(t)
        noise = torch.randn_like(x_0)
        x_t =   extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0 + \
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        return x_t
        
    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        return posterior_mean
    
    def cond_fn(self, pred_xstart, y, model, var, mask=0, 
            adver_scale=1, nb_iter_conf = 1, early_stop=True, contrastive=False):
        # eps = torch.sqrt(var) * 3
        delta = torch.zeros_like(pred_xstart)
        with torch.enable_grad():
            delta.requires_grad_()
            for _ in range(nb_iter_conf):
                tmpx = pred_xstart.detach() + delta  # range from -1~1
                attack_logits = model(torch.clamp((tmpx+1)/2.,0.,1.)) 
                if early_stop:
                    target = y
                    sign = torch.where(attack_logits.argmax(dim=1)==y, 1, 0)
                    if sign.sum() == 0: 
                        print("early stop")
                        break
                else:
                    target = torch.where(attack_logits.argmax(dim=1)==y, y, attack_logits.argmin(dim=1))
                    sign = torch.where(attack_logits.argmax(dim=1)==y, 1, -1)
                if not contrastive:
                    original_loss = attack_logits[range(len(attack_logits)), target.view(-1)]
                else:
                    v,i = torch.topk(attack_logits,10,dim=1)
                    original_loss = -v[:,1] + v[:,-1] + v[:, 0] # find secondary logits and last like logits
                selected = sign * original_loss
                loss = -selected.sum()
                loss.backward()
                grad_ = delta.grad.data.detach().clone()
                delta.data += grad_  * adver_scale *(1-mask) 
                # delta.data = torch.clamp(delta.data, -eps, eps)
                delta.grad.data.zero_() 
        return delta.data.float().detach()
    
    def p_mean_variance(self, x_t, t, labels, attack_model=None, kwargs=None):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        # labels should between 0~9, and diffusion use 0 as no label, so we need to add 1.
        eps = self.model(x_t, t, labels+1)
        nonEps = self.model(x_t, t, torch.zeros_like(labels))
        eps = (1. + self.w) * eps - self.w * nonEps
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps).clamp(-1, 1)
        xt_prev_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        if kwargs.get('ts', -1) <= int(t[0].detach().cpu()) <= kwargs.get('te', -1):             
            delta = self.cond_fn(pred_xstart, labels, attack_model, var, 
                mask=kwargs.get('mask', 0), 
                adver_scale=kwargs.get('adver_scale', 0), 
                nb_iter_conf=kwargs.get('nb_iter_conf', 0), 
                contrastive = kwargs.get('contrastive', False)) 
            xt_prev_mean = xt_prev_mean.float() + delta.data.float() * 2 
        return xt_prev_mean.detach(), var

    def forward(self, x_T, labels, model=None, start_T=None, kwargs=None, ):
        x_t = x_T
        for time_step in tqdm(reversed(range(start_T))):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, 
                labels=labels, attack_model=model, kwargs=kwargs)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
