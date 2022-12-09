
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm
from .lafeat import attack_batch
from . import logger
from .utils import add_border

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

    def guide_pred_xstart(self, pred_xstart, image, mask, p, scale):
        # sign =  > (1-mask) * p
        image, mask = image.detach().clone(), mask.detach().clone()
        sign = (mask > p).int() * (abs(pred_xstart-image) > p/5).int() 
        delta = torch.zeros_like(pred_xstart)
        with torch.enable_grad():
            delta.requires_grad_()
            # loss = -(abs(pred_xstart + delta -image) * sign).sum()
            loss = -(abs(pred_xstart + delta -image) * sign * mask).sum()
            loss.backward()
            grad_ = delta.grad.data.detach().clone()
            delta.grad.data.zero_()
        return (pred_xstart + grad_* scale).detach().clone()


    def p_mean_variance(self, x_t, t, labels, attack_model=None, kwargs=None):
        modelConfig = kwargs['modelConfig']
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps).clamp(-1, 1)
        # masking
        mask = kwargs['mask']
        # if modelConfig['perturbxp']:
        #     # TODO 
        #     pred_xstart = self.guide_pred_xstart(pred_xstart, kwargs['images'],
        #         mask, modelConfig['orginal_p'], modelConfig['L2scale'])
        #     # maks = mask.detach().clone() if mask != None else 0
        #     # pred_xstart = pred_xstart * (1-mask)  + kwargs['images'] * mask
        xt_prev_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        # xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        kw = None
        if modelConfig['perturbxt'] and attack_model is not None:
            ts, te= modelConfig['ts'], modelConfig['te']
            time = int(t[0].detach().cpu())
            if ts <=time <= te:# and time % 4 == 0:
                logger.log(f"current time {time}")
                kw = attack_batch((pred_xstart.detach().clone()+1)/2, labels-1, 
                    modelConfig['nb_iter_conf'], attack_model, 
                    modelConfig['Attacker'], modelConfig['adver_scale'], 
                    threat_name=modelConfig['threat_name'],
                    epsilon=modelConfig['eps'], resume=modelConfig['save_intermediate_result'], 
                    mask=(1-mask) if modelConfig['useCAM'] else 1, 
                    use_lpips=modelConfig['perturbxp'],
                    originalx=kwargs['images'], lpips_fn=kwargs['lpips'],
                    lpips_scale=modelConfig['lpips_scale']) # mask=(1-mask)
                delta = kw['delta']*2
                xt_prev_mean = xt_prev_mean + delta
        return xt_prev_mean, var, pred_xstart, kw

    def get_new_label(self, pred, labels):
        pred[range(len(pred)), labels-1] = -99999
        v, idx = torch.max(pred, 1)
        return idx

    def p_mean_variance_label(self, x_t, t, labels, attack_model=None, kwargs=None):
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps).clamp(-1, 1)

        pred = attack_model((pred_xstart+1)/2)
        new_label = self.get_new_label(pred, labels)
        new_label += 1 # label should +1 since it's train in [1,10]
        eps = self.model(x_t, t, new_label)
        nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        eps = (1. + self.w) * eps - self.w * nonEps
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps).clamp(-1, 1)

        xt_prev_mean = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x_t, t=t)
        return xt_prev_mean, var, pred_xstart

    def forward(self, x_T, labels, model=None, start_T=None, show=False, kwargs=None, ):
        """
        Algorithm 2.
        """
        x_t = x_T
        modelConfig = kwargs['modelConfig']
        if modelConfig['Attacker'] in ['lafeat', 'pgd']:
            p_func = self.p_mean_variance 
        elif modelConfig['Attacker'] == 'label':
            p_func = self.p_mean_variance_label
        else:
            raise ValueError("No such Attacker, plz choose  'lafeat', 'pgd' or 'label'")
        acc_cure, logit_cure = [], []
        for time_step in tqdm(reversed(range(start_T))):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var, pred_xstart, kw = p_func(x_t=x_t, t=t, labels=labels, attack_model=model, kwargs=kwargs)
            if show and model is not None:
                if kw is not None: 
                    acc_cure.append(kw['init_acc'])
                    acc_cure.append(kw['acc_after'])
                if kw is not None: 
                    logit_cure.append(kw['init_logit'])
                    logit_cure.append(kw['logit_after'])
                # if (time_step+1) % 10 == 0:
                #     pred_xstart = add_border(pred_xstart, ~kw['flag'])
                #     save_image(pred_xstart, os.path.join(kwargs['path'], f'xp{time_step}.png'), 
                #         nrow=8,  normalize=True, range=(-1, 1))
                #     save_image(mean, os.path.join(kwargs['path'], f'x{time_step}.png'), 
                #         nrow=8,  normalize=True, range=(-1, 1))
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1), acc_cure 


