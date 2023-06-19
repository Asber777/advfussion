import os
import lpips
import torch as th
from pytorch_msssim import ssim
def get_lp_lpips_ssim_between_noattack():
    TENSOR_NUM = 100
    BATCH = 5
    device = th.device('cuda')
    for t in range(0, 93, 4):
        loss_fn_alex = lpips.LPIPS(net='alex')
        loss_fn_alex = loss_fn_alex.to(device)
        target_err_total = th.tensor(0.0).to(device)
        eps_total = th.tensor(0.0).to(device)
        lpips_total = th.tensor(0.0).to(device)
        ssim_total = th.tensor(0.0).to(device)
        path1 = f'/root/hhtpro/123/result/inpainting/diff_t2_s/{t}'
        path2 = '/root/hhtpro/123/result/inpainting/diff_t2_s_no_attack'
        path1 = os.path.join(path1, 'result')
        path2 = os.path.join(path2, 'result')
        for i in range(BATCH, TENSOR_NUM+1, BATCH):
            tensor_name = '{:0>5d}.pt'.format(i)
            ResAdv = th.load(os.path.join(path1, tensor_name))
            label = ResAdv['y']
            at_predict = ResAdv['predict']
            sample = ResAdv['sample']
            img = th.load(os.path.join(path2, tensor_name))['sample']
            img = th.clamp((img+1)/2, 0, 1)
            budget = th.abs((sample+1)/2 - img).reshape(len(sample), -1).max(dim = -1)[0]
            err_mask = (at_predict.data != label.data)
            target_err_total += err_mask.float().sum()
            distance_batch = budget[err_mask]
            lpips_batch = loss_fn_alex.forward(img*2-1, sample)[err_mask]
            ssim_batch = ssim(img, (sample+1)/2, data_range=1., size_average=False)[err_mask]
            eps_total += distance_batch.sum()
            lpips_total += lpips_batch.sum()
            ssim_total += ssim_batch.sum()
        print(f'diff_t2_s:{t}')
        print(f'Avg distance of successfully transferred: {eps_total / target_err_total}')
        print(f'Avg LPIPS dis: {lpips_total / target_err_total}')
        print(f'Avg SSIM dis: {ssim_total / target_err_total}')

