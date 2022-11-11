import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2'
print(torch.cuda.device_count())
a = torch.randn(19)
a.to(torch.device('cuda:2'))