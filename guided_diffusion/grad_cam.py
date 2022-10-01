import os
import torch 
from torch import nn
from torchvision import utils
from robustbench.utils import load_model
from torch.nn.functional import relu
from torchvision.transforms import Resize


"""

在进行和原图相似逼近的时候可以用到
相似逼近的3个trick
1. 从xt开始 节省很多计算时间
2. ilvr和原图的xt进行比较 对于CAM小于一定阈值的 我们只需要色块相似即可
3. 对于CAM大于一定阈值的, 我们保持和原图一致 使用q(xt-1|x0)放上去
逼近之后, 在进行x0的对抗攻击的时候可以用到

"""

class GradCAM(object):
    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.gradient = None
        self.net.eval()
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output
        
    def _get_grads_hook(self, module, input_grad, output_grad):
        """
        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple, len(tuple)=1
        :return:
        """
        self.gradient = output_grad[0]

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        """
        :param inputs: [B,3,H,W]
        :param index: class id
        :return:
        """
        inputs = inputs.detach().clone()
        with torch.enable_grad():
            inputs.requires_grad_()
            self.net.zero_grad()
            _, _, H, W = inputs.shape
            output = self.net(inputs)  # [1,num_classes]
            index = output.argmax(dim=-1)
            target = output[range(len(index)), index].sum()
            target.backward()
            gradient = self.gradient  # [B,C,H,W]
            weight = torch.mean(gradient, dim=[2,3]) # [B,C]
            feature = self.feature  # [B,C,H,W]
            cam = feature * weight[:,:,None,None]  # [B,C,H,W]
            cam = relu(torch.sum(cam, dim=1))  # [B,H,W]
            # cam -= torch.min(torch.min(cam,dim=1)[0], dim=1)[0] # 大概率是0
            cam /= torch.max(torch.max(cam,dim=1)[0], dim=1)[0].reshape(-1,1,1)
            resize = Resize((H, W))
        return resize(cam)


class GradCamPlusPlus(GradCAM):
    def __init__(self, net, layer_name):
        super(GradCamPlusPlus, self).__init__(net, layer_name)

    def __call__(self, inputs, index=None):
        """
        :param inputs: [B,3,H,W]
        :param index: class id
        :return:
        """
        inputs = inputs.detach().clone()
        with torch.enable_grad():
            inputs.requires_grad_()
            N, C, H, W = inputs.shape
            self.net.zero_grad()
            output = self.net(inputs)  # [B,num_classes]
            index = output.argmax(dim=-1)
            target = output[range(N), index].sum()
            target.backward()
            gradient = relu(self.gradient)  # [B,C,H,W]
            indicate = torch.where(gradient > 0, 1., 0.)  # 示性函数
            norm_factor = torch.sum(gradient, dim=[2,3])  # [B,C]归一化
            norm_factor = torch.where(norm_factor>0,  
                1/norm_factor, torch.zeros_like(norm_factor))
            alpha = indicate * norm_factor[...,None,None]  # [B,C,H,W]
            weight = torch.sum(gradient * alpha, dim=[2,3])  # [C]  alpha*ReLU(gradient)
            cam = self.feature * weight[..., None, None]  # [B,C,H,W]*[B,C,H,W]
            cam = torch.sum(cam, dim=1)  # [B,H,W]
            # cam = np.maximum(cam, 0)  # ReLU
            cam -=  torch.min(torch.min(cam,dim=1)[0], dim=1)[0].reshape(-1,1,1)
            cam /=  torch.max(torch.max(cam,dim=1)[0], dim=1)[0].reshape(-1,1,1)
            resize = Resize((H, W))
        return resize(cam)


def get_last_conv_name(net):
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

# def main():
#     bs= 5
#     data = load_imagenet_batch(bs, '/root/hhtpro/123/imagenet')
#     net = load_model(model_name='Standard_R50',\
#         model_dir='root/hhtpro/123',dataset='imagenet',\
#             threat_model='Linf')
#     layer_name = get_last_conv_name(net)
#     grad_cam = GradCamPlusPlus(net, layer_name)
#     for i in range(2):
#         x, y = next(data)
#         x = (x+1)/2.
#         x.requires_grad_()
#         mask = grad_cam(x)  # cam mask
#         # gbp = GuidedBackPropagation(net)
#         # x.grad.zero_()  # 梯度置零
#         # grad = gbp(x)
#         # print(grad.shape, mask.shape) # B 3 H W * B 1 H W
#         # cam_gb = grad * mask.unsqueeze(1)
#         # print(cam_gb.shape)
#         # pic = mask.unsqueeze(1) * 0.5 + x * 0.5
#         pic = torch.where(mask.unsqueeze(1)>0.5, 1, 0) * x
#         for j in range(len(pic)):
#             out_path = os.path.join("/root/hhtpro/123/Grad-CAM.pytorch",
#                                     f"pic{str(i*bs+j).zfill(3)}.png")
#             utils.save_image(pic[j].unsqueeze(0), out_path, nrow=1, normalize=True, range=(0, 1),)
#     grad_cam.remove_handlers()

if __name__ == '__main__':
    main()
