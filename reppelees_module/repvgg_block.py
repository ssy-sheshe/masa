import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
import numpy as np

class RepVGG3x3Block(BaseModule):
    def __init__(self, C_in, C_out, stride, group=1,
                conv_cfg=None, norm_cfg=dict(type='BN'), repvgg_converted=False, name=''):
        super(RepVGG3x3Block, self).__init__()
        self.name = name
        self.C_in = C_in
        self.C_out = C_out
        self.group = group
        self.stride = stride
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.repvgg_converted = repvgg_converted
        self.nonlinearity = nn.ReLU(inplace=True)
        self.init_rbr()

    def init_rbr(self):
        if self.repvgg_converted:
            self.rbr_reparam = ConvModule(in_channels=self.C_in,
                                        out_channels=self.C_out,
                                        kernel_size=3,
                                        padding=1,
                                        stride=self.stride,
                                        conv_cfg=self.conv_cfg,
                                        norm_cfg=None,
                                        act_cfg=None)

        else:
            self.rbr_identity = nn.BatchNorm2d(self.C_in) if self.C_out == self.C_in and self.stride == 1 else None
            
            self.rbr_dense = ConvModule(in_channels=self.C_in,
                                        out_channels=self.C_out,
                                        kernel_size=3,
                                        padding=1,
                                        stride=self.stride,
                                        conv_cfg=self.conv_cfg,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=None)
            
            self.rbr_1x1 = ConvModule(in_channels=self.C_in,
                                        out_channels=self.C_out,
                                        kernel_size=1,
                                        padding=0,
                                        stride=self.stride,
                                        conv_cfg=self.conv_cfg,
                                        norm_cfg=self.norm_cfg,
                                        act_cfg=None)

    def forward(self, x):
        inputs = x
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        # print(branch)
        if branch is None:
            return 0, 0
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.C_in // self.group
                kernel_value = np.zeros((self.C_in, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.C_in):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def repvgg_convert(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        return kernel.detach(), bias.detach()
if __name__ == '__main__':
    rep_train = RepVGG3x3Block(16, 16, 2, conv_cfg=None, repvgg_converted=False, name='rep_train').cuda()
    rep_deploy = RepVGG3x3Block(16, 16, 2, conv_cfg=None, repvgg_converted=True, name='rep_deploy').cuda()
    kernel, bias = rep_train.repvgg_convert()
    all_weights = {}
    all_weights['rbr_reparam.conv.weight'] = kernel
    all_weights['rbr_reparam.conv.bias'] = bias
    rep_deploy.load_state_dict(all_weights, strict=False)
    img = torch.randn(1, 16, 4, 4).cuda()
    feat_train = rep_train(img)
    print(feat_train)
    feat_deploy = rep_deploy(img)
    print(feat_deploy)