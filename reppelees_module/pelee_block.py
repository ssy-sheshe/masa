import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

class PeleeDenseBlock(BaseModule):
    def __init__(self, C_in, C_out, stride, conv_cfg=None, norm_cfg=dict(type='BN'), act_cfg=dict(type='ReLU'), name=''):
        super(PeleeDenseBlock, self).__init__()

        self.name = name
        self.C_in = C_in
        self.stride = stride
        self.C_out = C_out
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        if self.C_out <= 128:
            out_tmp = 16
        elif self.C_out > 128 and self.C_out <= 256:
            out_tmp = 32
        else:
            out_tmp = 64

        self.conv_a_1x1 = ConvModule(in_channels=self.C_in,
                            out_channels=out_tmp,
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        self.conv_a_3x3 = ConvModule(in_channels=out_tmp,
                            out_channels=16,
                            kernel_size=3,
                            padding=1,
                            stride=self.stride,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        self.conv_b_1x1 = ConvModule(in_channels=self.C_in,
                            out_channels=out_tmp,
                            kernel_size=1,
                            padding=0,
                            stride=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        
        self.conv_b_3x3_0 = ConvModule(in_channels=out_tmp,
                            out_channels=16,
                            kernel_size=3,
                            padding=1,
                            stride=self.stride,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)
        self.conv_b_3x3_1 = ConvModule(in_channels=16,
                            out_channels=16,
                            kernel_size=3,
                            padding=1,
                            stride=1,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg,
                            act_cfg=self.act_cfg)

    def forward(self, x):
        outa = self.conv_a_1x1(x)
        outa = self.conv_a_3x3(outa)

        outb = self.conv_b_1x1(x)
        outb = self.conv_b_3x3_0(outb)
        outb = self.conv_b_3x3_1(outb)
        y = torch.cat((x, outa, outb), 1)
        return y

if __name__ == '__main__':
    pelee_0 = PeleeDenseBlock(16, 16, 1, conv_cfg=None, name='rep_0').cuda()
    img = torch.randn(1, 16, 640, 480).cuda()
    feat = pelee_0(img)
    print(feat.size())