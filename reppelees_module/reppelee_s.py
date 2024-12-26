import sys
import torch
import torch.nn as nn
from collections import OrderedDict
# sys.path.append('/face/ykhu/BHRFD/git/mmdetection/')
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from mmengine.model import constant_init, kaiming_init
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.logging import MMLogger
from .repvgg_block import RepVGG3x3Block
from .pelee_block import PeleeDenseBlock
from mmdet.registry import MODELS

@MODELS.register_module()
class RepPeleeNetS(BaseModule):
    def __init__(self, 
        out_indices=(1, 2, 3),
        conv_cfg=None, 
        norm_cfg=dict(type='BN'), 
        act_cfg=dict(type='ReLU'), 
        repvgg_converted=False,
        pretrained=None,
        freeze_backbone=False,
        freeze_bn=False):
        super(RepPeleeNetS, self).__init__()

        self.out_indices = out_indices
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.repvgg_converted = repvgg_converted
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = freeze_bn

        self.stage0_0 = RepVGG3x3Block(3, 16, 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_0')
        self.stage0_1 = RepVGG3x3Block(16, 16, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_1')
        self.stage0_2 = RepVGG3x3Block(16, 16, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_2')
        self.stage0_3 = RepVGG3x3Block(16, 16, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_3')
        self.stage0_4 = RepVGG3x3Block(16, 16, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_4')

        self.stage1_0 = RepVGG3x3Block(16, 32, 2, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_5')
        self.stage1_1 = RepVGG3x3Block(32, 32, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_6')
        self.stage1_2 = RepVGG3x3Block(32, 32, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_7')
        self.stage1_3 = RepVGG3x3Block(32, 32, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_8')
        self.stage1_4 = RepVGG3x3Block(32, 32, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, repvgg_converted=self.repvgg_converted, name='rep_9')

        self.stage2_0 = PeleeDenseBlock(32, 64, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_0')
        self.stage3_0 = PeleeDenseBlock(64, 96, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_1')
        self.stage4_0 = PeleeDenseBlock(96, 128, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_2')
        self.stage5_0 = ConvModule(128, 128, 1, 1, 0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        
        self.stage7_0 = PeleeDenseBlock(128, 160, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_3')
        self.stage8_0 = PeleeDenseBlock(160, 192, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_4')
        self.stage9_0 = PeleeDenseBlock(192, 224, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_5')
        self.stage10_0 = PeleeDenseBlock(224, 256, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_6')
        self.stage11_0 = ConvModule(256, 256, 1, 1, 0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.stage13_0 = PeleeDenseBlock(256, 288, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_7')
        self.stage14_0 = PeleeDenseBlock(288, 320, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_8')
        self.stage15_0 = PeleeDenseBlock(320, 352, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_9')
        self.stage16_0 = PeleeDenseBlock(352, 384, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_10')
        self.stage17_0 = ConvModule(384, 384, 1, 1, 0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.stage19_0 = PeleeDenseBlock(384, 416, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_11')
        self.stage20_0 = PeleeDenseBlock(416, 448, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_12')
        self.stage21_0 = PeleeDenseBlock(448, 480, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_13')
        self.stage22_0 = PeleeDenseBlock(480, 512, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, name='pelee_14 ')
        self.stage23_0 = ConvModule(512, 512, 1, 1, 0, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

        self.avgk2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self._init_weights()
    
    def _init_weights(self):

        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            load_checkpoint(self, self.pretrained, strict=False, logger=logger)
        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

        if self.freeze_backbone:
            self._freeze_backbone()
        if self.freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.SyncBatchNorm):
                print('freeze bn !!!!!!!!!!!!!!!!!')
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def _freeze_backbone(self):
        print('freeze backbone !!!!!!!!!!!!!!!!!')
        for m in self.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        output_layers = []
        x = self.stage0_0(x)
        x = self.stage0_1(x)
        x = self.stage0_2(x)
        x = self.stage0_3(x)
        x = self.stage0_4(x)

        x = self.stage1_0(x)
        x = self.stage1_1(x)
        x = self.stage1_2(x)
        x = self.stage1_3(x)
        x = self.stage1_4(x)

        x = self.stage2_0(x)
        x = self.stage3_0(x)
        x = self.stage4_0(x)
        x0 = self.stage5_0(x)
        output_layers.append(x0)

        x = self.avgk2(x0)
        x = self.stage7_0(x)
        x = self.stage8_0(x)
        x = self.stage9_0(x)
        x = self.stage10_0(x)
        x1 = self.stage11_0(x)
        output_layers.append(x1)
        # print(x1)

        x = self.avgk2(x1)
        x = self.stage13_0(x)
        x = self.stage14_0(x)
        x = self.stage15_0(x)
        x = self.stage16_0(x)
        x2 = self.stage17_0(x)
        output_layers.append(x2)
        # print(x2)

        x = self.avgk2(x2)
        x = self.stage19_0(x)
        x = self.stage20_0(x)
        x = self.stage21_0(x)
        x = self.stage22_0(x)
        x3 = self.stage23_0(x)
        output_layers.append(x3)
        # print(x3)

        outs = []
        for i, x in enumerate(output_layers):
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
    

def clean_dict(model, state_dict):
        """
        This function is created for processing state dict
        for loading pth in distributed trainer.
        """
        _state_dict = OrderedDict()

        for (k0, v0), (k1,v1) in zip(state_dict.items(), model.state_dict().items()):
            print(k0, v0.shape, k1, v1.shape)
            if v0.size() == v1.size():
                _state_dict[k1] = v0
                print(' : load {} - {}'.format(k1, v0.shape), 'green')
            else:
                 print(' : cannot load {} - {}'.format(k1, v0.shape), 'red')
        return _state_dict

if __name__ == '__main__':
    net = RepPeleeNetS(pretrained='/face/ykhu/BHRFD/git/mmdetection/resources/reppelees.pth', freeze_backbone=True).cuda()
    # checkpoint = torch.load('/face/ykhu/BHRFD/git/latest_version/log/pelee4_rep_imagenet/20221010-110657/model_final.pth', map_location='cpu')
    # state_dict = checkpoint['model']
    # state_dict = clean_dict(net, state_dict)
    # net.load_state_dict(state_dict, strict=True)
    # torch.save(net.state_dict(), '/face/ykhu/BHRFD/git/mmdetection/resources/reppelees.pth')
    img = torch.randn(1, 3, 576, 1024).cuda()
    feat = net(img)
    print(feat[0].size())
    print(feat[1].size())
    print(feat[2].size())
        