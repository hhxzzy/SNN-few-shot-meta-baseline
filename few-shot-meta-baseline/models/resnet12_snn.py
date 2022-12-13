import torch.nn as nn

from braincog.model_zoo.base_module import *
from braincog.base.node.node import *

from .models import register

from braincog.base.utils.visualization import spike_rate_vis, spike_rate_vis_1d


def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 3, padding=1, bias=False)


def conv1x1(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, 1, bias=False)


def norm_layer(planes):
    return nn.BatchNorm2d(planes)


class Block(nn.Module):

    def __init__(self, inplanes, planes, downsample, node=IFNode, act_fun=AtanGrad):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)

        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = norm_layer(inplanes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = norm_layer(planes)

        self.node1 = node(act_fun=act_fun)
        self.node2 = node(act_fun=act_fun)
        # self.node3 = node()

        self.downsample = downsample

        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):                
                       
        out = self.bn1(x)
        out = self.node1(out)
        out = self.conv1(out)
        
        out = self.bn2(out)
        out = self.node2(out)
        out = self.conv2(out)

        # 这里可能会有问题
        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class ResNet12_snn(BaseModule):

    def __init__(self, channels, step, encode_type, node, act_fun):
        super().__init__(step, encode_type)

        self.inplanes = 3
        
        self.node = eval(node)
        self.act_fun = act_fun

        self.layer1 = self._make_layer(channels[0])
        self.layer2 = self._make_layer(channels[1])
        self.layer3 = self._make_layer(channels[2])
        self.layer4 = self._make_layer(channels[3])

        self.out_dim = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes):
        downsample = nn.Sequential(
            norm_layer(self.inplanes),
            self.node(act_fun=self.act_fun),
            conv1x1(self.inplanes, planes),            
        )
        block = Block(self.inplanes, planes, downsample, node=self.node, act_fun = self.act_fun)
        self.inplanes = planes
        return block

    def forward(self, inputs):    
        inputs = self.encoder(inputs)
        self.reset()

        outputs = []
        step = self.step
        for t in range(step):
            x = inputs[t] 
             
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
                        

            x = x.view(x.shape[0], x.shape[1], -1).mean(dim=2)
            
            outputs.append(x)
        ret = sum(outputs) / len(outputs)
        return ret


@register('resnet12_snn')
def resnet12_snn(**kwargs):
    step = kwargs['step']
    encode_type = kwargs['encode_type']
    node = kwargs['node_type']
    act_fun = kwargs['act_fun']    
    return ResNet12_snn([64, 128, 256, 512], step, encode_type, node, act_fun)


@register('resnet12_snn-wide')
def resnet12_snn_wide(**kwargs):
    step = kwargs['step']
    encode_type = kwargs['encode_type']
    node = kwargs['node_type']
    act_fun = kwargs['act_fun']
    return ResNet12_snn([64, 160, 320, 640], step, encode_type, node, act_fun)

