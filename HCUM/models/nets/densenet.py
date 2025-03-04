
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):

        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        # 拼接输入和新生成的特征（即特征重用）
        return torch.cat([x, new_features], 1)


# DenseBlock：由若干个 DenseLayer 级联组成，每个 DenseLayer 的输入均为前面所有层的拼接
class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):

        super(_DenseBlock, self).__init__()
        layers = []
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            )
            layers.append(layer)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))



class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        super(DenseNet, self).__init__()
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        ]))


        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                trans = _Transition(num_features, num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        dimension = 512 * 4
        feat_dim = 512
        self.head = nn.Sequential(
            nn.Linear(dimension, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim)
        )


        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        # 全局平均池化
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        feat_mlp = F.normalize((self.head(out)))
        out = self.classifier(out)
        return [out, feat_mlp]


def DenseNet121(num_classes=1000):
    """DenseNet-121: block_config=(6,12,24,16), growth_rate=32, num_init_features=64"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                    num_init_features=64, bn_size=4, drop_rate=0, num_classes=num_classes)


def DenseNet169(num_classes=1000):
    """DenseNet-169: block_config=(6,12,32,32), growth_rate=32, num_init_features=64"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 32, 32),
                    num_init_features=64, bn_size=4, drop_rate=0, num_classes=num_classes)


def DenseNet201(num_classes=1000):
    """DenseNet-201: block_config=(6,12,48,32), growth_rate=32, num_init_features=64"""
    return DenseNet(growth_rate=32, block_config=(6, 12, 48, 32),
                    num_init_features=64, bn_size=4, drop_rate=0, num_classes=num_classes)


def DenseNet161(num_classes=1000):
    """DenseNet-161: block_config=(6,12,36,24), growth_rate=48, num_init_features=96"""
    return DenseNet(growth_rate=48, block_config=(6, 12, 36, 24),
                    num_init_features=96, bn_size=4, drop_rate=0, num_classes=num_classes)



if __name__ == '__main__':
    model = DenseNet121(num_classes=1000)
    print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("输出形状：", y.shape)

