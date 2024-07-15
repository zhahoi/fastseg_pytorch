import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

import os
import torch.utils.model_zoo as model_zoo

class _ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu6=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        self.bn = norm_layer(out_channels)
        self.relu = nn.ReLU6(True) if relu6 else nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SEModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SEModule, self).__init__()
        expand_size =  max(in_size // reduction, 8)
        
        self.fc = nn.Sequential(
            nn.Linear(in_size, expand_size, bias=False),
            nn.ReLU(True),
            nn.Linear(expand_size, in_size, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        x_se = x.mean((2, 3), keepdim=True)
        out = x_se.view(n, c)
        out = self.fc(out).view(n, c, 1, 1)
        
        return x * out.expand_as(x)
        

class InvertedResidual(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride, padding, dilation=1):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)
        self.se = SEModule(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        
        if self.skip is not None:
            skip = self.skip(skip)

        return self.act3(out + skip)


class _DWConvBNReLU(nn.Module):
    """Depthwise Separable Convolution in MobileNet.
    depthwise convolution + pointwise convolution
    """

    def __init__(self, in_channels, dw_channels, out_channels, stride, dilation=1, norm_layer=nn.BatchNorm2d, **kwargs):
        super(_DWConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            _ConvBNReLU(in_channels, dw_channels, 3, stride, dilation, dilation, in_channels, norm_layer=norm_layer),
            _ConvBNReLU(dw_channels, out_channels, 1, norm_layer=norm_layer))

    def forward(self, x):
        return self.conv(x)


class MobileV3Small(nn.Module):
    def __init__(self, num_classes=3, num_filters=128, act=nn.Hardswish):
        super(MobileV3Small, self).__init__()
        # 1, 16, 208, 208
        self.early = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act(inplace=True)
        )
        # 1, 16, 104, 104
        self.block0 = nn.Sequential(
            _DWConvBNReLU(16, 16, 16, 2, 1)
        )
        # 1, 24, 52, 52
        self.block1 = nn.Sequential(
            InvertedResidual(3, 16, 72, 24, nn.ReLU, False, 2, 1, 1),
            InvertedResidual(3, 24, 88, 24, nn.ReLU, False, 1, 1, 1)
        )
        # 1, 40, 52, 52
        self.block2 = nn.Sequential(
            InvertedResidual(5, 24, 96, 40, act, True, 1, 4, 2),
            InvertedResidual(5, 40, 240, 40, act, True, 1, 4, 2),
            InvertedResidual(5, 40, 240, 40, act, True, 1, 4, 2)
        )
        # 1, 48, 52, 52
        self.block3 = nn.Sequential(
            InvertedResidual(5, 40, 120, 48, act, True, 1, 4, 2),
            InvertedResidual(5, 48, 144, 48, act, True, 1, 4, 2),
        )
        # 1, 96, 52, 52
        self.block4 = nn.Sequential(
            InvertedResidual(5, 48, 288, 96, act, True, 1, 8, 4),
            InvertedResidual(5, 96, 576, 96, act, True, 1, 8, 4),
            InvertedResidual(5, 96, 576, 96, act, True, 1, 8, 4),
        )
        # 1, 576, 52, 52
        self.block5 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, dilation=4, bias=False),
            nn.BatchNorm2d(576),
            act(inplace=True)
        )
        
        # LRASPP
        self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(576, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
        self.aspp_conv2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
            nn.Conv2d(576, num_filters, 1, bias=False),
            nn.Sigmoid(),
        )
        
        self.convs2 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(16, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.conv_up2 = _ConvBNReLU(num_filters + 64, num_filters, kernel_size=1)
        self.conv_up3 = _ConvBNReLU(num_filters + 32, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        s2 = self.early(x)  # 2x
        s4 = self.block0(s2)  # 2x
        block1 = self.block1(s4)  # 4x
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        final = self.block5(block4)
        
        # lr_aspp
        aspp =  self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return y
    

class MobileV3Large(nn.Module):
    def __init__(self, num_classes=3, num_filters=128, act=nn.Hardswish):
        super(MobileV3Large, self).__init__()
        # 1, 16, 208, 208
        self.early = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act(inplace=True)
        )
        # 1, 16, 208, 208
        self.block0 = nn.Sequential(
            _DWConvBNReLU(16, 16, 16, 1, 1)
        )
        # 1, 24, 104, 104
        self.block1 = nn.Sequential(
            InvertedResidual(3, 16, 64, 24, nn.ReLU, False, 2, 1, 1),
            InvertedResidual(3, 24, 72, 24, nn.ReLU, False, 1, 1, 1)
        )
        # 1, 40, 52, 52
        self.block2 = nn.Sequential(
            InvertedResidual(5, 24, 72, 40, nn.ReLU, True, 2, 2, 1),
            InvertedResidual(5, 40, 120, 40, nn.ReLU, True, 1, 2, 1),
            InvertedResidual(5, 40, 120, 40, nn.ReLU, True, 1, 2, 1)
        )
        # 1, 80, 52, 52
        self.block3 = nn.Sequential(
            InvertedResidual(3, 40, 240, 80, act, False, 1, 2, 2),
            InvertedResidual(3, 80, 200, 80, act, False, 1, 2, 2),
            InvertedResidual(3, 80, 184, 80, act, False, 1, 2, 2),
            InvertedResidual(3, 80, 184, 80, act, False, 1, 2, 2)
        )
        # 1, 112, 52, 52
        self.block4 = nn.Sequential(
            InvertedResidual(3, 80, 480, 112, act, True, 1, 2, 2),
            InvertedResidual(3, 112, 672, 112, act, True, 1, 2, 2)
        )
        # 1, 160, 52, 52
        self.block5 = nn.Sequential(
            InvertedResidual(5, 112, 672, 160, act, True, 1, 8, 4),
            InvertedResidual(5, 160, 672, 160, act, True, 1, 8, 4),
            InvertedResidual(5, 160, 960, 160, act, True, 1, 8, 4)
        )
        # 1, 960, 52, 52
        self.block6 = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, dilation=4, bias=False),
            nn.BatchNorm2d(960),
            act(inplace=True)
        )
        
        # LRASPP
        self.aspp_conv1 = nn.Sequential(
                nn.Conv2d(960, num_filters, 1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
            )
        self.aspp_conv2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(49, 49), stride=(16, 20)),
            nn.Conv2d(960, num_filters, 1, bias=False),
            nn.Sigmoid(),
        )
        
        self.convs2 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.convs4 = nn.Conv2d(24, 64, kernel_size=1, bias=False)
        self.conv_up1 = nn.Conv2d(num_filters, num_filters, kernel_size=1)
        self.conv_up2 = _ConvBNReLU(num_filters + 64, num_filters, kernel_size=1)
        self.conv_up3 = _ConvBNReLU(num_filters + 32, num_filters, kernel_size=1)
        self.last = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.early(x)  
        s2 = self.block0(out)  # 2x
        s4 = self.block1(s2)  # 4x
        block2 = self.block2(s4)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        final = self.block6(block5) 
        
        # lr_aspp
        aspp =  self.aspp_conv1(final) * F.interpolate(
                self.aspp_conv2(final),
                final.shape[2:],
                mode='bilinear',
                align_corners=True
            )
        
        y = self.conv_up1(aspp)
        y = F.interpolate(y, size=s4.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs4(s4)], 1)
        y = self.conv_up2(y)
        y = F.interpolate(y, size=s2.shape[2:], mode='bilinear', align_corners=False)

        y = torch.cat([y, self.convs2(s2)], 1)
        y = self.conv_up3(y)
        y = self.last(y)
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return y
    
    
def load_url(url, model_dir='./model_data', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if os.path.exists(cached_file):
        return torch.load(cached_file, map_location=map_location)
    else:
        return model_zoo.load_url(url,model_dir=model_dir)


# 假的代码，因为没有预训练的模型
def fastseg(pretrained=True, large=True, num_classes=3, num_filters=128):
    if large and pretrained:
        model = MobileV3Large(num_classes=num_classes, num_filters=num_filters)
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth'), strict=False)
    else:
        model = MobileV3Small(num_classes=num_classes, num_filters=num_filters)
        model.load_state_dict(load_url('https://github.com/bubbliiiing/deeplabv3-plus-pytorch/releases/download/v1.0/xception_pytorch_imagenet.pth'), strict=False)
    return model


if __name__ == '__main__':
    model = MobileV3Small(num_classes=3)
    input = torch.rand(1, 3, 416, 416)
    # low, mid, x = model(input)
    # summary(model, (1, 3, 416, 416))
    x = model(input)
    print(model)
    print(sum(p.numel() for p in model.parameters()), ' parameters')
    print(x.size())