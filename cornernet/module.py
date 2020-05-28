import torch.nn as nn
import torchvision.models as models
from ._cpools import TopPool, BottomPool, LeftPool, RightPool


#conv+bn+relu
#conv+relu
class convolution(nn.Module):

    def __init__(self, kernel, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(inp_dim, out_dim, (kernel, kernel), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn = nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn = self.bn(conv)
        relu = self.relu(bn)
        return relu


class residual(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(residual, self).__init__()

        self.conv1 = nn.Conv2d(inp_dim, out_dim, (3, 3), padding=(1, 1), stride=(stride, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_dim, out_dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)

        self.skip = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, (1, 1), stride=(stride, stride), bias=False),
            nn.BatchNorm2d(out_dim)
        ) if stride != 1 or inp_dim != out_dim else nn.Sequential()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)

        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)

        skip = self.skip(x)
        return self.relu(bn2 + skip)


def make_layer(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = [layer(k, inp_dim, out_dim, **kwargs)]
    for _ in range(1, modules):
        layers.append(layer(k, out_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_layer_revr(k, inp_dim, out_dim, modules, layer=convolution, **kwargs):
    layers = []
    for _ in range(modules - 1):
        layers.append(layer(k, inp_dim, inp_dim, **kwargs))
    layers.append(layer(k, inp_dim, out_dim, **kwargs))
    return nn.Sequential(*layers)


def make_pool_layer(dim):
    return nn.MaxPool2d(kernel_size=2, stride=2)


def make_unpool_layer(dim):
    return nn.Upsample(scale_factor=2)


class MergeUp(nn.Module):
    def forward(self, up1, up2):
        return up1 + up2


def make_merge_layer(dim):
    return MergeUp()


class backbone(nn.Module):
    def __init__(
        self, n, dims, modules, layer=residual,
        make_up_layer=make_layer, make_low_layer=make_layer,
        make_hg_layer=make_layer, make_hg_layer_revr=make_layer_revr,
        make_pool_layer=make_pool_layer, make_unpool_layer=make_unpool_layer,
        make_merge_layer=make_merge_layer, **kwargs
    ):
        super(backbone, self).__init__()

        self.n   = n    # 5

        curr_mod = modules[0]   # 2
        next_mod = modules[1]   # 2

        curr_dim = dims[0]      # 256
        next_dim = dims[1]      # 256

        self.up1  = make_up_layer(
            3, curr_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.max1 = make_pool_layer(curr_dim)
        self.low1 = make_hg_layer(
            3, curr_dim, next_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.low2 = backbone(
            n - 1, dims[1:], modules[1:], layer=layer,
            make_up_layer=make_up_layer,
            make_low_layer=make_low_layer,
            make_hg_layer=make_hg_layer,
            make_hg_layer_revr=make_hg_layer_revr,
            make_pool_layer=make_pool_layer,
            make_unpool_layer=make_unpool_layer,
            make_merge_layer=make_merge_layer,
            **kwargs
        ) if self.n > 1 else \
        make_low_layer(
            3, next_dim, next_dim, next_mod,
            layer=layer, **kwargs
        )
        self.low3 = make_hg_layer_revr(
            3, next_dim, curr_dim, curr_mod,
            layer=layer, **kwargs
        )
        self.up2  = make_unpool_layer(curr_dim)

        self.merge = make_merge_layer(curr_dim)

    def forward(self, x):
        up1  = self.up1(x)
        max1 = self.max1(x)
        low1 = self.low1(max1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.merge(up1, up2)
'''
class backbone(nn.Module):
    
    
    
    def __init__(self):
        super(backbone, self).__init__()

    def forward(self, x):
        x = self.conv(x)

        return x
'''

class tl_corner_pooling(nn.Module):

    def __init__(self, dim):
        super(tl_corner_pooling, self).__init__()
        self.standard_conv1 = convolution(3, dim, 128)
        self.standard_conv2 = convolution(3, dim, 128)

        self.top_pool = TopPool()
        self.left_pool = LeftPool()

        self.conv2 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        self.skip_conv = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.skip_bn = nn.BatchNorm2d(dim)

        self.relu1 = nn.ReLU(inplace=True)
        self.standard_conv3 = convolution(3, dim, dim)


    def forward(self, x):
        conv1 = self.standard_conv1(x)
        conv2 = self.standard_conv2(x)

        top_ = self.top_pool(conv1)
        left_ = self.left_pool(conv2)

        conv1_2 = self.conv2(top_ + left_)
        bn1_2 = self.bn2(conv1_2)

        skip_conv = self.skip_conv(x)
        skip_bn = self.skip_bn(skip_conv)

        out = self.relu1(skip_bn + bn1_2)
        out = self.standard_conv3(out)

        return out


class br_corner_pooling(nn.Module):

    def __init__(self, dim):
        super(br_corner_pooling, self).__init__()
        self.standard_conv1 = convolution(3, dim, 128)
        self.standard_conv2 = convolution(3, dim, 128)

        self.bottom_pool = BottomPool()
        self.right_pool = RightPool()

        self.conv2 = nn.Conv2d(128, dim, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(dim)

        self.skip_conv = nn.Conv2d(dim, dim, (1, 1), bias=False)
        self.skip_bn = nn.BatchNorm2d(dim)

        self.relu1 = nn.ReLU(inplace=True)
        self.standard_conv3 = convolution(3, dim, dim)

    def forward(self, x):
        conv1 = self.standard_conv1(x)
        conv2 = self.standard_conv2(x)

        bottom_ = self.bottom_pool(conv1)
        right_ = self.right_pool(conv2)

        conv1_2 = self.conv2(bottom_ + right_)
        bn1_2 = self.bn2(conv1_2)

        skip_conv = self.skip_conv(x)
        skip_bn = self.skip_bn(skip_conv)

        out = self.relu1(skip_bn + bn1_2)
        out = self.standard_conv3(out)

        return out

'''
class top_pool(nn.Module):

    def __init__(self):
        super(top_pool, self).__init__()


class left_pool(nn.Module):

    def __init__(self):
        super(left_pool, self).__init__()


class bottom_pool(nn.Module):

    def __init__(self):
        super(bottom_pool, self).__init__()


class right_pool(nn.Module):

    def __init__(self):
        super(right_pool, self).__init__()
'''


def out_conv_layer(cnv_dim, curr_dim, out_dim):

    return nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=False),
        nn.Conv2d(curr_dim, out_dim, (1, 1))
    )


