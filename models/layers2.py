import torch.nn as nn


def crop_like(input, target):
    if input.size()[2:] == target.size()[2:]:
        return input
    else:
        return input[:, :, : target.size(2), : target.size(3)]


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 3, kernel_size=3, stride=1, padding=1, bias=False)


def conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), padding=(1, 1, 1), stride=(1, 1, 1), dilation=(1, 1, 1), bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.输出形状等于输入形状
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size[0] - 1) * (dilation[0] - 1) + kernel_size[0]
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(same_padding, same_padding, same_padding),
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )
    
def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=True):
    """`same` convolution with LeakyReLU, i.e. output shape equals input shape.
  Args:
    in_planes (int): The number of input feature maps.
    out_planes (int): The number of output feature maps.
    kernel_size (int): The filter size.
    dilation (int): The filter dilation factor.
    stride (int): The filter stride.
  """
    # compute new filter size after dilation
    # and necessary padding for `same` output size
    dilated_kernel_size = (kernel_size - 1) * (dilation - 1) + kernel_size
    same_padding = (dilated_kernel_size - 1) // 2

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=same_padding,
            dilation=dilation,
            bias=bias,
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )

def tac_deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_planes, out_planes, kernel_size=4, stride=2, padding=1, bias=False
        ),
        nn.LeakyReLU(0.1, inplace=True),
    )


def tac_predict(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=False)



class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Flatten(nn.Module):
    """Flattens convolutional feature maps for fc layers.
  """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.reshape(x.size(0), -1)


class CausalConv1D(nn.Conv1d):
    """A causal 1D convolution.
  """

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, initailize_weights=True, device='cuda'
    ):
        self.__padding = (kernel_size - 1) * dilation
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            bias=bias,
        )
        self.leaky_relu = nn.LeakyReLU(0.1, inplace=True).cuda()
        if initailize_weights:
            nn.init.kaiming_normal_(self.weight.data)
            if self.bias is not None:
                self.bias.data.zero_()
        self.to(device)
        
    def forward(self, x):
        res = super(CausalConv1D, self).forward(x)
        if self.__padding != 0:
            return self.leaky_relu(res[:, :, : -self.__padding]), res[:, :, : -self.__padding]
        return self.leaky_relu(res), res


class ResidualBlock(nn.Module):
    """A simple residual block.
  """

    def __init__(self, channels):
        super().__init__()

        self.conv1 = conv3d(channels, channels, bias=False)
        self.conv2 = conv3d(channels, channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)  # nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.act(x)
        out = self.act(self.bn1(self.conv1(out)))
        out = self.bn2(self.conv2(out))
        return out + x
