import mindspore.nn as nn
from mindspore.common.initializer import initializer
import mindspore.common.dtype as mstype
import ops
import math
from mindspore import Tensor
import numpy as np

def _make_layer(in_channels, base, batch_norm):
    """Make stage network of VGG."""
    layers = []
    in_channels = in_channels
    for v in base:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            weight_shape = (v, in_channels, 3, 3)
            #fan = v * 3 * 3
            #gain = math.sqrt(2.0)
            #std = gain / math.sqrt(fan)
            #weight = Tensor(np.random.normal(0, std, weight_shape), mstype.float32)
            weight = initializer('HeUniform', shape=weight_shape, dtype=mstype.float32).to_tensor() 
            conv2d = nn.Conv2d(in_channels=in_channels,
                               out_channels=v,
                               kernel_size=3,
                               padding=1,
                               has_bias=True,
                               pad_mode='pad',
                               weight_init=weight,
                               bias_init='zeros')
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.SequentialCell(layers)



class Vgg(nn.Cell):
    """
    VGG network definition.

    Args:
        base (list): Configuration for different layers, mainly the channel number of Conv layer.
        num_classes (int): Class numbers. Default: 1000.
        batch_norm (bool): Whether to do the batchnorm. Default: False.
        batch_size (int): Batch size. Default: 1.

    Returns:
        Tensor, infer output tensor.

    Examples:
        >>> Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        >>>     num_classes=1000, batch_norm=False, batch_size=1)
    """

    def __init__(self, base, batch_norm=False):
        super(Vgg, self).__init__()
        '''
        self.conv1_1 = ops.conv_bn_relu(3, 64, 1, 3)
        self.conv1_2 = ops.conv_bn_relu(64, 64, 1, 3)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ops.conv_bn_relu(64, 128, 1, 3)
        self.conv2_2 = ops.conv_bn_relu(128, 128, 1, 3)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv3_1 = ops.conv_bn_relu(128, 256, 1, 3)
        self.conv3_2 = ops.conv_bn_relu(256, 256, 1, 3)
        self.conv3_3 = ops.conv_bn_relu(256, 256, 1, 3)
        self.max3 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv4_1 = ops.conv_bn_relu(256, 512, 1, 3)
        self.conv4_2 = ops.conv_bn_relu(512, 512, 1, 3)
        self.conv4_3 = ops.conv_bn_relu(512, 512, 1, 3)
        self.max4 = nn.MaxPool2d(kernel_size=2, stride=2)


        self.conv5_1 = ops.conv_bn_relu(512, 512, 1, 3)
        self.conv5_2 = ops.conv_bn_relu(512, 512, 1, 3)
        self.conv5_3 = ops.conv_bn_relu(512, 512, 1, 3)
        self.max5 = nn.MaxPool2d(kernel_size=2, stride=2)
        '''

        self.layer1 = _make_layer(3, base[0], batch_norm)
        self.layer2 = _make_layer(64, base[1], batch_norm)
        self.layer3 = _make_layer(128, base[2], batch_norm)
        self.layer4 = _make_layer(256, base[3], batch_norm)
        self.layer5 = _make_layer(512, base[4], batch_norm)

    def construct(self, x):
        '''
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.max1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.max2(x)
        x2 = x

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.max3(x)
        x3 = x

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.max3(x)
        x4 = x
 
        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.max3(x)
        x5 = x
        '''

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x5 = self.layer5(x4)


        return [x5, x4, x3, x2]


cfg = {
    '16': [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512, 'M']],
}


def vgg16(batch_norm=True):
    """
    Get Vgg16 neural network with batch normalization.

    Args:
        num_classes (int): Class numbers. Default: 1000.

    Returns:
        Cell, cell instance of Vgg16 neural network with batch normalization.

    Examples:
        >>> vgg16(num_classes=1000)
    """

    net = Vgg(cfg['16'], batch_norm=True)
    return net

if __name__ == "__main__":
    import mindspore.context as context

    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=7
    )

    import numpy as np
    import mindspore
    inputs = np.random.randn(64, 3, 512, 512)
    inputs = mindspore.Tensor(inputs, mindspore.float32)
    network = vgg16()
    a,b = network(inputs)
    print(a.shape)
    for i in b:
        print(i.shape)