import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common.parameter import Parameter
from mindspore import Tensor
from mindspore.common.initializer import initializer
from mindspore._checkparam import check_int_positive
import math
import mindspore.common.dtype as mstype

#source code:
#   https://github.com/NaxAlpha/EAST/blob/master/pvanet.py

#should use different inittializer for different variables
def _weight_variable(shape, TYPE="RANDOM", factor=0.01):
    if TYPE == "ZERO":
        init_value = np.zeros(shape).astype(np.float32)
    elif TYPE == "ONE":
        init_value = np.ones(shape).astype(np.float32)
    elif TYPE == "RANDOM":
        init_value = np.random.randn(*shape).astype(np.float32) * factor
    else:
        raise ValueError("TYPE == ZERO | ONE | RANDOM")

    return Tensor(init_value)



def _conv(in_channel, out_channel, stride=1, kernel_size=1, padding='same', padding_number=0):
    weight_shape = (out_channel, in_channel, kernel_size, kernel_size) 
    fan = kernel_size * kernel_size * out_channel
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    #weight = mindspore.Tensor(np.random.normal(0, std, weight_shape), mindspore.float32)
    weight = initializer('HeUniform', shape=weight_shape, dtype=mstype.float32).to_tensor() 
    return nn.Conv2d(in_channel, out_channel,
                     kernel_size=kernel_size, stride=stride, padding=padding_number, pad_mode=padding, has_bias=True, weight_init=weight, bias_init='zeros')

class conv_bn_relu(nn.Cell):
    def __init__(self, in_channel, out_channel, stride=1, kernel_size=1, padding='same', padding_number=0):
        super(conv_bn_relu, self).__init__()
        self.conv = _conv(in_channel, out_channel, stride=stride, kernel_size=kernel_size, padding=padding, padding_number=padding_number)
        self.bn = nn.BatchNorm2d(num_features=out_channel)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def _scale(channel):
    return nn.BatchNorm2d(num_features=channel, eps=1.0, affine=True, moving_mean_init='zeros', 
                moving_var_init='zeros', gamma_init='ones', beta_init='zeros', use_batch_statistics=False)


class _bn_relu_conv(nn.Cell):
    def __init__(self, in_channel, out_channel, stride, kernel_size):
        super(_bn_relu_conv, self).__init__()
        self.batch_norm = nn.BatchNorm2d(num_features=in_channel)
        self.scale = _scale(in_channel)
        self.relu = nn.ReLU()
        self.conv = _conv(in_channel, out_channel, stride, kernel_size)

    def construct(self, x):
        x = self.batch_norm(x)
        x = self.scale(x)
        x = self.relu(x)
        x = self.conv(x)
        return x


class _conv_bn_relu(nn.Cell):
    def __init__(self, in_channel, out_channel, stride, kernel_size):
        super(_conv_bn_relu, self).__init__()
        in_channel = int(in_channel)
        out_channel = int(out_channel)
        self.conv = _conv(in_channel, out_channel, stride, kernel_size)
        self.batch_norm = nn.BatchNorm2d(num_features=out_channel)
        self.scale = _scale(out_channel)
        self.relu = nn.ReLU()
    
    def construct(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.scale(x)
        x = self.relu(x)
        return x

class _bn_crelu(nn.Cell):
    # outputs's channel = inputs'channel * 2
    def __init__(self, channel):
        super(_bn_crelu, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=channel)
        self.neg = P.Neg()
        self.concat = P.Concat(axis=1) #channel
        self.scale = _scale(channel * 2)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.bn(x)
        neg_x = self.neg(x)
        x = self.concat((x, neg_x))
        x = self.scale(x)
        x = self.relu(x)
        return x

class _conv_bn_crelu(nn.Cell):
    def __init__(self, in_channel, out_channel, stride, kernel_size):
        super(_conv_bn_crelu, self).__init__()
        self.conv = _conv(in_channel, out_channel, stride, kernel_size)
        self.bn_crelu = _bn_crelu(out_channel)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn_crelu(x)
        return x


class _bn_crelu_conv(nn.Cell):
    def __init__(self, in_channel, out_channel, stride, kernel_size):
        super(_bn_crelu_conv, self).__init__()
        self.bn_crelu = _bn_crelu(in_channel)
        self.conv = _conv(in_channel * 2, out_channel, stride, kernel_size)

    def construct(self, x):
        x = self.bn_crelu(x)
        x = self.conv(x)
        return x





class _mCReLU(nn.Cell):
    def __init__(self, in_channel, out_channels, stride, preact_bn=False):
        super(_mCReLU, self).__init__()
        conv1_fn = None
        if preact_bn:
            conv1_fn = _bn_relu_conv
        else:
            conv1_fn = _conv

        if len(out_channels) != 3:
            raise ValueError("out_channels should be list and len(out_channels) == 3")

        self.sub_conv1 = conv1_fn(in_channel, out_channels[0], stride, kernel_size=1)
        self.sub_conv2 = _bn_relu_conv(out_channels[0], out_channels[1], stride=1, kernel_size=3)
        self.sub_conv3 = _bn_crelu_conv(out_channels[1], out_channels[2], stride=1, kernel_size=1)

        if in_channel == out_channels[2]:
            self.last = None
        else:
            self.last = _conv(in_channel, out_channels[2], stride, kernel_size=1)

        self.add = P.TensorAdd()

    def construct(self, x):
        inputs = x
        x = self.sub_conv1(x)
        x = self.sub_conv2(x)
        x = self.sub_conv3(x)

        if self.last is not None:
            inputs = self.last(inputs)

        conv_proj = inputs
        return self.add(x, conv_proj)
        


class _inception(nn.Cell):
    def __init__(self, in_channel, num_outputs, stride, preact_bn=False):
        super(_inception, self).__init__()

        num_outputs = num_outputs.split()
        num_outputs = [s.split('-') for s in num_outputs]
        inception_out_channel = int(num_outputs[-1][0])
        num_outputs = num_outputs[:-1]
        pool_path_outputs = 0

        self.stride = stride

        if stride > 1:
            pool_path_outputs = num_outputs[-1][0]
            num_outputs = num_outputs[:-1]

        self.preact = preact_bn
        if preact_bn:
            self.pre_bn = nn.BatchNorm2d(in_channel)
            self.pre_scale = _scale(in_channel) 
            self.pre_relu = nn.ReLU()
        else:
            self.pre_bn = None
            self.pre_scale = None
            self.pre_relu = None

        self.ops = []
        
        #path1 
        self.conv1 = _conv_bn_relu(in_channel, num_outputs[0][0], stride=stride, kernel_size=1)

        #path2 
        self.conv2_1 = _conv_bn_relu(in_channel, num_outputs[1][0], stride=stride, kernel_size=1)
        self.conv2_2 = _conv_bn_relu(num_outputs[1][0], num_outputs[1][1], stride=1, kernel_size=3)

        #path3
        self.conv3_1 = _conv_bn_relu(in_channel, num_outputs[2][0], stride=stride, kernel_size=1)
        self.conv3_2 = _conv_bn_relu(num_outputs[2][0], num_outputs[2][1], stride=1, kernel_size=3)
        self.conv3_3 = _conv_bn_relu(num_outputs[2][1], num_outputs[2][2], stride=1, kernel_size=3)

        #pool_conv
        if stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
            self.conv_pool = _conv_bn_relu(in_channel, out_channel=pool_path_outputs, stride=1, kernel_size=1) 

        #concat
        self.concat = P.Concat(axis=1)
        _in_channel = int(num_outputs[0][-1]) + int(num_outputs[1][-1]) + int(num_outputs[2][-1]) + int(pool_path_outputs)
        self.conv_concat = _conv(_in_channel, inception_out_channel, stride=1, kernel_size=1)

        #res
        self.conv_for_last = in_channel != inception_out_channel
        if self.conv_for_last:
            self.conv_last = _conv(in_channel, inception_out_channel, stride=stride, kernel_size=1)
        else:
            self.conv_last = None

        self.add = P.TensorAdd()

        self.p = P.Print()

    def construct(self, x):
        #preact
        if self.preact:
            x = self.pre_bn(x)
            x = self.pre_scale(x)
            x = self.pre_relu(x)
        inputs = x

        #path1
        path1_out = self.conv1(inputs)

        #path2
        path2_out = self.conv2_1(inputs)
        path2_out = self.conv2_2(path2_out)

        #path3
        path3_out = self.conv3_1(inputs)
        path3_out = self.conv3_2(path3_out)
        path3_out = self.conv3_3(path3_out)

        #pool_conv
        pool = None
        if self.stride > 1:
            pool = self.pool(inputs)
            pool = self.conv_pool(pool)

        #concat
        if self.stride > 1:
            x = self.concat((path1_out, path2_out, path3_out, pool))
        else:
            x = self.concat((path1_out, path2_out, path3_out))

        x = self.conv_concat(x)

        #res
        if self.conv_for_last:
            inputs = self.conv_last(inputs)
        x = self.add(x, inputs)
        
        return x

        




        




            

        


'''
if __name__ == "__main__":
    import numpy as np
    import mindspore
    import mindspore.context as context
    context.set_context(
        mode=context.GRAPH_MODE,
        device_target="Ascend",
        device_id=3
    )
    
    def get_tensor(shape):
        np_array = np.random.randn(*shape)
        return mindspore.Tensor(np_array, mindspore.float32)

    inputs = get_tensor((64, 128, 256, 256))
    conv4_1 = _inception(128, '64 48-128 24-48-48 128 256', 2, True)
    conv4_1(inputs)
'''

