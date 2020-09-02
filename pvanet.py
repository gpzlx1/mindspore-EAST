import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
import numpy as np
import ops




class PAVNet(nn.Cell):
    def __init__(self, include_last_bn_relu=True):
        super(PAVNet, self).__init__()
        self.conv1_1 = ops._conv_bn_crelu(3, 32, 2, 7)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        self.conv2_1 = ops._mCReLU(64, [48, 48, 128], 1, False)
        self.conv2_2 = ops._mCReLU(128, [48, 48, 128], 1, True)
        self.conv2_3 = ops._mCReLU(128, [48, 48, 128], 1, True)


        self.conv3_1 = ops._mCReLU(128, [96, 96, 256], 2, True)
        self.conv3_2 = ops._mCReLU(256, [96, 96, 256], 1, True)
        self.conv3_3 = ops._mCReLU(256, [96, 96, 256], 1, True)
        self.conv3_4 = ops._mCReLU(256, [96, 96, 256], 1, True)

        
        self.conv4_1 = ops._inception(256, '128 96-256 48-96-96 256 512', 2, True)
        self.conv4_2 = ops._inception(512, '128 96-256 48-96-96 512', 1, True)
        self.conv4_3 = ops._inception(512, '128 96-256 48-96-96 512', 1, True)
        self.conv4_4 = ops._inception(512, '128 96-256 48-96-96 512', 1, True)

        
        self.conv5_1 = ops._inception(512, '128 192-384 64-128-128 256 768', 2, True)
        self.conv5_2 = ops._inception(768, '128 192-384 64-128-128 768', 1, True)
        self.conv5_3 = ops._inception(768, '128 192-384 64-128-128 768', 1, True)
        self.conv5_4 = ops._inception(768, '128 192-384 64-128-128 768', 1, True)

        self.include_last_bn_relu = include_last_bn_relu
        if include_last_bn_relu:
            self.last_bn = nn.BatchNorm2d(num_features=768)
            self.last_scale = ops._scale(768)
            self.last_relu = nn.ReLU()

        #self.p = P.Print()
        


    def construct(self, x):
        x = self.conv1_1(x)
        x = self.pool(x)


        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x2 = x #(B, 128, 128, 128)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x3 = x #(B, 256, 64, 64)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)

        x4 = x #(B, 512, 32, 32)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)

        

        if self.include_last_bn_relu:
            x = self.last_bn(x)
            x = self.last_scale(x)
            x = self.last_relu(x)

        x5 = x #(B, 768, 16, 16)
            
        return x, [x5, x4, x3, x2]  # pool2, pool3, pool4, pool5

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
    network = PAVNet()
    results = network(inputs)
    for i in results:
        print(i)


