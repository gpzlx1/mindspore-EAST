import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P 
import numpy as np
import pvanet
import ops
import math

def unpool(size):
    return P.ResizeBilinear(size)


class EAST(nn.Cell):
    def __init__(self):
        super(EAST, self).__init__()

        #param
        self.TEXT_SCALE = 512
        self.pi = math.pi / 1.
        self.pi = mindspore.Tensor([self.pi], mindspore.float32)

        #network
        self.pvanet = pvanet.PAVNet(True)
        
        #for i = 0
        self.unpool0 = unpool((32, 32))

        #for i = 1
        self.concat1 = P.Concat(axis=1)
        self.conv1_1 = ops.conv_bn_relu(1280, 128, stride=1, kernel_size=1, padding='valid')
        self.conv1_2 = ops.conv_bn_relu(128, 128, stride=1, kernel_size=3, padding='pad', padding_number=1)
        self.unpool1 = unpool((64, 64))

        #for i = 2
        self.concat2 = P.Concat(axis=1)
        self.conv2_1 = ops.conv_bn_relu(384, 64, stride=1, kernel_size=1, padding='valid')
        self.conv2_2 = ops.conv_bn_relu(64, 64, stride=1, kernel_size=3, padding='pad', padding_number=1)
        self.unpool2 = unpool((128, 128))


        #for i = 3
        self.concat3 = P.Concat(axis=1)
        self.conv3_1 = ops.conv_bn_relu(192, 32, stride=1, kernel_size=1, padding='valid')
        self.conv3_2 = ops.conv_bn_relu(32, 32, stride=1, kernel_size=3, padding='pad', padding_number=1)
        self.conv3_3 = ops.conv_bn_relu(32, 32, stride=1, kernel_size=3, padding='pad', padding_number=1)


        #output
        ## F_score
        self.conv_for_fscore = ops._conv(32, 1, stride=1, kernel_size=1, padding='valid')
        self.sigmoid_for_fscore = P.Sigmoid()

        ## geo_map
        self.conv_for_geo_map = ops._conv(32, 4, stride=1, kernel_size=1, padding='valid')
        self.sigmoid_for_geo_map = P.Sigmoid()

        ## angle_map
        self.conv_for_angle_map = ops._conv(32, 1, stride=1, kernel_size=1, padding='valid')
        self.sigmoid_for_angle_map = P.Sigmoid()

        ## F_geometry 
        self.concat_for_F_geometry  = P.Concat(axis=1)


        ## other
        self.mul = P.Mul()
        self.add = P.TensorAdd()




    def construct(self, x):
        #i = 0
        _, f = self.pvanet(x)
        
        x = self.unpool0(f[0])

        #i = 1
        x = self.concat1((x, f[1]))
        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.unpool1(x)

        #i = 2
        x = self.concat2((x, f[2]))
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.unpool2(x)

        #i = 3
        x = self.concat3((x, f[3]))
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        g_3 = self.conv3_3(x)

        #F_score 
        F_score = self.conv_for_fscore(g_3)
        F_score = self.sigmoid_for_fscore(F_score)

        #geo_map
        geo_map = self.sigmoid_for_geo_map(self.conv_for_geo_map(g_3))
        geo_map = self.mul(geo_map, self.TEXT_SCALE)

        #angle_map
        angle_map = self.sigmoid_for_angle_map(self.conv_for_angle_map(g_3)) - 0.5
        angle_map = self.mul(angle_map, self.pi)


        # F_geometry 
        F_geometry = self.concat_for_F_geometry((geo_map, angle_map))

        return F_score, F_geometry


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
    network = EAST()
    results = network(inputs)
    for i in results:
        print(i)

