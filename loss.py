import mindspore 
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from mindspore.ops import functional as F
import numpy as np
from mindspore import Tensor, context
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from mindspore.common import dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.train.parallel_utils import ParallelMode
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode, _get_mirror_mean



class LossFunc(nn.Cell):
    def __init__(self, weight_angle=10):
        super(LossFunc, self).__init__()
        self.split = P.Split(1, 5)
        self.min = P.Minimum()
        self.log = P.Log()
        self.cos = P.Cos()
        self.mean = P.ReduceMean()
        #self.flatten = P.Flatten()
        self.sum = P.ReduceSum()
        self.weight_angle = weight_angle
        self.max = P.Maximum()
        self.print = P.Print()
        
    def get_dice_loss(self, gt_score, pred_score):
        inter = self.sum(gt_score * pred_score)
        union = self.sum(gt_score) + self.sum(pred_score) + 1e-5
        return 1. - (2 * inter / union)

    
    def get_geo_loss(self, gt_geo, pred_geo):
        d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = self.split(gt_geo)
        d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = self.split(pred_geo)
        area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
        area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)
        w_union = self.min(d3_gt, d3_pred) + self.min(d4_gt, d4_pred)
        h_union = self.min(d1_gt, d1_pred) + self.min(d2_gt, d2_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        iou_loss_map = -self.log((area_intersect + 1.0)/(area_union + 1.0))
        angle_loss_map = 1 - self.cos(angle_pred - angle_gt)
        return iou_loss_map, angle_loss_map 

    def construct(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map):

        #flag = self.max(y_true_cls)
        gt_score_sum = self.sum(gt_score)

        #@if y_true_cls_sum < 1:
        #    total_loss = self.sum(y_pred_cls + y_pred_geo) * 0

        #part 1
        classify_loss = self.get_dice_loss(gt_score, pred_score * (1 - ignored_map))
        iou_loss_map, angle_loss_map = self.get_geo_loss(gt_geo, pred_geo)
        
        angle_loss = self.sum(angle_loss_map * gt_score) / (self.sum(gt_score) + 1e-5)
        iou_loss = self.sum(iou_loss_map * gt_score) / (self.sum(gt_score) + 1e-5)
        geo_loss = self.weight_angle * angle_loss + iou_loss
        total_loss = geo_loss + classify_loss

        #self.print(self.sum(gt_score))
        #self.print(angle_loss)
        #self.print(iou_loss)

        return total_loss #* flag




class EASTWithLossCell(nn.Cell):
    def __init__(self, network):
        super(EASTWithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = LossFunc()

    def construct(self, x, gt_score, gt_geo, ignored_map):
        pred_score, pred_geo  = self.network(x)
        model_loss = self.loss(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
        total_loss = model_loss

        return model_loss



class TrainingWrapper(nn.Cell):
    
    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.add_flags(defer_inline=True)
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation('grad', get_by_list=True, sens_param=True)
        self.sens = sens
        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = _get_parallel_mode()
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_mirror_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.print = P.Print()

    def construct(self, x, gt_score, gt_geo, ignored_map):
        weights = self.weights
        loss = self.network(x, gt_score, gt_geo, ignored_map)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(x, gt_score, gt_geo, ignored_map, sens)
        self.print(grads[0])
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(loss, self.optimizer(grads))


