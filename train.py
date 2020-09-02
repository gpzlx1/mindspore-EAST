import os
import EAST
import EAST_VGG
import loss as my_loss
#import moxing
import dataset as my_dataset
import datasetV2
import datasetV3
import mindspore
from mindspore import context, Tensor, nn
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor, TimeMonitor
from mindspore.train import Model
from mindspore.nn.dynamic_lr import piecewise_constant_lr
from mindspore.train.loss_scale_manager import FixedLossScaleManager, DynamicLossScaleManager


def train():
    context.set_context(
        mode=context.GRAPH_MODE, 
        device_target="Ascend", 
        #save_graphs=True,
        #save_graphs_path="/home/work/user-job-dir/EAST/",
        #enable_reduce_precision=False,
        #device_id=5
    )

    epoch = 600

    my_dataset.download_dataset()
    train_img_path = os.path.abspath('/cache/train_img')
    train_gt_path  = os.path.abspath('/cache/train_gt')
    #my_dataset.data_to_mindrecord_byte_image(train_img_path, train_gt_path, mindrecord_dir='/cache', prefix='icdar_train.mindrecord',file_num=1)
    #dataset = my_dataset.create_icdar_train_dataset(mindrecord_file=['icdar_train.mindrecord0','icdar_train.mindrecord1','icdar_train.mindrecord2','icdar_train.mindrecord3'], batch_size=32, repeat_num=epoch, 
    #                            is_training=True, num_parallel_workers=8, length=512, scale=0.25)
    #dataset = my_dataset.create_icdar_train_dataset(mindrecord_file='/cache/icdar_train.mindrecord', batch_size=32, repeat_num=epoch, 
    #                            is_training=True, num_parallel_workers=24, length=512, scale=0.25)
    #dataset = my_dataset.create_demo_dataset(batch_size=21, repeat_num=2)
    #train_img_path = os.path.abspath('/home/licheng/gpzlx1/ICDAR_2015/train/img')
    #train_gt_path  = os.path.abspath('/home/licheng/gpzlx1/ICDAR_2015/train/gt')
    dataset = datasetV2.create_icdar_train_dataset(train_img_path, train_gt_path, batch_size=14, repeat_num=1, is_training=True, num_parallel_workers=24)
    #dataset = datasetV3.create_icdar_train_dataset(train_img_path, train_gt_path, batch_size=14, repeat_num=1, is_training=True, num_parallel_workers=24)
    dataset_size = dataset.get_dataset_size()

    print("Create dataset done!, dataset_size: ", dataset_size)


    
    #east = EAST.EAST()
    net = EAST_VGG.EAST()
    
    #ckpt_config = CheckpointConfig(save_checkpoint_steps=dataset_size * 20)
    #ckpoint_cb = ModelCheckpoint(prefix='EAST', directory='/cache', config=ckpt_config)


    milestone = [100, 300]
    learning_rates = [1e-3, 1e-4]
    lr = piecewise_constant_lr(milestone, learning_rates)
    opt = nn.Adam(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate=lr)
    net = my_loss.EASTWithLossCell(net)
    net = my_loss.TrainingWrapper(net, opt)
    net.set_train(True)

    callback = [TimeMonitor(data_size=dataset_size), LossMonitor()]#, ckpoint_cb]


    model = Model(net)
    dataset_sink_mode = False
    print("start trainig")
    model.train(epoch, dataset, callbacks=callback, dataset_sink_mode=dataset_sink_mode)


    
if __name__ == "__main__":
    train()