import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import torch.distributed as dist
import math
import os
import torch.nn.functional as F
from utils.utils import label_edge_prediction,get_learning_rate,adjust_lr,structure_loss,save_model, clip_gradient, adjust_lr,init_logger
from utils.data_v1 import get_loader
from utils.options import config
from utils.time_util_v2 import time_recoder
from Models.DPPNet import DPPNet


def train(net, logging, args):

    # set the device for training
    if args.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')

    cudnn.benchmark = True
    net.cuda()
    params = net.parameters()
    optimizer = torch.optim.Adam(params, args.lr)

    print('load data...')
    train_loader = get_loader(args.rgb_label_root, args.gt_label_root, args.depth_label_root, args.depth_quality_label_root,batchsize=args.batch_size,
                              trainsize=args.img_size)

    logging.info('''
              Starting training:
                  Train epoch: {}
                  Batch size: {}
                  Learning rate: {}
                  Training size: {}
              '''.format(args.epochs, args.batch_size, args.lr, len(train_loader.dataset)))

    os.makedirs(args.save_model_dir, exist_ok=True)
    global best_fm, best_epoch,  best_train_loss
    best_train_loss = 10.0
    best_epoch = -1

    for epoch in range(args.epochs):
         train_vali(epoch,args.epochs,args, logging, net, optimizer, train_loader)


@time_recoder
def train_vali(now_epoch, total_epoch, args, logging, net, optimizer, train_loader):

    total_step = len(train_loader)
    epoch_step =0
    loss_all =0
    adjust_lr(optimizer, args.lr, now_epoch, decay_rate=0.1, decay_epoch=args.decay_epoch)
    logging.info('Starting epoch {}/{}.'.format(now_epoch, total_epoch))
    logging.info('epoch:{0}-------lr:{1}'.format(now_epoch, get_learning_rate(optimizer)))
    net.train()
    for i, (images, gts, depths, depth_qualitys) in enumerate(train_loader, start=1):

        optimizer.zero_grad()
        images = images.cuda()
        gts = gts.cuda()
        depths = depths.cuda()
        depth_qualitys = depth_qualitys.cuda()
        gt_edges = label_edge_prediction(gts).cuda()
        pre_res = net(images, depths)

        s_loss = structure_loss(pre_res[0][-1], gts)
        edge_loss = structure_loss(pre_res[1][-1], gt_edges)
        depth_quality_loss = structure_loss(pre_res[2][-1], depth_qualitys)
        depth_loss = structure_loss(pre_res[3], gts)
        rgb_loss = structure_loss(pre_res[4], gts)

        total_loss = s_loss + edge_loss + depth_quality_loss + depth_loss + rgb_loss

        total_loss.backward()

        clip_gradient(optimizer, args.clip)
        optimizer.step()
        epoch_step += 1
        loss_all += total_loss.data

        if i % 50 == 0 or i == total_step or i == 1:
            logging.info(
                '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], saliency Loss: {:.4f} rgb Loss: {:0.4f} edge Loss: {:0.4f} depth Loss: {:0.4f} depth quality Loss: {:0.4f}'.
                format(now_epoch, total_epoch, i, total_step, s_loss.data, rgb_loss.data, edge_loss.data, depth_quality_loss.data,
                       depth_loss.data))


    loss_all /= epoch_step
    message = save_model(args, logging, net, now_epoch, loss_all,best_train_loss,best_epoch)

    logging.info('#TRAIN#:Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(now_epoch, total_epoch, loss_all))
    return message


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    num_gpus = torch.cuda.device_count()
    # load config file
    args = config(num_gpus=num_gpus)
    net = DPPNet(args)
    logging = init_logger("v0")
    train(net=net,logging=logging, args=args)



