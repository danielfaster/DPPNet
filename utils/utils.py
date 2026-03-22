import torch
import  os
import logging
import torch.nn.functional as F
def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay*init_lr
        lr=param_group['lr']
    return lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 

def init_logger(version):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(version + '.log')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    return logger

def save_edge(save_path, filename, predict_edge, transform):

    #  prepost edge
    output_edge = F.sigmoid(predict_edge)
    output_edge = output_edge.data.cpu().squeeze(0)
    output_edge = transform(output_edge)

    save_test_edge_path = save_path + '/RGBD_VST/edge'
    os.makedirs(save_test_edge_path, exist_ok=True)
    output_edge.save(os.path.join(save_test_edge_path, filename + '.png'))

def save_depth_quality(save_path, filename, predict_depth_quality, transform):


    output_depth_quality = F.sigmoid(predict_depth_quality)
    output_depth_quality = output_depth_quality.data.cpu().squeeze(0)
    output_depth_quality = transform(output_depth_quality)

    save_ssim_path = save_path + '/RGBD_VST/ssim2'
    os.makedirs(save_ssim_path, exist_ok=True)
    output_depth_quality.save(os.path.join(save_ssim_path, filename + '.png'))

def save_RGB_branch(save_path, filename, predict_rgb, transform):


    predict_rgb = F.sigmoid(predict_rgb)
    predict_rgb = predict_rgb.data.cpu().squeeze(0)
    predict_rgb = transform(predict_rgb)

    save_rgb_branch_path = save_path + '/RGBD_VST/rgb'
    os.makedirs(save_rgb_branch_path, exist_ok=True)
    predict_rgb.save(os.path.join(save_rgb_branch_path, filename + '.png'))

def save_Depth_branch(save_path, filename, predict_depth, transform):


    predict_depth = F.sigmoid(predict_depth)
    predict_depth = predict_depth.data.cpu().squeeze(0)
    predict_depth = transform(predict_depth)

    save_depth_branch_path = save_path + '/RGBD_VST/depth'
    os.makedirs(save_depth_branch_path, exist_ok=True)
    predict_depth.save(os.path.join(save_depth_branch_path, filename + '.png'))



def get_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return  param_group['lr']

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = decay * init_lr
        lr = param_group['lr']
    return lr

def structure_loss(pred, mask):
    weit  = 1+5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15)-mask)
    wbce  = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce  = (weit*wbce).sum(dim=(2,3))/weit.sum(dim=(2,3))

    pred  = torch.sigmoid(pred)
    inter = ((pred*mask)*weit).sum(dim=(2,3))
    union = ((pred+mask)*weit).sum(dim=(2,3))
    wiou  = 1-(inter+1)/(union-inter+1)
    return (wbce+wiou).mean()

def save_model(args, logging, model, epoch, train_loss,best_train_loss,best_epoch):

    print(f"old best_train_loss: {best_train_loss:.6f}, new train_loss: {train_loss:.6f}")

    if train_loss < best_train_loss:
        print("[Congratulation! New best model found!]")
        print(f"save best model with train_loss: {train_loss:.6f}, epoch: {epoch}")

        model_state_dict_cpu = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }

        save_path = os.path.join(args.save_model_dir, "RGBD_VST.pth")
        torch.save(model_state_dict_cpu, save_path)

        best_train_loss = train_loss
        best_epoch = epoch

        save_message = (
            f"[Best Updated]\n"
            f"epoch: {epoch}, train_loss: {train_loss:.6f}\n"
            f"saved to: {save_path}"
        )
    else:
        print("[Not best model, skip saving]")
        print(
            f"current epoch: {epoch}, train_loss: {train_loss:.6f}\n"
            f"best epoch: {best_epoch}, best_train_loss: {best_train_loss:.6f}"
        )

        save_message = (
            f"[Skip]\n"
            f"epoch: {epoch}, train_loss: {train_loss:.6f}\n"
            f"best epoch: {best_epoch}, best_train_loss: {best_train_loss:.6f}"
        )

    logging.info(save_message)
    return save_message


# The edge code refers to 'Non-Local Deep Features for Salient Object Detection', CVPR 2017.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

fx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)
fy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).astype(np.float32)
fx = np.reshape(fx, (1, 1, 3, 3))
fy = np.reshape(fy, (1, 1, 3, 3))
fx = Variable(torch.from_numpy(fx)).cuda()
fy = Variable(torch.from_numpy(fy)).cuda()
contour_th = 1.5



def label_edge_prediction(label):
    # convert label to edge
    label = label.gt(0.5).float()
    label = F.pad(label, (1, 1, 1, 1), mode='replicate')
    label_fx = F.conv2d(label, fx)
    label_fy = F.conv2d(label, fy)
    label_grad = torch.sqrt(torch.mul(label_fx, label_fx) + torch.mul(label_fy, label_fy))
    label_grad = torch.gt(label_grad, contour_th).float()

    return label_grad