import time
import math
import re
import sys
import os
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.backends import cudnn
from torch.utils import model_zoo
import torchvision
from torchvision import transforms
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
sys.path.append('../')
from sixdrepnet.model import SixDRepNet, SixDRepNet2, SixDofNet
from sixdrepnet import datasets
from sixdrepnet.loss import GeodesicLoss
from utils import compute_rotation_matrix_from_ortho6d, AverageMeter, calculate_error
import wandb


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Head pose estimation using the 6DRepNet.')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument('--img-size', dest='img_sz', default=224, type=int,help="Input image size (default : 224)")
    parser.add_argument('--model', default='RepVGG-B1g2', help="Using DoF model (default : 'RepVGG-B1g2' )")
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=80, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=80, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument('--scheduler', default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset type.',
        default='Pose_300W_LP', type=str) #Pose_300W_LP
    parser.add_argument(
        '--data-dir', dest='data_dir', help='Directory path for data.',
        default='datasets/300W_LP', type=str)#BIWI_70_30_train.npz
    parser.add_argument(
        '--filename_list', dest='filename_list',
        help='Path to text file containing relative paths for every example.',
        default='datasets/300W_LP/files.txt', type=str) #BIWI_70_30_train.npz #300W_LP/files.txt
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='', type=str)
    parser.add_argument(
        '--yolo-norm', action='store_true', help="use yolo norm (img = img / 255) (default=imagenet normalize)"
    )
    parser.add_argument('--wandb', default=False, help="logging wandb")

    args = parser.parse_args()
    return args

def load_filtered_state_dict(model, snapshot):
    # By user apaszke from discuss.pytorch.org
    model_dict = model.state_dict()
    snapshot = {k: v for k, v in snapshot.items() if k in model_dict}
    model_dict.update(snapshot)
    model.load_state_dict(model_dict)


if __name__ == '__main__':

    args = parse_args()
    cudnn.enabled = True
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    img_sz = args.img_sz
    gpu = args.gpu_id
    b_scheduler = args.scheduler
    model_name = args.model

    use_wandb = args.wandb
    

    if not os.path.exists('output/snapshots'):
        os.makedirs('output/snapshots')

    summary_name = '{}_{}_bs{}_{}'.format(
        model_name, int(time.time()), args.batch_size, args.img_sz)

    if not os.path.exists('output/snapshots/{}'.format(summary_name)):
        os.makedirs('output/snapshots/{}'.format(summary_name))

    if 'RepVGG' in model_name: 
        print("Using RepVGG model")
        pretrained_backbone_name= f'{model_name}-train.pth'
        if os.path.exists(pretrained_backbone_name):
            print("pretrained backbone file is exists :", pretrained_backbone_name)
            model = SixDRepNet(backbone_name=model_name,
                                backbone_file=pretrained_backbone_name,
                                deploy=False,
                                pretrained=True)
        else:
            print('pretrained backbone cant find, so fully trained')
            model = SixDRepNet(backbone_name=model_name,
                                backbone_file='',
                                deploy=False,
                                pretrained=False)
    else: # using normal neural network model
        print("Using normal model :", model_name)
        model = SixDofNet(model_name=model_name, pretrained=True)
    if not args.snapshot == '':
        saved_state_dict = torch.load(args.snapshot)
        model.load_state_dict(saved_state_dict['model_state_dict'])

    print('Loading data.')


    if args.yolo_norm:
        print("use yolo normalize")
        normalize = None
        transformations = transforms.Compose([transforms.RandomResizedCrop(size=(args.img_sz, args.img_sz),scale=(0.8,1)),
                                          transforms.ToTensor()])
    else:
        print("use imagenet normalize")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
        transformations = transforms.Compose([transforms.RandomResizedCrop(size=(args.img_sz, args.img_sz), scale=(0.8,1)),
                                            transforms.ToTensor(),
                                            normalize])

    pose_dataset = datasets.getDataset(
        args.dataset, args.data_dir, args.filename_list, transformations)
    
    train_loader = torch.utils.data.DataLoader(
        dataset=pose_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    model.cuda(gpu)
    crit = GeodesicLoss().cuda(gpu) #torch.nn.MSELoss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)


    if not args.snapshot == '':
        optimizer.load_state_dict(saved_state_dict['optimizer_state_dict'])

    if use_wandb:
        wandb.init(config=args, project='SixDof Training')

    #milestones = np.arange(num_epochs)
    milestones = [10, 20]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.5)

    print('Starting training.')
    for epoch in range(num_epochs):
        loss_sum = AverageMeter()
        pitch_error = yaw_error = roll_error = mae_error = AverageMeter()
        iter = 0
        for i, (images, gt_mat, cont_labels, _) in enumerate(train_loader):
            iter += 1
            images = torch.Tensor(images).cuda(gpu)

            # Forward pass
            # To change the model to tensorrt, the internal utils.compute_rotation_matrix_from_ortho6d(poses) was erased.
            out = model(images)
            pred_mat = compute_rotation_matrix_from_ortho6d(out)

            # Calc loss
            loss = crit(gt_mat.cuda(gpu), pred_mat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                pitch, yaw, roll = calculate_error(pred_mat, gt_mat, cont_labels)            

            loss_sum.update(loss.item())
            pitch_error.update(pitch)
            yaw_error.update(yaw)
            roll_error.update(roll)
            mae_error.update((pitch+yaw+roll)/3)

            if (i+1) % 100 == 0 or i == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.6f P/Y/R [%.6f / %.6f / %.6f]'  % (
                          epoch+1,
                          num_epochs,
                          i+1,
                          len(pose_dataset)//batch_size,
                          loss.item(),
                          pitch,
                          yaw,
                          roll
                        )
                      )
        if use_wandb: 
            wandb.log({'loss':loss_sum.avg, 'pitch':pitch_error.avg, 'yaw':yaw_error.avg, 'roll':roll_error.avg, 'MAE':mae_error.avg}, step=epoch)

        if b_scheduler:
            scheduler.step()

        # Save models at numbered epochs.
        if epoch % 10 == 0 and epoch < num_epochs:
            print(f'epoch {epoch} save',
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + args.output_string +
                      '_epoch_' + str(epoch) + '.tar')
                  )
            # removed pretrained
        print('Taking snapshot...', epoch,
                  torch.save({
                      'epoch': epoch,
                      'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict(),
                  }, 'output/snapshots/' + summary_name + '/' + 'last_checkpoint.tar')
                  )