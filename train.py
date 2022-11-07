import time
import pandas as pd
from Mynet import mynet
import os
import argparse
import numpy as np
from tqdm import tqdm
import joblib
import glob
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from my_data_driven import GetDataset
import MS_SSIM
import MSSIM_L1_loss
def parse_args():  # 参数解析器
    parser = argparse.ArgumentParser()
    # 增加属性
    parser.add_argument('--name', default='Mynet', help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=20, type=int)  # 原来是100
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float)  # 5e-4
    parser.add_argument('--weight', default=[0.16, 0.84], type=float)
    parser.add_argument('--gamma', default=0.9, type=float)  # ExponentialLR gamma
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--eps', default=1e-8, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)  # 5e-4
    parser.add_argument('--training_dir',default="VOC2012_240/train/sourceA/",type=str)
    parser.add_argument('--test_dir', default="VOC2012_240/test/sourceA/", type=str)

    args = parser.parse_args()  # 属性给与args实例：add_argument 返回到 args 子类实例
    return args

def main():
    args = parse_args()
    if not os.path.exists('models/%s' % args.name):
        os.makedirs('models/%s' % args.name)  # 创建文件夹保存模型
    print('Config -----')
    for arg in vars(args):
        print('%s: %s' % (arg, getattr(args, arg)))  # 打印参数配置
    print('------------')
    with open('models/%s/args.txt' % args.name, 'w') as f:  # 写入参数文件
        for arg in vars(args):
            print('%s: %s' % (arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' % args.name)
    cudnn.benchmark = True

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device=torch.device('cuda:0')
    use_gpu=torch.cuda.is_available()
    if use_gpu:
        print('GPU Mode Acitavted')
    else:
        print('CPU Mode Acitavted')

    # 定义文件dataset
    folder_dataset_train = glob.glob(args.training_dir + "*.jpg")
    folder_dataset_test = glob.glob(args.test_dir + "*.jpg")
    #定义预处理方式
    transform_train = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.485, 0.456, 0.406),
                                                               (0.229, 0.224, 0.225))
                                          ])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))
                                         ])

    #定义数据集
    dataset_train = GetDataset(imageFolderDataset=folder_dataset_train,
                               transform=transform_train)
    dataset_test = GetDataset(imageFolderDataset=folder_dataset_test,
                              transform=transform_test)

    #数据集loader
    train_loader = DataLoader(dataset_train,
                              shuffle=True,
                              batch_size=args.batch_size)

    test_loader = DataLoader(dataset_test,
                             shuffle=True,
                             batch_size=args.batch_size)
    net = mynet()
    if use_gpu:
        net=net.cuda()
        net.cuda()
    else:
        net=net
    # critertion1=nn.L1Loss()
    # critertion2=MS_SSIM.MS_SSIM()
    critertion3=MSSIM_L1_loss.MS_SSIM_L1_LOSS()
    #Adam
    optimizer = optim.Adam(net.parameters(), lr=args.lr,
                           betas=args.betas, eps=args.eps, weight_decay=args.weight_decay)  # Adam法优化,filter是为了固定部分参数
    scheduler=lr_scheduler.ExponentialLR(optimizer,gamma=args.gamma)

    running_train_loss = 0
    running_val_loss = 0
    log = pd.DataFrame(index=[],
                       columns=['epoch',
                                'lr',
                                'train_loss',
                                'val_loss'])
    for epoch in range(args.epochs):
        #训练
        net.train()
        t1=time.time()
        for i,(img0,img1,label) in tqdm(enumerate(train_loader),total=len(train_loader)):
            if use_gpu:
                img0=img0.cuda()
                img1=img1.cuda()
                label=label.cuda()
            else:
                img0 = img0
                img1 = img1
                label = label
            out=net(img0,img1)
            # L1_loss=critertion1(out,label)
            # MS_SSIM_loss=critertion2(out,label)
            # loss=args.weight[0]*L1_loss + args.weight[1]*MS_SSIM_loss
            loss=critertion3(out,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_train_loss+=loss.item()
            print("[epoch: %3d/%3d, progress: %5d/%5d] train loss: %8f " % (epoch + 1, args.epochs, (i + 1) * args.batch_size, len(dataset_train), loss.item()))
        print('finish train epoch: [{}/{}] costs:{}  avg_loss:{}'.format(epoch + 1 ,args.epochs,(time.time() - t1),(running_train_loss/len(train_loader))))
        scheduler.step()
        if (epoch + 1) % 2 == 0:#每10个epoch记录一次
            torch.save(net.state_dict(), 'models/{}/model_{}.pth'.format(args.name,(epoch + 1)))
        train_log = OrderedDict([('train_loss', running_train_loss/len(train_loader))])
        running_train_loss = 0
        #验证
        net.eval()
        with torch.no_grad():
            t1 = time.time()
            for i, (img0, img1, label) in tqdm(enumerate(test_loader),total=len(test_loader)):
                if use_gpu:
                    img0 = img0.cuda()
                    img1 = img1.cuda()
                    label = label.cuda()
                else:
                    img0 = img0
                    img1 = img1
                    label = label
                out = net(img0, img1)
                # L1_loss = critertion1(out, label)
                # MS_SSIM_loss = critertion2(out, label)
                # loss = args.weight[0] * L1_loss + args.weight[1] * MS_SSIM_loss
                loss=critertion3(out,label)
                running_val_loss += loss.item()
                print("[epoch: %3d/%3d, batch: %5d/%5d] test loss: %8f " % (epoch + 1, args.epochs, (i + 1) * args.batch_size, len(dataset_test), loss.item()))
            val_log = OrderedDict([('val_loss', running_val_loss / len(test_loader))])
            print('finish val epoch: [{}/{}] costs:{}  avg_loss:{}'.format((epoch + 1),args.epochs,(time.time() - t1),(running_val_loss/len(test_loader))))
            running_train_loss = 0
        tmp = pd.Series([
            epoch + 1,
            scheduler.get_last_lr(),
            train_log['train_loss'],
            val_log['val_loss'],
        ], index=['epoch', 'lr', 'train_loss', 'val_loss'])  # Series创建字典
        log = pd.concat([log, tmp], ignore_index=True)  # 新写的
        log.to_csv('models/%s/log.csv' % args.name, index=False)  # log:训练的日志记录

if __name__ == '__main__':
    main()
