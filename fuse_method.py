import torch
import argparse
from torchvision import transforms
import time
import os
from PIL import Image
from Mynet import mynet
import cv2
import numpy as np
def fusion(args,intype='jpg',outtype='jpg'):
    couples=int(len(os.listdir(args.fuse_Data_dir))/2)
    net = mynet()
    if use_gpu:
        net = net.cuda()
        net.cuda()
        net.load_state_dict(torch.load(args.dict_path))
    else:
        net = net
        net.load_state_dict(torch.load(args.dict_path,map_location=torch.device('cpu')))

    net.eval()
    with torch.no_grad():
        for num in range(1, couples + 1):
            t1 = time.time()
            path0 = args.fuse_Data_dir+ '/lytro-{}{}-A.'.format(num // 10, num % 10) + intype  # for the "Lytro" dataset
            path1 = args.fuse_Data_dir+ '/lytro-{}{}-B.'.format(num // 10, num % 10) + intype  # for the "Lytro" dataset
            img0 = cv2.imread(path0)
            img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
            # img0 = cv2.resize(img0, (800, 600), interpolation=cv2.INTER_AREA)
            img1 = cv2.imread(path1)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # img1 = cv2.resize(img1, (800, 600), interpolation=cv2.INTER_AREA)
            transform_fuse = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.485, 0.456, 0.406),
                                                                      (0.229, 0.224, 0.225))
                                                 ])
            img0 = transform_fuse(img0)
            img1 = transform_fuse(img1)
            img0.unsqueeze_(0)
            img1.unsqueeze_(0)
            if use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()
            else:
                img0 = img0
                img1 = img1
            out=net(img0,img1)
            out=tensor2np2color(out)
            cv2.imwrite("result/{}/color_lytro_{}.{}".format(args.name,str(num).zfill(2),str(outtype)),out)
            t2=time.time()
            print(t2-t1)
            print('Finish no.{} fusion'.format(num))
def tensor2np2color(intensor):
    # trans = transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
    # intensor=trans(intensor)
    out = intensor[0].cpu().detach().numpy().transpose(1, 2, 0)  # + 1) / 2)
    out = (out - np.min(out)) / (np.max(out) - np.min(out)) * 255.0  # 将图像数据扩展到[0,255]
    out = np.array(out, dtype='uint8')  # 改为Unit8
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Mynet', help='model name: (default: arch+timestamp)')
    # parser.add_argument('--fuse_Data_dir', default="DataSet/lytro",type=str)
    parser.add_argument('--fuse_Data_dir', default="DataSet/lytro", type=str)
    parser.add_argument('--dict_path',default='models/Mynet/model_20.pth',type=str)#模型选择改这里
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU Mode Acitavted')
    else:
        print('CPU Mode Acitavted')

    if not os.path.exists('result/%s'% args.name):
        os.makedirs('result/%s' % args.name)  # 创建文件夹保存模型
    fusion(args,intype='jpg',outtype='jpg')

