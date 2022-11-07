import torch
import argparse
from torchvision import transforms
import time
import numpy as np
import os
from Mynet import mynet
import glob
import random
import cv2
def fusion(args,path0,path1,width=800,height=600):
    net = mynet()
    if use_gpu:
        net = net.cuda()
        net.cuda()
        net.load_state_dict(torch.load(args.dict_path))
    else:
        net = net
        net.load_state_dict(torch.load(args.dict_path, map_location=torch.device('cpu')))
    net.eval()
    if isinstance(path0,str):
        img0 = cv2.imread(path0)
        img0 =cv2.resize(img0,(width,height),interpolation=cv2.INTER_AREA)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    else :
        img0=path0
        img0=np.array(img0)
        img0 =cv2.resize(img0, (width,height), interpolation=cv2.INTER_AREA)
        img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    if isinstance(path1,str):
        img1 = cv2.imread(path1)
        img1 =cv2.resize(img1,(width,height), interpolation=cv2.INTER_AREA)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    else :
        img1 = path1
        img1 = np.array(img1)
        img1 =cv2.resize(img1,(width,height), interpolation=cv2.INTER_AREA)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    transform_fuse = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406),
                                                              (0.229, 0.224, 0.225))])
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
    out = tensor2np2color(out)
    return out
def tensor2np2color(tensor):
    out = tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    out = (out - np.min(out)) / (np.max(out) - np.min(out)) * 255.0  # 将图像数据扩展到[0,255]
    out = np.array(out, dtype='uint8')  # 改为Unit8
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    return out
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='Mynet', help='model name: (default: arch+timestamp)')
    parser.add_argument('--type',default='jpg',type=str)
    parser.add_argument('--fuse_Data_dir', default="DataSet/data_sequence/",type=str)
    parser.add_argument('--dict_path',default='models/Mynet/model_20.pth',type=str)#模型选择改这里
    return parser.parse_args()
def maintain_aspect_ratio_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    if width is None and height is None:
         return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

if __name__ == '__main__':
    args = parse_args()
    device = torch.device('cuda:0')
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print('GPU Mode Acitavted')
    else:
        print('CPU Mode Acitavted')

    if not os.path.exists('result/%s'% args.name+'sequence'):
        os.makedirs('result/%s' % args.name+'sequence')
    pic_sequence_list=glob.glob(args.fuse_Data_dir+'*.jpg')
    random.shuffle(pic_sequence_list)
    temp_pic_sequence_list=[None]*(len(pic_sequence_list)-1)
    # for i, data in enumerate(temp_pic_sequence_list):
    #     if len(temp_pic_sequence_list)%2==0:
    #         pic_fusion_list=[]
    #     else:
    #         pic_fusion_list = []
    for i,data in enumerate(temp_pic_sequence_list):
        if i ==0:
            t1=time.time()
            fuse=fusion(args,pic_sequence_list[i],pic_sequence_list[i+1])
            temp_pic_sequence_list[i]=fuse
            # temp_pic_sequence_list[i].save("result/{}/color_lytro_{}.{}".format(args.name+'sequence',str(i),args.type))
            cv2.imwrite("result/{}/color_lytro_{}.{}".format(args.name+'sequence',str(i),args.type),temp_pic_sequence_list[i])
            print('Complete the transition fusion{},cost:{}'.format(str(i),time.time()-t1))
        else:
            t1 = time.time()
            fuse = fusion(args, temp_pic_sequence_list[i-1], pic_sequence_list[i+1])
            temp_pic_sequence_list[i] = fuse
            cv2.imwrite("result/{}/color_lytro_{}.{}".format(args.name + 'sequence', str(i), args.type),
                        temp_pic_sequence_list[i])
            # temp_pic_sequence_list[i].save("result/{}/color_lytro_{}{}".format(args.name + 'sequence',str(i),args.type))
            print('Complete the transition fusion{},cost:{}'.format(str(i),time.time()-t1))
    print('Finish fusion!!!')



