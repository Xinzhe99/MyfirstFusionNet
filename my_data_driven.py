import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch
def rotate(image, s):
    if s == 0:
        image = image
    if s == 1:
        HF = transforms.RandomHorizontalFlip(p=1)  # 闅忔満姘村钩缈昏浆
        image = HF(image)
    if s == 2:
        VF = transforms.RandomVerticalFlip(p=1)  # 闅忔満鍨傜洿缈昏浆
        image = VF(image)
    return image

# def color2gray(image, s):
#     if s == 0:
#         image = image
#     if s == 1:
#         l = image.convert('L')
#         n = np.array(l)  # 杞寲鎴恘umpy鏁扮粍
#         image = np.expand_dims(n, axis=2)
#         image = np.concatenate((image, image, image), axis=-1)  # axis=-1灏辨槸鏈€鍚庝竴涓€氶亾
#         image = Image.fromarray(image).convert('RGB')
#     return image

class GetDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = self.imageFolderDataset[index]
        img1_tuple = img0_tuple[0:-19] + 'B' + img0_tuple[-18:]
        truth_fuse = './VOC2012_240/groundtruth' + img0_tuple[-18:-4] + '.jpg'
        img0 = Image.open(img0_tuple).convert('RGB')  # input color image pair
        img1 = Image.open(img1_tuple).convert('RGB')
        fuse = Image.open(truth_fuse).convert('RGB')
        # ------------------data enhancement--------------------------#
        j = np.random.randint(0, 3, size=1)  # 随机0-3之间的整数
        img0 = rotate(img0, j)
        img1 = rotate(img1, j)
        fuse = rotate(fuse, j)
        # ------------------To tensor------------------#
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            fuse = self.transform(fuse)
            return img0, img1, fuse

    def __len__(self):
        return len(self.imageFolderDataset)

