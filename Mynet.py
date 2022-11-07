import torch
import torch.nn as nn
import torchvision.models
from PIL import Image
from torchvision import transforms as transforms
from torchvision.transforms import ToPILImage
#协调注意力模块
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32, threshold=8):     # dafault=32, dafault=8
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(threshold, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out
#残差快
class ResblockwithCA(nn.Module):
    expansion = 1
    def __init__(self, out_planes, downsample=None, use_ca=True):
        super(ResblockwithCA, self).__init__()
        self.conv1 = nn.Conv2d(out_planes, out_planes,stride=1,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes,stride=1,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = downsample
        if use_ca:
            self.ca = CoordAtt(out_planes,out_planes)
        else:
            self.ca = None
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        if not self.ca is None:
            out = self.ca(out)
        out += residual
        out = self.relu(out)
        return out


class mynet(nn.Module):
    def __init__(self,in_ch=3,out_ch=64):
        super(mynet,self).__init__()
        self.conv_1=nn.Conv2d(in_channels=in_ch,out_channels=out_ch,kernel_size=1,stride=1,padding=0)
        self.Resblock1 = ResblockwithCA(out_ch)
        self.Resblock2 = ResblockwithCA(out_ch)
        self.Resblock3 = ResblockwithCA(out_ch)
        #reconstruct
        self.conv_2=nn.Conv2d(in_channels=out_ch,out_channels=out_ch,kernel_size=3,stride=1,padding=1)
        self.conv_3 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        #final
        self.conv_4=nn.Conv2d(in_channels=out_ch, out_channels=in_ch, kernel_size=3, stride=1, padding=1)
        # self.sig=nn.Sigmoid()
    # Element Max
    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor
    def forward(self,x1,x2):
        x1 = self.conv_1(x1)
        x2 = self.conv_1(x2)
        x1 = self.Resblock1(x1)
        x2 =self.Resblock1(x2)
        x1 = self.Resblock2(x1)
        x2= self.Resblock3(x2)
        tensor_list = [x1,x2]
        fuse = self.tensor_max(tensor_list)
        out = self.conv_2(fuse)
        out = self.conv_3(out)
        out = self.conv_4(out)
        # out = self.sig(out)
        return out
if __name__ == '__main__':
    image_path1 = 'lytro-02-A.jpg'
    image_path2 = 'lytro-02-B.jpg'
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")
    image_tensor1 = transforms.ToTensor()(img1)
    image_tensor2 = transforms.ToTensor()(img2)
    image_tensor1_add = image_tensor1.unsqueeze(0)
    image_tensor2_add = image_tensor2.unsqueeze(0)
    net = mynet()
    print(net)
    out = net(image_tensor1_add,image_tensor2_add)
    print(out.shape)
    img=torch.squeeze(out,dim=0)
    print(img.shape)
    img = ToPILImage()(img)
    img.show()
