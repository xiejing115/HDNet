
import functools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PAM(nn.Module):

    def __init__(self, dim=16, bias=False
                , act_layer=nn.PReLU, norm_layer=nn.InstanceNorm2d):
        super().__init__()
        self.signlRep = SignlRep(dim, dim, bias=bias, norm=norm_layer)
        self.norm = nn.InstanceNorm2d(dim, affine=True)
        self.mlp = Mlp(in_ch=dim, hidden_ch=int(dim * 2),out_ch=dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.signlRep(x)
        x = x + self.mlp(self.norm(x))
        return x

class Weight_fuse(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.fc = Mlp(dim, dim // 4, dim * 2)

    def forward(self, x):
        bs, c, _, _ =x[0].size()
        feats=torch.stack(x,0)
        U=sum(feats)
        S=F.adaptive_avg_pool2d(U, (1, 1))
        attention_weughts = self.fc(S).reshape(bs,c, 2,1,1).permute(2, 0, 1,3,4).softmax(dim=0)
        V = (attention_weughts * feats).sum(0)

        return V


class SignlRep(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, norm=nn.InstanceNorm2d):
        super().__init__()
        self.norm = norm(in_dim,affine=True)
        self.fcs = nn.ModuleList([])
        for i in range(9):
            self.fcs.append(nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=bias))
        self.f5 = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=5 // 2, groups=out_dim // 4, bias=bias)
        self.f7 = nn.Conv2d(in_dim, out_dim, kernel_size=7, stride=1, padding=7 // 2, groups=out_dim // 2, bias=bias)
        self.f9 = nn.Conv2d(in_dim, out_dim, kernel_size=9, stride=1, padding=9 // 2, groups=out_dim, bias=bias)

        self.fuse1 = Weight_fuse(out_dim)
        self.fuse2 = Weight_fuse(out_dim)
        self.fuse3 = Weight_fuse(out_dim)

        self.conv = nn.Conv2d(out_dim, out_dim, 1, 1, bias=True)

    def forward(self, x):
        x = self.norm(x)
        f_in=[]
        for fc in self.fcs:
            f_in.append(fc(x))
        cos= self.f5(f_in[3] * torch.cos(f_in[4]))
        sin = self.f7(f_in[5] * torch.sin(f_in[6]))
        tanh = self.f9(f_in[7] * torch.tanh(f_in[8]))
        f_cos=self.fuse1([f_in[0], cos])
        f_sin = self.fuse2([f_in[1], sin])
        f_tanh = self.fuse3([f_in[2], tanh])
        x = f_cos + f_sin + f_tanh
        x = self.conv(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_ch, hidden_ch, out_ch, act_layer=nn.PReLU):
        super().__init__()

        self.act = act_layer()
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, 1, 1)
        self.fc2 = nn.Conv2d(hidden_ch, out_ch, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class CAM(nn.Module):
    def __init__(self, in_channels, channels):
        super(CAM, self).__init__()

        self.in_r = nn.Conv2d(in_channels // 4, channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.in_g = nn.Conv2d(in_channels // 4, channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.in_b = nn.Conv2d(in_channels // 4, channels // 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_r = nn.InstanceNorm2d(channels // 2, affine=True)
        self.norm_g = nn.InstanceNorm2d(channels // 2, affine=True)
        self.norm_b = nn.InstanceNorm2d(channels // 2, affine=True)

        self.out_r = nn.Conv2d(channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_g = nn.Conv2d(channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_b = nn.Conv2d(channels // 2, in_channels // 4, kernel_size=1, stride=1, padding=0, bias=False)



    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)

        x_1 = self.in_r(x1)
        x_2 = self.in_g(x2)
        x_3 = self.in_b(x3)

        norm_r = self.norm_r(x_1)
        norm_g = self.norm_g(x_2)
        norm_b = self.norm_b(x_3)

        out_r = self.out_r(norm_r)
        out_g = self.out_g(norm_g)
        out_b = self.out_b(norm_b)

        base= out_r + out_g + out_b + x4

        rbg = torch.cat((out_r, out_g, out_b, base), dim=1)

        return rbg
    
class DRB(nn.Module):
    def __init__(self, in_channels):
        super(DRB, self).__init__()
        # 初始化卷积层和PReLU激活函数
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()

        # 第二个卷积层的输入通道数是growth_rate
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU()

        # 第三个卷积层的输入通道数是growth_rate * 2
        self.conv3 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.prelu3 = nn.PReLU()

        # 第四个卷积层的输入通道数是growth_rate * 3，因为它接收前三个卷积层的输出
        self.conv4 = nn.Conv2d(in_channels * 3, in_channels, kernel_size=3, padding=1)
        self.prelu4 = nn.PReLU()


    def forward(self, x):
        # 跳跃链接
        identity = x

        # x = self.norm(x)
        # 通过第一个卷积层
        out1 = self.prelu1(self.conv1(x))

        # 通过第二个卷积层
        out2 = self.prelu2(self.conv2(out1))

        # 将前两个卷积层的输出拼接
        combined = torch.cat([out1, out2], dim=1)

        # 通过第三个卷积层
        out3 = self.prelu3(self.conv3(combined))

        # 将前三个卷积层的输出拼接
        combined = torch.cat([combined, out3], dim=1)

        # 通过第四个卷积层
        out4 = self.prelu4(self.conv4(combined))

        # 跳跃链接，加上初始输入
        out = out4 + identity

        return out
class HDNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, channel=16):
        super(HDNet, self).__init__()

        self.channel = channel
        self.out_nc = out_nc
        self.cam1 = CAM(channel, channel * 2)  
        self.cam2 = CAM(channel, channel * 2) 
        self.pam = nn.Sequential(PAM(channel))   

        self.conv1 = nn.Conv2d(in_nc, channel, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(channel, channel, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(channel, out_nc, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(channel, out_nc, 1, 1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.fuse1 = DRB(channel)
        self.fuse2 = DRB(channel)

    def forward(self, x):

        out = self.conv1(x)  # 先将3通道扩张成
        cam1 = self.cam1(out)

        fuse_1 = self.fuse1(cam1)

        pam = self.pam(fuse_1)


        mid = self.act(pam)

        mid_out = self.conv4(mid)

        fuse2=self.fuse2(mid) 

        cam2 = self.cam2(fuse2)

        out = cam2

        out = self.act(out)

        out = self.conv3(out)

        return out, mid_out


if __name__ == '__main__':

    model = HDNet().to(torch.device("cuda:0"))
    out = model(torch.randn(16,3,256,256).to(torch.device("cuda:0")))
    print(out[0].shape)

    from ptflops import get_model_complexity_info
    import torch
    import numpy as np
    import time

    model = HDNet().cuda().eval()
    H, W = 512, 512
    flops_t, params_t = get_model_complexity_info(model, (3, H, W), as_strings=True, print_per_layer_stat=True)
    print("Network :HDNet")
    print(f"net flops:{flops_t} parameters:{params_t}")
    # model = nn.DataParallel(model)
    x = torch.ones([1, 3, H, W]).cuda()

    b_1, b_stagehead = model(x)
    steps = 50
    # print(b)
    time_avgs = []
    with torch.no_grad():
        for step in range(steps):

            torch.cuda.synchronize()
            start = time.time()
            result = model(x)
            torch.cuda.synchronize()
            time_interval = time.time() - start
            if step > 5:
                time_avgs.append(time_interval)
            # print('run time:',time_interval)
    print('avg time:', np.mean(time_avgs), 'fps:', (1 / np.mean(time_avgs)), ' size:', H, W)



