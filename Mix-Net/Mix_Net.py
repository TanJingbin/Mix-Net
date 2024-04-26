import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F
import numpy as np


# Multi_Canny
def multi_scale_canny_down(image, low_threshold=50, high_threshold=150, blur_kernel_sizes=[3, 5, 7]):
    # 1. 初始化一个空白图像，用于保存多尺度的边缘检测结果
    multi_scale_edges = torch.zeros_like(image)

    # 2. 对每个尺度的模糊处理和边缘检测
    for blur_kernel_size in blur_kernel_sizes:
        # 将图像张量的批次维度去除，并转换为 NumPy 数组
        image_np = image[0].cpu().detach().numpy()

        # 高斯模糊
        blurred_np = cv2.GaussianBlur(image_np.transpose(1, 2, 0), (blur_kernel_size, blur_kernel_size), 0)

        # Canny 边缘检测
        edges_np = cv2.Canny(blurred_np.astype('uint8'), low_threshold, high_threshold)
        edges = torch.tensor(edges_np).to(image.device)

        # 将当前尺度的边缘检测结果添加到多尺度边缘图像中
        multi_scale_edges = torch.maximum(multi_scale_edges, edges)

    return multi_scale_edges


def multi_scale_canny_up(image, low_threshold=50, high_threshold=150, blur_kernel_sizes=[3, 5, 7]):
    # 1. 初始化一个空白图像，用于保存多尺度的边缘检测结果
    multi_scale_edges = torch.zeros_like(image)

    # 2. 对每个尺度的模糊处理和边缘检测
    for blur_kernel_size in blur_kernel_sizes:
        # 将图像张量的批次维度去除，并转换为 NumPy 数组
        image_np = image[0].cpu().detach().numpy()

        # 高斯模糊
        blurred_np = cv2.GaussianBlur(image_np, (blur_kernel_size, blur_kernel_size), 0)

        # Canny 边缘检测
        edges_np = cv2.Canny(blurred_np.astype('uint8'), low_threshold, high_threshold)
        edges = torch.tensor(edges_np).to(image.device)
        edges = edges.unsqueeze(2).expand_as(multi_scale_edges)

        # 将当前尺度的边缘检测结果添加到多尺度边缘图像中
        multi_scale_edges = torch.maximum(multi_scale_edges, edges)

    return multi_scale_edges


# WS
def weight_standardization(weight: torch.Tensor, eps: float):
    c_out, c_in, *kernel_shape = weight.shape
    weight = weight.view(c_out, -1)
    var, mean = torch.var_mean(weight, dim=1, keepdim=True)
    weight = (weight - mean) / (torch.sqrt(var + eps))
    return weight.view(c_out, c_in, *kernel_shape)


# cSE
class SeModule(nn.Module):
    def __init__(self, channel, reduction=3):
        super(SeModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        z = x * y.expand_as(x)
        return z


# sSE
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制


# scSE
class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = SeModule(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_sse = self.sSE(U)
        U_cse = self.cSE(U)
        return U_cse + U_sse


# shuffle_channel
def channel_shuffle(x, groups=3):
    batchsize, num_channels, height, width = x.data.size()
    if num_channels % groups:
        return x
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, height, width)
    return x


############################################################################
# This class is responsible for a single depth-wise separable convolution step
class dilated_downsample_up(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_downsample_up, self).__init__()
        self.GL = nn.Sequential(
            nn.GroupNorm(3, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer3(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class dilated_downsample_down(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_downsample_down, self).__init__()
        self.GL = nn.Sequential(
            nn.GroupNorm(3, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer3(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


# This class is responsible for a single depth-wise separable De-convolution step
class dilated_upsample_up(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_upsample_up, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(3, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.GL(x)
        x = self.layer3(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class dilated_upsample_down(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding, dilation):
        super(dilated_upsample_down, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(3, C_out),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, 1, 1, 0),
        )
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, kernel, stride, padding, groups=C_out, dilation=dilation),
        )
        self.SE = nn.Sequential(
            scSE(C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.GL(x)
        x = self.layer3(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = channel_shuffle(x)
        x = self.SE(x)
        return x


class multi_downsample(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super(multi_downsample, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, 1, 0),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(C_in, C_in, kernel, stride, padding, groups=C_out),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(3, C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = multi_scale_canny_down(x)
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer1(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        return x


class multi_upsample(nn.Module):
    def __init__(self, C_in, C_out, kernel, stride, padding):
        super(multi_upsample, self).__init__()
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(C_in, C_out, 1, 1, 0),
        )
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(C_out, C_out, kernel, stride, padding, groups=C_out),
        )
        self.GL = nn.Sequential(
            nn.GroupNorm(3, C_out),
        )
        self.Hswish = nn.Sequential(
            nn.Hardswish()
        )
        self.ReLU6 = nn.Sequential(
            nn.ReLU6()
        )

    def forward(self, x):
        x = multi_scale_canny_up(x)
        x = self.layer1(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        x = self.layer2(x)
        x = self.GL(x)
        x = torch.where((x >= -3) & (x <= 3), self.Hswish(x),
                        torch.clamp(x, min=0, max=6) + 1e-2 * torch.clamp(x, max=0))
        return x


# This class incorporates the convolution steps in parallel with different dilation rates and
# concatenates their output. This class is called at each level of the U-NET encoder


class BasicBlock_downsample_up(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_downsample_up, self).__init__()
        self.d1 = dilated_downsample_up(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_downsample_up(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_downsample_up(c1, c2, k3, s3, p3, dilation=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        return torch.cat([x1, x2, x3], dim=1)


class Multi_Canny_down(nn.Module):
    def __init__(self, c1, c2, k1, s1, p1):
        super(Multi_Canny_down, self).__init__()
        self.d1 = multi_downsample(c1, c2, k1, s1, p1)

    def forward(self, x):
        x = self.d1(x)
        return x


class Multi_Canny_up(nn.Module):
    def __init__(self, c1, c2, k1, s1, p1):
        super(Multi_Canny_up, self).__init__()
        self.d1 = multi_upsample(c1, c2, k1, s1, p1)

    def forward(self, x):
        x = self.d1(x)
        return x


class BasicBlock_downsample_down(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_downsample_down, self).__init__()
        self.d1 = dilated_downsample_down(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_downsample_down(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_downsample_down(c1, c2, k3, s3, p3, dilation=1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(x)
        x3 = self.d3(x)
        return torch.cat([x1, x2, x3], dim=1)


# This class incorporates the De-convolution steps in parallel with different dilation rates and
# concatenates their output. This class is called at each level of the U-NET Decoder

class BasicBlock_upsample_down(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_upsample_down, self).__init__()

        self.d1 = dilated_upsample_down(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_upsample_down(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_upsample_down(c1, c2, k3, s3, p3, dilation=1)

        self.resize = dilated_upsample_down(c2, c2, 2, 1, 0, 1)

    def forward(self, x, y=None):
        x1 = self.d1(x)
        x2 = self.resize(self.d2(x))
        x3 = self.d3(x)

        x_result = torch.cat([x1, x2, x3], dim=1)

        if y is not None:
            return torch.cat([x_result, y], dim=1)
        return x_result


class BasicBlock_upsample_up(nn.Module):
    def __init__(self, c1, c2, k1, k2, k3, s1, s2, s3, p1, p2, p3):
        super(BasicBlock_upsample_up, self).__init__()

        self.d1 = dilated_upsample_up(c1, c2, k1, s1, p1, dilation=3)
        self.d2 = dilated_upsample_up(c1, c2, k2, s2, p2, dilation=2)
        self.d3 = dilated_upsample_up(c1, c2, k3, s3, p3, dilation=1)

        self.resize = dilated_upsample_up(c2, c2, 2, 1, 0, 1)

    def forward(self, x, y=None):
        x1 = self.d1(x)
        x2 = self.resize(self.d2(x))
        x3 = self.d3(x)

        x_result = torch.cat([x1, x2, x3], dim=1)

        if y is not None:
            return torch.cat([x_result, y], dim=1)
        return x_result


class Dilated_UNET(nn.Module):
    def __init__(self):
        super(Dilated_UNET, self).__init__()
        # Encoder1 - with shape output commented for each step
        self.d2 = BasicBlock_downsample_up(3, 3, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3, p3=1)  # 36,128,128
        self.d3 = BasicBlock_downsample_up(12, 12, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3, p3=1)  # 108,64,64
        self.d4 = BasicBlock_downsample_down(48, 48, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3,
                                             p3=1)
        self.d5 = BasicBlock_downsample_down(192, 192, k1=3, k2=4, k3=4, s1=2, s2=2, s3=2, p1=3, p2=3,
                                             p3=1)

        # Decoder1 - with shape output commented for each step
        self.u1 = BasicBlock_upsample_down(768, 48, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)
        self.u2 = BasicBlock_upsample_down(384, 12, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)
        self.u3 = BasicBlock_upsample_up(96, 3, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)
        self.u4 = BasicBlock_upsample_up(24, 3, k1=4, k2=4, k3=4, s1=2, s2=2, s3=2, p1=4, p2=3, p3=1)

        # Multi-Canny-down
        self.Md1 = Multi_Canny_down(3, 3, k1=4, s1=2, p1=1)
        self.Md2 = Multi_Canny_down(12, 12, k1=4, s1=2, p1=1)
        self.Md3 = Multi_Canny_down(48, 48, k1=4, s1=2, p1=1)
        self.Md4 = Multi_Canny_down(192, 192, k1=4, s1=2, p1=1)

        # Multi-Canny-up
        self.Mu1 = Multi_Canny_up(768, 48, k1=4, s1=2, p1=1)
        self.Mu2 = Multi_Canny_up(384, 12, k1=4, s1=2, p1=1)
        self.Mu3 = Multi_Canny_up(96, 3, k1=4, s1=2, p1=1)
        self.Mu4 = Multi_Canny_up(24, 3, k1=4, s1=2, p1=1)

        # Classifier
        self.classifier = nn.Conv2d(12, 20, 3, 1, 1)

        # Dropout layers
        self.drop1 = nn.Dropout2d(p=0.2)
        self.drop2 = nn.Dropout2d(p=0.15)

    def forward(self, x):
        x1 = x
        Multi_1 = self.drop1(self.Md1(x1))
        down1 = self.drop1(self.d2(x))
        down1 = torch.cat([down1, Multi_1], dim=1)

        Multi_2 = self.drop1(self.Md2(down1))
        down2 = self.drop1(self.d3(down1))
        down2 = torch.cat([down2, Multi_2], dim=1)

        Multi_3 = self.drop1(self.Md3(down2))
        down3 = self.drop1(self.d4(down2))
        down3 = torch.cat([down3, Multi_3], dim=1)

        Multi_4 = self.drop1(self.Md4(down3))
        down4 = self.drop1(self.d5(down3))
        down4 = torch.cat([down4, Multi_4], dim=1)

        Multi_5 = self.drop2(self.Mu1(down4))
        up1 = self.drop2(self.u1(down4, down3))
        up1 = torch.cat([up1, Multi_5], dim=1)

        Multi_6 = self.drop2(self.Mu2(up1))
        up2 = self.drop2(self.u2(up1, down2))
        up2 = torch.cat([up2, Multi_6], dim=1)

        Multi_7 = self.drop2(self.Mu3(up2))
        up3 = self.drop2(self.u3(up2, down1))
        up3 = torch.cat([up3, Multi_7], dim=1)

        Multi_8 = self.drop2(self.Mu4(up3))
        up4 = self.drop2(self.u4(up3))
        up4 = torch.cat([up4, Multi_8], dim=1)

        return self.classifier(up4)
