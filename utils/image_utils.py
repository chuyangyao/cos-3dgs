#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn as nn
import torch.nn.functional as F
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

class DoGFilter(nn.Module):
    def __init__(self, channels, sigma1):
        super(DoGFilter, self).__init__()
        self.channels = channels
        self.sigma1 = sigma1
        self.sigma2 = 2 * sigma1  # Ensure the 1:2 ratio
        self.kernel_size1 = int(2 * round(3 * self.sigma1) + 1)
        self.kernel_size2 = int(2 * round(3 * self.sigma2) + 1)
        self.padding1 = (self.kernel_size1 - 1) // 2
        self.padding2 = (self.kernel_size2 - 1) // 2
        self.weight1 = self.get_gaussian_kernel(self.kernel_size1, self.sigma1)
        self.weight2 = self.get_gaussian_kernel(self.kernel_size2, self.sigma2)


    def get_gaussian_kernel(self, kernel_size, sigma):
        x_cord = torch.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma**2.
        
        kernel = torch.exp(-(xy_grid - mean).pow(2).sum(dim=-1) / (2 * variance))
        kernel = kernel / kernel.sum()  # Normalize the kernel
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        
        return kernel

    @torch.no_grad()
    def forward(self, x):
        gaussian1 = F.conv2d(x, self.weight1.to(x.device), bias=None, stride=1, padding=self.padding1, groups=self.channels)
        gaussian2 = F.conv2d(x, self.weight2.to(x.device), bias=None, stride=1, padding=self.padding2, groups=self.channels)
        return gaussian1 - gaussian2
def apply_dog_filter(batch, freq=50, scale_factor=0.5):
    """
    Apply a Difference of Gaussian filter to a batch of images.
    
    Args:
        batch: torch.Tensor, shape (B, C, H, W)
        freq: Control variable ranging from 0 to 100.
              - 0 means original image
              - 1.0 means smoother difference
              - 100 means sharpest difference
        scale_factor: Factor by which the image is downscaled before applying DoG.
    
    Returns:
        torch.Tensor: Processed image using DoG.
    """
    # Convert to grayscale if it's a color image
    if batch.size(1) == 3:
        batch = torch.mean(batch, dim=1, keepdim=True)

    # Downscale the image
    downscaled = F.interpolate(batch, scale_factor=scale_factor, mode='bilinear', align_corners=False)

    channels = downscaled.size(1)

    # Set sigma1 value based on freq parameter. sigma2 will be 2*sigma1.
    sigma1 = 0.1 + (100 - freq) * 0.1 if freq >=50 else 0.1 + freq * 0.1

    dog_filter = DoGFilter(channels, sigma1)
    mask = dog_filter(downscaled)

    # Upscale the mask back to original size
    upscaled_mask = F.interpolate(mask, size=batch.shape[-2:], mode='bilinear', align_corners=False)

    upscaled_mask = upscaled_mask - upscaled_mask.min()
    upscaled_mask = upscaled_mask / upscaled_mask.max() if freq >=50 else  1.0 - upscaled_mask / upscaled_mask.max()
    
    upscaled_mask = (upscaled_mask >=0.5).to(torch.float)
    return upscaled_mask[:,0,...]

# 新增：边缘质量评估
class EdgeDetector(nn.Module):
    def __init__(self, channels=1):
        super(EdgeDetector, self).__init__()
        self.channels = channels
        
        # Sobel算子
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                    dtype=torch.float32).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                    dtype=torch.float32).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        
    def to(self, device):
        self.sobel_x = self.sobel_x.to(device)
        self.sobel_y = self.sobel_y.to(device)
        return self
        
    @torch.no_grad()
    def forward(self, x):
        # 确保输入是灰度图
        if x.size(1) == 3:
            x_gray = 0.299 * x[:,0:1] + 0.587 * x[:,1:2] + 0.114 * x[:,2:3]
            # 重新创建与灰度图匹配的卷积核
            channels = 1
            sobel_x = self.sobel_x[0:1]
            sobel_y = self.sobel_y[0:1]
        else:
            x_gray = x
            channels = x.size(1)
            sobel_x = self.sobel_x
            sobel_y = self.sobel_y
        
        # 使用Sobel算子检测边缘
        grad_x = F.conv2d(F.pad(x_gray, (1, 1, 1, 1), mode='reflect'), 
                          sobel_x, groups=channels)
        grad_y = F.conv2d(F.pad(x_gray, (1, 1, 1, 1), mode='reflect'), 
                          sobel_y, groups=channels)
        
        # 计算梯度幅度
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
        return grad_magnitude

def edge_psnr(img1, img2, threshold=0.1):
    """
    计算边缘区域的PSNR，用于评估高频细节恢复质量
    
    Args:
        img1: 预测图像
        img2: 真实图像
        threshold: 边缘区域阈值
    
    Returns:
        float: 边缘区域的PSNR值
    """
    # 确保创建正确通道数的边缘检测器
    input_channels = img1.size(1)
    edge_detector = EdgeDetector(input_channels).to(img1.device)
    edges_gt = edge_detector(img2)
    
    # 二值化边缘区域
    edge_mask = (edges_gt > threshold).float()
    
    # 计算边缘区域的MSE
    edge_mse = (((img1 - img2) * edge_mask) ** 2).sum() / (edge_mask.sum() + 1e-8)
    edge_psnr_value = 20 * torch.log10(1.0 / torch.sqrt(edge_mse + 1e-8))
    
    return edge_psnr_value

# 新增：多频段质量评估
def frequency_band_metric(img1, img2, bands=4):
    """
    在多个频率带上评估图像质量
    
    Args:
        img1: 预测图像
        img2: 真实图像
        bands: 频率带数量
    
    Returns:
        dict: 包含各频段的质量指标
    """
    # 确保输入为灰度图
    if img1.size(1) == 3:
        img1_gray = 0.299 * img1[:,0:1] + 0.587 * img1[:,1:2] + 0.114 * img1[:,2:3]
        img2_gray = 0.299 * img2[:,0:1] + 0.587 * img2[:,1:2] + 0.114 * img2[:,2:3]
    else:
        img1_gray = img1
        img2_gray = img2
    
    # 转到频域
    fft1 = torch.fft.rfft2(img1_gray)
    fft2 = torch.fft.rfft2(img2_gray)
    
    # 计算频域幅度
    mag1 = torch.abs(fft1)
    mag2 = torch.abs(fft2)
    
    # 创建频段掩码
    B, C, H, W_rfft = mag1.shape
    result = {}
    
    for band in range(bands):
        # 创建环形掩码，将频率分为几个频带
        y_coords = torch.arange(H, device=mag1.device).view(-1, 1).repeat(1, W_rfft)
        x_coords = torch.arange(W_rfft, device=mag1.device).view(1, -1).repeat(H, 1)
        
        center_y, center_x = 0, 0
        dist_from_center = torch.sqrt((y_coords / H - center_y)**2 + (x_coords / W_rfft - center_x)**2)
        
        # 将距离归一化到[0,1]
        normalized_dist = dist_from_center / torch.sqrt(1.0)
        
        # 定义当前频带的范围
        band_min = band / bands
        band_max = (band + 1) / bands
        
        # 创建当前频带掩码
        band_mask = ((normalized_dist >= band_min) & (normalized_dist < band_max)).float()
        band_mask = band_mask.unsqueeze(0).unsqueeze(0).expand_as(mag1)
        
        # 计算当前频带的差异
        band_diff = torch.abs(mag1 - mag2) * band_mask
        band_mse = (band_diff**2).sum() / (band_mask.sum() + 1e-8)
        band_psnr = 20 * torch.log10(mag1.max() / (torch.sqrt(band_mse) + 1e-8))
        
        result[f'band_{band}_psnr'] = band_psnr.item()
    
    return result

# 新增：纹理细节评估
class TextureDetailMetric:
    @staticmethod
    @torch.no_grad()
    def compute(img1, img2, patch_size=16, stride=8):
        """
        评估纹理细节的恢复质量
        
        Args:
            img1: 预测图像
            img2: 真实图像
            patch_size: 评估用的小块大小
            stride: 移动步长
        
        Returns:
            dict: 包含纹理评估指标
        """
        B, C, H, W = img1.shape
        
        # 计算Gram矩阵 (纹理表示)
        def gram_matrix(x, normalize=True):
            b, c, h, w = x.size()
            features = x.view(b, c, h * w)
            gram = torch.bmm(features, features.transpose(1, 2))
            if normalize:
                gram = gram / (c * h * w)
            return gram
        
        # 使用滑动窗口计算每个patch的纹理相似度
        texture_corr = []
        variance_ratio = []
        
        for i in range(0, H - patch_size + 1, stride):
            for j in range(0, W - patch_size + 1, stride):
                patch1 = img1[:, :, i:i+patch_size, j:j+patch_size]
                patch2 = img2[:, :, i:i+patch_size, j:j+patch_size]
                
                # 计算每个patch的Gram矩阵
                gram1 = gram_matrix(patch1)
                gram2 = gram_matrix(patch2)
                
                # 计算纹理相关性
                cos_sim = F.cosine_similarity(gram1.view(B, -1), gram2.view(B, -1), dim=1)
                texture_corr.append(cos_sim)
                
                # 计算局部方差比率(高频细节指标)
                var1 = torch.var(patch1, dim=(2, 3))
                var2 = torch.var(patch2, dim=(2, 3))
                ratio = var1 / (var2 + 1e-8)  # 防止除零
                variance_ratio.append(ratio)
        
        # 整合所有patch的结果
        texture_corr = torch.cat(texture_corr, dim=0).mean()
        variance_ratio = torch.cat(variance_ratio, dim=0).mean(dim=0)  # 按通道计算
        
        return {
            'texture_correlation': texture_corr.item(),
            'variance_ratio': variance_ratio.mean().item(),  # 均值化为单一指标
        }
