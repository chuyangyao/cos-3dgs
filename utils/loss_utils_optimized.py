"""
utils/loss_utils_optimized.py
优化的损失函数实现，使用空域方法避免FFT开销
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple, Optional
import math

# 保留原始的基础损失函数
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    """SSIM损失计算"""
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

# ============================================================
# 独立的梯度计算函数（不使用JIT编译）
# ============================================================
def compute_gradients_fast(image: Tensor) -> Tuple[Tensor, Tensor]:
    """
    快速梯度计算（不使用JIT编译以避免兼容性问题）
    """
    # 转换为灰度图
    if image.dim() == 3 and image.shape[0] == 3:
        gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
    else:
        gray = image.squeeze()
    
    # 添加batch和channel维度
    if gray.dim() == 2:
        gray = gray.unsqueeze(0).unsqueeze(0)
    
    # 简单的差分操作（更快）
    dx = gray[:, :, :, 1:] - gray[:, :, :, :-1]
    dy = gray[:, :, 1:, :] - gray[:, :, :-1, :]
    
    # Pad to original size
    dx = F.pad(dx, (0, 1, 0, 0))
    dy = F.pad(dy, (0, 0, 0, 1))
    
    # 梯度幅值
    grad_mag = torch.sqrt(dx**2 + dy**2 + 1e-8)
    
    return grad_mag.squeeze(), torch.abs(dx).squeeze() + torch.abs(dy).squeeze()

# ============================================================
# 优化的频率感知损失（空域方法）
# ============================================================

class OptimizedFrequencyLoss:
    """
    高效的频率感知损失，使用空域梯度方法
    避免FFT的计算开销
    """
    
    def __init__(self, device='cuda'):
        # 预计算Sobel算子
        self.register_kernels(device)
        
    def register_kernels(self, device):
        """注册卷积核到设备"""
        # Sobel算子
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0,  0,  0],
                                [1,  2,  1]], dtype=torch.float32)
        
        # Laplacian算子
        laplacian = torch.tensor([[0,  1,  0],
                                  [1, -4,  1],
                                  [0,  1,  0]], dtype=torch.float32)
        
        # 扩展维度用于conv2d
        self.sobel_x = sobel_x.view(1, 1, 3, 3).to(device)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).to(device)
        self.laplacian = laplacian.view(1, 1, 3, 3).to(device)
    
    def compute_multi_scale_gradients(self, image: Tensor, levels: int = 3) -> Dict[str, Tensor]:
        """
        计算多尺度梯度（金字塔方法）
        不同尺度对应不同频率
        """
        gradients = {}
        current = image
        
        for level in range(levels):
            # 计算当前尺度的梯度（使用独立函数）
            grad_mag, grad_sum = compute_gradients_fast(current)
            gradients[f'level_{level}'] = grad_mag
            
            # 下采样到下一尺度
            if level < levels - 1:
                current = F.avg_pool2d(current.unsqueeze(0) if current.dim() == 3 else current, 
                                       kernel_size=2, stride=2).squeeze(0)
        
        return gradients
    
    def compute_loss(self, rendered: Tensor, gt: Tensor, 
                    wave_vectors: Optional[Tensor] = None,
                    visibility_filter: Optional[Tensor] = None,
                    iteration: int = 0) -> Tuple[Tensor, Dict]:
        """
        计算优化的频率感知损失
        
        Args:
            rendered: 渲染图像
            gt: 真实图像
            wave_vectors: Wave参数（用于引导）
            visibility_filter: 可见性掩码
            iteration: 当前迭代次数
        
        Returns:
            total_loss: 总损失
            loss_dict: 损失分量字典
        """
        loss_dict = {}
        
        # 1. 基础L1和SSIM损失
        L_l1 = l1_loss(rendered, gt)
        L_ssim = 1.0 - ssim(rendered.unsqueeze(0) if rendered.dim() == 3 else rendered,
                            gt.unsqueeze(0) if gt.dim() == 3 else gt)
        
        loss_dict['l1'] = L_l1.item()
        loss_dict['ssim'] = L_ssim.item()
        
        # 2. 快速梯度损失（代替FFT）- 使用独立函数
        rendered_grad, rendered_edges = compute_gradients_fast(rendered)
        gt_grad, gt_edges = compute_gradients_fast(gt)
        
        # 边缘损失（高频）
        edge_loss = F.l1_loss(rendered_edges, gt_edges)
        loss_dict['edge'] = edge_loss.item()
        
        # 梯度损失
        grad_loss = F.l1_loss(rendered_grad, gt_grad)
        loss_dict['gradient'] = grad_loss.item()
        
        # 3. 多尺度损失（可选，更准确但稍慢）
        if iteration % 100 == 0:  # 每100次迭代计算一次
            rendered_ms = self.compute_multi_scale_gradients(rendered, levels=2)
            gt_ms = self.compute_multi_scale_gradients(gt, levels=2)
            
            ms_loss = 0
            for level in rendered_ms:
                ms_loss += F.l1_loss(rendered_ms[level], gt_ms[level])
            ms_loss /= len(rendered_ms)
            loss_dict['multi_scale'] = ms_loss.item()
        else:
            ms_loss = torch.tensor(0.0, device=rendered.device)
        
        # 4. Wave引导损失
        wave_loss = torch.tensor(0.0, device=rendered.device)
        if wave_vectors is not None:
            # 计算缺失的高频细节
            missing_details = torch.relu(gt_edges - rendered_edges)
            target_wave_strength = missing_details.mean() * 2.0
            
            # 处理visibility_filter
            if visibility_filter is not None and visibility_filter.shape[0] == wave_vectors.shape[0]:
                visible_waves = wave_vectors[visibility_filter] if visibility_filter.sum() > 0 else wave_vectors
            else:
                visible_waves = wave_vectors
            
            wave_norms = torch.norm(visible_waves, dim=1)
            
            # 引导wave朝向需要高频的方向
            wave_loss = F.smooth_l1_loss(wave_norms.mean(), target_wave_strength)
            
            # 稀疏性正则化
            sparsity_loss = torch.sqrt(wave_norms + 1e-8).mean() * 0.0001
            wave_loss = wave_loss + sparsity_loss
            
            loss_dict['wave_guide'] = wave_loss.item()
        
        # 5. 动态权重调整
        progress = min(iteration / 30000, 1.0)
        
        # 早期重视基础重建，后期重视细节
        w_base = 1.0 - 0.3 * progress  # 1.0 -> 0.7
        w_detail = 0.2 + 0.8 * progress  # 0.2 -> 1.0
        
        # 组合总损失
        total_loss = (
            w_base * (0.8 * L_l1 + 0.2 * L_ssim) +  # 基础损失
            w_detail * (0.5 * edge_loss + 0.3 * grad_loss + 0.2 * ms_loss) +  # 细节损失
            0.001 * wave_loss  # Wave正则化
        )
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict

# ============================================================
# 简化版损失（用于快速调试）
# ============================================================

class SimplifiedFrequencyLoss:
    """极简的频率感知损失，用于快速调试"""
    
    def compute_loss(self, rendered: Tensor, gt: Tensor,
                    wave_vectors: Optional[Tensor] = None,
                    iteration: int = 0) -> Tuple[Tensor, Dict]:
        """
        简化的损失计算，只保留核心部分
        """
        # 基础损失
        L_l1 = l1_loss(rendered, gt)
        L_ssim = 1.0 - ssim(rendered.unsqueeze(0) if rendered.dim() == 3 else rendered,
                            gt.unsqueeze(0) if gt.dim() == 3 else gt)
        
        # 简单的梯度损失（最快的高频指标）
        dx_r = rendered[:, :, 1:] - rendered[:, :, :-1] if rendered.dim() == 3 else rendered[:, 1:] - rendered[:, :-1]
        dx_g = gt[:, :, 1:] - gt[:, :, :-1] if gt.dim() == 3 else gt[:, 1:] - gt[:, :-1]
        grad_loss = F.l1_loss(dx_r, dx_g)
        
        # Wave正则化
        wave_reg = 0.0
        if wave_vectors is not None:
            wave_norms = torch.norm(wave_vectors, dim=1)
            wave_reg = wave_norms.mean() * 0.001
            
            # 防止wave过大
            if wave_norms.max() > 5.0:
                wave_reg += torch.relu(wave_norms - 5.0).mean() * 0.01
        
        # 总损失
        total_loss = 0.8 * L_l1 + 0.2 * L_ssim + 0.1 * grad_loss + wave_reg
        
        return total_loss, {
            'l1': L_l1.item(),
            'ssim': L_ssim.item(),
            'gradient': grad_loss.item(),
            'wave_reg': wave_reg,
            'total': total_loss.item()
        }

# ============================================================
# 全局函数接口（向后兼容）
# ============================================================

# 创建全局实例
_global_freq_loss = OptimizedFrequencyLoss()
_global_simple_loss = SimplifiedFrequencyLoss()

def compute_optimized_loss(rendered, gt, wave_vectors=None, visibility_filter=None, 
                          iteration=0, use_simple=False):
    """
    统一的损失接口
    
    Args:
        use_simple: 是否使用简化版本（更快）
    """
    if use_simple:
        return _global_simple_loss.compute_loss(rendered, gt, wave_vectors, iteration)
    else:
        return _global_freq_loss.compute_loss(rendered, gt, wave_vectors, visibility_filter, iteration)