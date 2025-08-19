import torch
import torch.nn.functional as F
from math import exp
import numpy as np
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
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

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def compute_gradient_loss(rendered, gt):
    """
    计算图像梯度损失，用于捕捉高频细节
    使用Sobel算子计算更准确的梯度
    """
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=rendered.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=rendered.device)
    
    # 重塑为卷积核 [out_channels, in_channels, H, W]
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # 处理多通道图像
    if rendered.dim() == 3 and rendered.shape[0] == 3:  # RGB图像
        # 转换为灰度
        rendered_gray = 0.299 * rendered[0] + 0.587 * rendered[1] + 0.114 * rendered[2]
        gt_gray = 0.299 * gt[0] + 0.587 * gt[1] + 0.114 * gt[2]
        rendered_gray = rendered_gray.unsqueeze(0).unsqueeze(0)
        gt_gray = gt_gray.unsqueeze(0).unsqueeze(0)
    else:
        rendered_gray = rendered.unsqueeze(0) if rendered.dim() == 2 else rendered
        gt_gray = gt.unsqueeze(0) if gt.dim() == 2 else gt
        if rendered_gray.dim() == 3:
            rendered_gray = rendered_gray.unsqueeze(0)
            gt_gray = gt_gray.unsqueeze(0)
    
    # 计算梯度
    grad_x_rendered = F.conv2d(rendered_gray, sobel_x, padding=1)
    grad_y_rendered = F.conv2d(rendered_gray, sobel_y, padding=1)
    grad_x_gt = F.conv2d(gt_gray, sobel_x, padding=1)
    grad_y_gt = F.conv2d(gt_gray, sobel_y, padding=1)
    
    # 梯度幅度
    grad_mag_rendered = torch.sqrt(grad_x_rendered**2 + grad_y_rendered**2 + 1e-8)
    grad_mag_gt = torch.sqrt(grad_x_gt**2 + grad_y_gt**2 + 1e-8)
    
    # L1损失
    grad_loss = torch.abs(grad_mag_rendered - grad_mag_gt).mean()
    
    return grad_loss

def compute_wave_regularization(wave_vectors, iteration, max_iter, phase=1):
    """
    Wave正则化：鼓励稀疏性和合理范围
    
    Args:
        wave_vectors: wave参数
        iteration: 当前迭代
        max_iter: 最大迭代
        phase: 训练阶段 (1, 2, 3)
    """
    if wave_vectors is None:
        return torch.tensor(0.0), {}
    
    wave_norms = torch.norm(wave_vectors, dim=1)
    
    # 稀疏性损失（L1正则化）
    sparsity_loss = wave_norms.mean()
    
    # 范围约束
    max_wave = 5.0  # 对应最小波长约1.26
    overflow_loss = torch.relu(wave_norms - max_wave).mean()
    
    # 根据阶段调整权重
    if phase == 1:
        # 阶段1：不应该有wave
        reg_weight = 10.0  # 强正则化
    elif phase == 2:
        # 阶段2：允许wave学习，但保持适度
        progress = min(iteration / max_iter, 1.0)
        reg_weight = 0.1 * (1.0 - 0.5 * progress)  # 从0.1降到0.05
    else:  # phase == 3
        # 阶段3：减少正则化，允许更自由的wave
        reg_weight = 0.01
    
    total_reg = reg_weight * (sparsity_loss + 0.5 * overflow_loss)
    
    stats = {
        'sparsity': sparsity_loss.item(),
        'overflow': overflow_loss.item(),
        'weight': reg_weight,
        'mean_norm': wave_norms.mean().item(),
        'max_norm': wave_norms.max().item() if wave_norms.numel() > 0 else 0.0,
        'active_ratio': (wave_norms > 0.01).sum().item() / wave_norms.numel() if wave_norms.numel() > 0 else 0.0
    }
    
    return total_reg, stats

def compute_comprehensive_frequency_loss(rendered, gt, wave_vectors=None, 
                                        visibility_filter=None, iteration=0, 
                                        max_iter=40000, phase=1):
    """
    综合损失函数，根据训练阶段调整
    
    Args:
        rendered: 渲染图像
        gt: 真实图像
        wave_vectors: wave参数
        visibility_filter: 可见性过滤器
        iteration: 当前迭代
        max_iter: 最大迭代
        phase: 训练阶段 (1, 2, 3)
    """
    loss_dict = {}
    
    # 1. 基础L1损失（始终重要）
    L_l1 = l1_loss(rendered, gt)
    loss_dict['l1'] = L_l1.item()
    
    # 2. SSIM损失（始终重要）
    if rendered.dim() == 3:
        rendered_4d = rendered.unsqueeze(0)
        gt_4d = gt.unsqueeze(0)
    else:
        rendered_4d = rendered
        gt_4d = gt
    
    ssim_val = ssim(rendered_4d, gt_4d)
    L_ssim = 1.0 - ssim_val
    loss_dict['ssim'] = L_ssim.item()
    
    # 3. 梯度损失（阶段2和3）
    L_grad = torch.tensor(0.0, device=rendered.device)
    if phase >= 2:
        L_grad = compute_gradient_loss(rendered, gt)
        loss_dict['gradient'] = L_grad.item()
    else:
        loss_dict['gradient'] = 0.0
    
    # 4. Wave正则化（阶段2和3）
    L_wave_reg = torch.tensor(0.0, device=rendered.device)
    if wave_vectors is not None and phase >= 2:
        # 如果有visibility_filter，只对可见的高斯计算
        if visibility_filter is not None and visibility_filter.sum() > 0:
            visible_waves = wave_vectors[visibility_filter]
        else:
            visible_waves = wave_vectors
        
        L_wave_reg, wave_stats = compute_wave_regularization(
            visible_waves, iteration, max_iter, phase
        )
        loss_dict.update({f'wave_{k}': v for k, v in wave_stats.items()})
    
    # 权重调整
    lambda_dssim = 0.2  # 原始3DGS权重
    
    # 根据阶段调整权重
    if phase == 1:
        # 阶段1：只有L1和SSIM
        grad_weight = 0.0
        wave_weight = 0.0
    elif phase == 2:
        # 阶段2：引入梯度损失，开始wave学习
        phase_progress = (iteration - 15000) / 15000.0 if iteration > 15000 else 0.0
        grad_weight = 0.05 * phase_progress  # 从0增加到0.05
        wave_weight = 1.0  # 正常wave正则化
    else:  # phase == 3
        # 阶段3：保持梯度损失，减少wave正则化
        grad_weight = 0.05
        wave_weight = 0.5  # 减少正则化，允许更自由的wave
    
    # 组合总损失
    total_loss = (
        (1.0 - lambda_dssim) * L_l1 +      # 0.8 * L1
        lambda_dssim * L_ssim +             # 0.2 * SSIM  
        grad_weight * L_grad +              # 梯度损失
        wave_weight * L_wave_reg            # Wave正则化
    )
    
    loss_dict['total'] = total_loss.item()
    loss_dict['grad_weight'] = grad_weight
    loss_dict['phase'] = phase
    
    return total_loss, loss_dict

# ===================== 保留的辅助函数（向后兼容） =====================

def compute_frequency_loss(img1, img2, frequency_weights=None):
    """向后兼容"""
    return compute_gradient_loss(img1, img2)

def compute_wave_sparsity_loss(wave_vectors, temperature=0.1):
    """向后兼容"""
    if wave_vectors is None:
        return torch.tensor(0.0)
    wave_norms = torch.norm(wave_vectors, dim=1)
    return wave_norms.mean()

def compute_wave_smoothness_loss(wave_vectors, positions, k_nearest=8):
    """向后兼容"""
    return torch.tensor(0.0)

def compute_wave_diversity_loss(wave_vectors, target_std=0.1):
    """向后兼容"""
    return torch.tensor(0.0)

def compute_wave_gradient_guidance_loss(wave_vectors, rendered, gt, temperature=0.1):
    """向后兼容"""
    return compute_gradient_loss(rendered, gt) * 0.01

def compute_adaptive_frequency_loss(rendered, gt, wave_vectors, iteration, max_iter):
    """向后兼容"""
    phase = 2 if iteration > 15000 else 1
    loss, loss_dict = compute_comprehensive_frequency_loss(
        rendered, gt, wave_vectors, None, iteration, max_iter, phase
    )
    return loss, loss_dict

def compute_frequency_spectrum_loss(rendered, gt):
    """简化版本"""
    grad_loss = compute_gradient_loss(rendered, gt)
    return grad_loss * 0.5, grad_loss

def compute_multi_scale_gradient_loss(rendered, gt, scales=[1, 2, 4]):
    """简化版本"""
    return compute_gradient_loss(rendered, gt), {}

def compute_local_frequency_guidance_loss(rendered, gt, wave_vectors, visibility_filter, device='cuda'):
    """简化版本"""
    grad_loss = compute_gradient_loss(rendered, gt)
    return grad_loss * 0.01, {}

def laplacian_loss(rendered, gt, max_levels=3):
    """简化版本"""
    return compute_gradient_loss(rendered, gt)

def compute_simple_frequency_guidance(rendered, gt, wave_vectors, k=10):
    """简化版本"""
    return torch.tensor(1.0, device=rendered.device)

def compute_high_frequency_detail_loss(rendered, gt, wave_vectors=None, opt=None):
    """
    专门针对高频细节（如草地纹理）的损失函数
    综合多种高频检测方法
    """
    device = rendered.device
    loss_dict = {}
    
    # 确保输入格式正确
    if rendered.dim() == 3:  # C,H,W
        rendered = rendered.unsqueeze(0)  # -> B,C,H,W
    if gt.dim() == 3:  # C,H,W
        gt = gt.unsqueeze(0)  # -> B,C,H,W
    
    # 1. 边缘检测损失（Sobel + Laplacian）
    edge_loss = compute_edge_aware_loss(rendered, gt)
    loss_dict['edge'] = edge_loss.item()
    
    # 2. 纹理丰富度损失
    texture_loss = compute_texture_richness_loss(rendered, gt)
    loss_dict['texture'] = texture_loss.item()
    
    # 3. 频域损失
    freq_loss = compute_frequency_domain_loss(rendered, gt)
    loss_dict['frequency'] = freq_loss.item()
    
    # 4. 局部对比度损失
    contrast_loss = compute_local_contrast_loss(rendered, gt)
    loss_dict['contrast'] = contrast_loss.item()
    
    # 5. Wave引导损失
    wave_loss = torch.tensor(0.0, device=device)
    if wave_vectors is not None and opt is not None:
        wave_loss = compute_wave_guidance_loss(
            rendered, gt, wave_vectors, opt
        )
        loss_dict['wave_guide'] = wave_loss.item()
    
    # 组合总损失（使用opt中的权重）
    if opt is not None:
        total_loss = (
            edge_loss * opt.edge_loss_weight +
            texture_loss * opt.texture_loss_weight +
            freq_loss * opt.high_freq_loss_weight +
            contrast_loss * 0.05 +
            wave_loss
        )
    else:
        total_loss = edge_loss * 0.15 + texture_loss * 0.1 + freq_loss * 0.2 + contrast_loss * 0.05 + wave_loss
    
    loss_dict['total_high_freq'] = total_loss.item()
    
    return total_loss, loss_dict


def compute_edge_aware_loss(rendered, gt):
    """
    边缘感知损失，使用多种边缘检测算子
    """
    device = rendered.device
    
    # Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # Laplacian算子
    laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], 
                            dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # 转换为灰度
    rendered_gray = 0.299 * rendered[:, 0] + 0.587 * rendered[:, 1] + 0.114 * rendered[:, 2]
    gt_gray = 0.299 * gt[:, 0] + 0.587 * gt[:, 1] + 0.114 * gt[:, 2]
    rendered_gray = rendered_gray.unsqueeze(1)
    gt_gray = gt_gray.unsqueeze(1)
    
    # Sobel边缘
    rendered_sx = F.conv2d(rendered_gray, sobel_x, padding=1)
    rendered_sy = F.conv2d(rendered_gray, sobel_y, padding=1)
    gt_sx = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_sy = F.conv2d(gt_gray, sobel_y, padding=1)
    
    rendered_sobel = torch.sqrt(rendered_sx**2 + rendered_sy**2 + 1e-8)
    gt_sobel = torch.sqrt(gt_sx**2 + gt_sy**2 + 1e-8)
    
    # Laplacian边缘
    rendered_lap = F.conv2d(rendered_gray, laplacian, padding=1)
    gt_lap = F.conv2d(gt_gray, laplacian, padding=1)
    
    # 组合损失
    sobel_loss = F.l1_loss(rendered_sobel, gt_sobel)
    lap_loss = F.l1_loss(rendered_lap, gt_lap)
    
    return sobel_loss * 0.7 + lap_loss * 0.3


def compute_texture_richness_loss(rendered, gt, window_sizes=[3, 5, 7]):
    """
    纹理丰富度损失，使用多尺度局部统计
    """
    device = rendered.device
    total_loss = torch.tensor(0.0, device=device)
    
    # 转换为灰度
    rendered_gray = 0.299 * rendered[:, 0] + 0.587 * rendered[:, 1] + 0.114 * rendered[:, 2]
    gt_gray = 0.299 * gt[:, 0] + 0.587 * gt[:, 1] + 0.114 * gt[:, 2]
    rendered_gray = rendered_gray.unsqueeze(1)
    gt_gray = gt_gray.unsqueeze(1)
    
    for window_size in window_sizes:
        # 使用unfold提取patches
        unfold = torch.nn.Unfold(kernel_size=window_size, stride=1, padding=window_size//2)
        
        rendered_patches = unfold(rendered_gray)
        gt_patches = unfold(gt_gray)
        
        # 计算局部统计
        rendered_mean = rendered_patches.mean(dim=1, keepdim=True)
        rendered_var = rendered_patches.var(dim=1, keepdim=True)
        rendered_std = torch.sqrt(rendered_var + 1e-8)
        
        gt_mean = gt_patches.mean(dim=1, keepdim=True)
        gt_var = gt_patches.var(dim=1, keepdim=True)
        gt_std = torch.sqrt(gt_var + 1e-8)
        
        # 方差损失（纹理强度）
        var_loss = F.l1_loss(rendered_var, gt_var)
        
        # 标准差损失（对比度）
        std_loss = F.l1_loss(rendered_std, gt_std)
        
        # 加权组合
        weight = 1.0 / window_size  # 小窗口权重更大
        total_loss += weight * (var_loss + std_loss)
    
    return total_loss / len(window_sizes)


def compute_frequency_domain_loss(rendered, gt):
    """
    频域损失，评估高频成分的重建质量
    """
    # 转换为灰度
    if rendered.shape[1] == 3:
        rendered_gray = 0.299 * rendered[:, 0] + 0.587 * rendered[:, 1] + 0.114 * rendered[:, 2]
        gt_gray = 0.299 * gt[:, 0] + 0.587 * gt[:, 1] + 0.114 * gt[:, 2]
        rendered_gray = rendered_gray.unsqueeze(1)
        gt_gray = gt_gray.unsqueeze(1)
    else:
        rendered_gray = rendered
        gt_gray = gt
    
    # FFT变换
    rendered_fft = torch.fft.rfft2(rendered_gray, norm='ortho')
    gt_fft = torch.fft.rfft2(gt_gray, norm='ortho')
    
    # 幅度谱
    rendered_mag = torch.abs(rendered_fft) + 1e-8
    gt_mag = torch.abs(gt_fft) + 1e-8
    
    # 使用log幅度谱（更稳定）
    rendered_log_mag = torch.log(rendered_mag)
    gt_log_mag = torch.log(gt_mag)
    
    # 创建高频掩码（外围区域）
    B, C, H, W_rfft = rendered_mag.shape
    high_freq_mask = torch.ones_like(rendered_mag)
    high_freq_mask[:, :, :H//4, :W_rfft//4] = 0.0  # 去除低频中心
    
    # 高频损失
    diff = (rendered_log_mag - gt_log_mag) ** 2
    high_freq_loss = (diff * high_freq_mask).sum() / (high_freq_mask.sum() + 1e-8)
    
    return high_freq_loss


def compute_local_contrast_loss(rendered, gt, window_size=11):
    """
    局部对比度损失
    """
    device = rendered.device
    
    # 转换为灰度
    rendered_gray = 0.299 * rendered[:, 0] + 0.587 * rendered[:, 1] + 0.114 * rendered[:, 2]
    gt_gray = 0.299 * gt[:, 0] + 0.587 * gt[:, 1] + 0.114 * gt[:, 2]
    rendered_gray = rendered_gray.unsqueeze(1)
    gt_gray = gt_gray.unsqueeze(1)
    
    # 计算局部均值
    kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size ** 2)
    rendered_mean = F.conv2d(rendered_gray, kernel, padding=window_size//2)
    gt_mean = F.conv2d(gt_gray, kernel, padding=window_size//2)
    
    # 计算局部对比度
    rendered_contrast = torch.abs(rendered_gray - rendered_mean)
    gt_contrast = torch.abs(gt_gray - gt_mean)
    
    # 对比度损失
    contrast_loss = F.l1_loss(rendered_contrast, gt_contrast)
    
    return contrast_loss


def compute_wave_guidance_loss(rendered, gt, wave_vectors, opt):
    """
    Wave引导损失，根据重建误差引导wave学习
    """
    device = rendered.device
    
    # 计算需要重建的高频细节
    edge_error = compute_edge_reconstruction_error(rendered, gt)
    
    # 统计信息
    mean_error = edge_error.mean()
    std_error = edge_error.std()
    max_error = edge_error.max()
    
    # 计算目标wave强度
    # 误差越大，需要的wave越强
    target_wave_strength = (mean_error * 3.0 + std_error * 1.0).clamp(0, opt.max_wave_norm)
    
    # 当前wave统计
    wave_norms = torch.norm(wave_vectors, dim=1)
    current_mean = wave_norms.mean()
    current_std = wave_norms.std()
    
    # 引导损失
    strength_loss = F.smooth_l1_loss(current_mean, target_wave_strength)
    
    # 多样性损失（防止所有wave都一样）
    diversity_target = target_wave_strength * 0.3  # 期望的标准差
    diversity_loss = F.smooth_l1_loss(current_std, diversity_target)
    
    # 稀疏性损失
    sparsity_loss = torch.sqrt(wave_norms + 1e-8).mean() * opt.wave_sparsity_weight
    
    # 上界约束
    over_limit = torch.relu(wave_norms - opt.max_wave_norm)
    limit_loss = over_limit.mean() * 0.1
    
    total_loss = strength_loss * 0.5 + diversity_loss * 0.2 + sparsity_loss + limit_loss
    
    return total_loss


def compute_edge_reconstruction_error(rendered, gt):
    """
    计算边缘重建误差
    """
    device = rendered.device
    
    # Sobel边缘检测
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                          dtype=torch.float32, device=device).view(1, 1, 3, 3)
    
    # 转换为灰度
    rendered_gray = 0.299 * rendered[:, 0] + 0.587 * rendered[:, 1] + 0.114 * rendered[:, 2]
    gt_gray = 0.299 * gt[:, 0] + 0.587 * gt[:, 1] + 0.114 * gt[:, 2]
    rendered_gray = rendered_gray.unsqueeze(1)
    gt_gray = gt_gray.unsqueeze(1)
    
    # 计算边缘
    rendered_edges_x = F.conv2d(rendered_gray, sobel_x, padding=1)
    rendered_edges_y = F.conv2d(rendered_gray, sobel_y, padding=1)
    gt_edges_x = F.conv2d(gt_gray, sobel_x, padding=1)
    gt_edges_y = F.conv2d(gt_gray, sobel_y, padding=1)
    
    rendered_edges = torch.sqrt(rendered_edges_x**2 + rendered_edges_y**2 + 1e-8)
    gt_edges = torch.sqrt(gt_edges_x**2 + gt_edges_y**2 + 1e-8)
    
    # 边缘误差
    edge_error = torch.abs(gt_edges - rendered_edges).squeeze()
    
    return edge_error


# 多尺度损失函数
def compute_multi_scale_loss(rendered, gt, scales=[1, 2, 4], weights=None):
    """
    多尺度损失，在不同分辨率下计算损失
    """
    if weights is None:
        weights = [2.0/s for s in scales]  # 默认权重：高分辨率权重更大
    
    total_loss = 0
    loss_dict = {}
    
    for scale, weight in zip(scales, weights):
        if scale == 1:
            scaled_rendered = rendered
            scaled_gt = gt
        else:
            # 下采样
            scaled_rendered = F.avg_pool2d(rendered, scale)
            scaled_gt = F.avg_pool2d(gt, scale)
        
        # 计算L1损失
        l1 = F.l1_loss(scaled_rendered, scaled_gt)
        
        # 计算梯度损失
        grad_loss = compute_gradient_loss(scaled_rendered, scaled_gt)
        
        scale_loss = l1 + 0.1 * grad_loss
        total_loss += weight * scale_loss
        
        loss_dict[f'scale_{scale}'] = scale_loss.item()
    
    return total_loss / sum(weights), loss_dict


def compute_gradient_loss(img1, img2):
    """
    梯度损失（保留原有函数，增强版本）
    """
    # 计算x方向梯度
    grad_x1 = img1[..., :, 1:] - img1[..., :, :-1]
    grad_x2 = img2[..., :, 1:] - img2[..., :, :-1]
    
    # 计算y方向梯度
    grad_y1 = img1[..., 1:, :] - img1[..., :-1, :]
    grad_y2 = img2[..., 1:, :] - img2[..., :-1, :]
    
    # 梯度差异
    grad_diff_x = torch.abs(grad_x1 - grad_x2)
    grad_diff_y = torch.abs(grad_y1 - grad_y2)
    
    return (grad_diff_x.mean() + grad_diff_y.mean()) * 0.5
