from __future__ import annotations
import os
import datetime
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim, gradient_loss
# Need F for conv2d
import torch.nn.functional as F 
import math
from gaussian_renderer import render, network_gui, render_fast_split
import sys
from scene import Scene
from scene.new_model import SplitGaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, apply_dog_filter
from utils.extra_utils import random_id
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import time  # 添加计时功能
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def compute_image_gradients(image_batch):
    """Computes image gradients (Sobel magnitude) for a batch of images."""
    # Ensure image is in float format
    image_batch = image_batch.float()
    # Convert to grayscale if it has 3 channels
    if image_batch.shape[1] == 3:
        # Standard RGB to grayscale conversion weights
        weights = torch.tensor([0.299, 0.587, 0.114], device=image_batch.device).view(1, 3, 1, 1)
        grayscale_batch = (image_batch * weights).sum(dim=1, keepdim=True)
    elif image_batch.shape[1] == 1:
        grayscale_batch = image_batch
    else:
        raise ValueError(f"Unsupported number of channels: {image_batch.shape[1]}")

    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=image_batch.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=image_batch.device).unsqueeze(0).unsqueeze(0)

    # Apply convolution
    grad_x = F.conv2d(grayscale_batch, sobel_x, padding=1)
    grad_y = F.conv2d(grayscale_batch, sobel_y, padding=1)
    
    # Return gradient magnitude
    gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8) # Add epsilon for stability
    return gradient_magnitude

def gradient(image):
    """计算图像梯度，用于SCS损失计算"""
    if len(image.shape) == 3:  # 如果输入是3D（C,H,W），添加批次维度
        image = image.unsqueeze(0)
    
    # 调用现有函数计算梯度
    grad_magnitude = compute_image_gradients(image)
    
    # 归一化梯度 (0-1范围)
    grad_normalized = grad_magnitude / (grad_magnitude.max() + 1e-8)
    
    # 移除批次维度，如果输入是3D
    if len(image.shape) == 4 and image.shape[0] == 1:
        grad_normalized = grad_normalized.squeeze(0)
    
    return grad_normalized

def compute_scs_loss(image, gt_image):
    """计算结构余弦相似度损失(SCS Loss)"""
    # 计算图像梯度
    grad_gt = gradient(gt_image)
    grad_image = gradient(image)
    
    # 计算低频结构图
    l_grad_gt = (1.0 - grad_gt) * gt_image
    l_grad_image = (1.0 - grad_image) * image
    
    # 计算余弦相似度
    num = (l_grad_gt * l_grad_image).sum(dim=(0, 1, 2))
    den = (torch.norm(l_grad_gt, p=2, dim=(0, 1, 2)) * torch.norm(l_grad_image, p=2, dim=(0, 1, 2)))
    cos = num / (den + 1e-8)
    
    # 返回损失值：1-cos
    return 1 - cos.mean()

def detect_high_frequency_regions(image, gt_image, threshold=0.3):
    """
    检测图像中的高频区域，用于指导分裂策略
    
    Args:
        image: 渲染图像 [3, H, W]
        gt_image: 真实图像 [3, H, W]  
        threshold: 高频阈值
        
    Returns:
        high_freq_mask: 高频区域掩码 [H, W]
        freq_score: 整体高频分数
    """
    # 转换为灰度
    if image.shape[0] == 3:
        image_gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gt_gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]
    else:
        image_gray = image[0]
        gt_gray = gt_image[0]
    
    # 计算图像梯度 (Sobel)
    grad_image = compute_image_gradients(image_gray.unsqueeze(0).unsqueeze(0)).squeeze()
    grad_gt = compute_image_gradients(gt_gray.unsqueeze(0).unsqueeze(0)).squeeze()
    
    # 计算梯度差异
    grad_diff = torch.abs(grad_image - grad_gt)
    
    # 高频区域标识
    high_freq_mask = grad_diff > threshold
    
    # 整体高频分数
    freq_score = grad_diff.mean().item()
    
    return high_freq_mask, freq_score

def compute_adaptive_split_budget(iteration, total_iterations, initial_gaussians, max_split_ratio=0.1):
    """
    计算自适应的分裂预算，避免过度分裂
    
    Args:
        iteration: 当前迭代
        total_iterations: 总迭代数
        initial_gaussians: 初始高斯球数量
        max_split_ratio: 最大分裂比例
        
    Returns:
        split_budget: 允许的分裂数量
    """
    # 训练进度 (0-1)
    progress = iteration / total_iterations
    
    # 分阶段分裂策略
    if progress < 0.3:  # 前30% - 保守分裂
        split_ratio = max_split_ratio * 0.3
    elif progress < 0.7:  # 30%-70% - 逐渐增加
        split_ratio = max_split_ratio * (0.3 + 0.4 * (progress - 0.3) / 0.4)
    else:  # 70%+ - 精细分裂
        split_ratio = max_split_ratio * 0.7
    
    split_budget = int(initial_gaussians * split_ratio)
    return max(split_budget, 1000)  # 最少保证1000个分裂

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args=None, log_to_wandb=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = SplitGaussianModel(dataset.sh_degree)  # 改为使用SplitGaussianModel
    
    # 获取下采样因子
    downsample_factor = getattr(args, 'downsample_factor', 1.0)
    scene = Scene(dataset, gaussians, downsample_factor=downsample_factor)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    # 如果没有传入args，创建一个带有默认值的Namespace对象
    if args is None:
        args = Namespace()  # 简化：统一使用快速渲染，无需额外参数

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 记录初始高斯球数量，用于计算分裂预算
    initial_gaussian_count = gaussians.get_xyz.shape[0]
    print(f"[训练初始化] 初始高斯球数量: {initial_gaussian_count:,}")

    # 添加一个强制初始化波向量的步骤，确保分裂效果
    # if args.force_init_wave and hasattr(gaussians, '_wave') and gaussians._wave is not None:
    #     print("\n[初始化] 强制初始化波向量以提高分裂效果")
    #     force_initialize_wave_vectors(scene, min_magnitude=args.wave_min_magnitude, max_magnitude=args.wave_max_magnitude)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # 为渲染时间和FPS测量添加变量
    render_times = []
    render_time_start = torch.cuda.Event(enable_timing = True)
    render_time_end = torch.cuda.Event(enable_timing = True)
    avg_render_time = 0.0
    avg_fps = 0.0
    render_time_window = 100  # 每100次迭代计算一次平均值

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1
    
    # 注释掉分阶段训练参数设置
    # 分阶段训练参数设置
    # total_iters = opt.iterations
    # 第一阶段：基础表示学习（70%的总训练时间）
    # stage1_end = int(total_iters * 0.7)
    # 第二阶段：波向量强化优化（20%的总训练时间）
    # stage2_end = int(total_iters * 0.9)
    # 第三阶段：整体精细调整（10%的总训练时间）
    
    # 存储原始学习率和权重，用于恢复
    # original_params = {
    #     "xyz_lr": None,
    #     "feature_lr": None,
    #     "rotation_lr": None,
    #     "scaling_lr": None,
    #     "opacity_lr": None,
    #     "wave_lr": opt.wave_lr if hasattr(opt, 'wave_lr') else 0.0,
    #     "lambda_jacobian_wave": opt.lambda_jacobian_wave if hasattr(opt, 'lambda_jacobian_wave') else 0.0,
    #     "lambda_wave_reg": opt.lambda_wave_reg if hasattr(opt, 'lambda_wave_reg') else 0.0
    # }
    
    # 查找优化器中的学习率参数
    # for group in gaussians.optimizer.param_groups:
    #     if group["name"] == "xyz":
    #         original_params["xyz_lr"] = group["lr"]
    #     elif group["name"] == "f_dc" or group["name"] == "f_rest":
    #         original_params["feature_lr"] = group["lr"]
    #     elif group["name"] == "rotation":
    #         original_params["rotation_lr"] = group["lr"]
    #     elif group["name"] == "scaling":
    #         original_params["scaling_lr"] = group["lr"]
    #     elif group["name"] == "opacity":
    #         original_params["opacity_lr"] = group["lr"]
    
    # 当前训练阶段
    # current_stage = 1
    
    # Get loss weights from opt once
    lambda_dssim = opt.lambda_dssim
    lambda_im_laplace = opt.lambda_im_laplace
    lambda_grad_diff = opt.lambda_grad_diff
    lambda_wave_reg = opt.lambda_wave_reg
    lambda_jacobian_wave = 0.0  # 禁用雅可比权重，因为我们不使用雅可比
    lambda_freq_wave = getattr(opt, 'lambda_freq_wave', 0.01)  # 新增：频率域wave损失权重
    # 将SCS损失权重设为0（等同于禁用）
    lambda_scs = 0.0  # 禁用SCS损失
    
    for iteration in range(first_iter, opt.iterations + 1):
        # 定期清理显存
        if iteration % 10 == 0:
            torch.cuda.empty_cache()
            
        # 每100次迭代检查显存使用情况
        if iteration % 100 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            print(f"\n[显存监控 ITER {iteration}] 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB")
            
        # 检查是否需要切换训练阶段
        # if iteration == stage1_end + 1 and current_stage == 1:
        #     current_stage = 2
        #     print(f"\n[训练阶段] 进入第二阶段：波向量强化优化 (迭代 {iteration}/{total_iters})")
            
        #     # 第二阶段参数调整：降低其他参数学习率，增强波向量学习率和雅各比权重
        #     for group in gaussians.optimizer.param_groups:
        #         if group["name"] == "xyz":
        #             group["lr"] = original_params["xyz_lr"] * 0.1  # 位置学习率降低为原来的10%
        #         elif group["name"] == "f_dc" or group["name"] == "f_rest":
        #             group["lr"] = original_params["feature_lr"] * 0.1  # 特征学习率降低为原来的10%
        #         elif group["name"] == "rotation":
        #             group["lr"] = original_params["rotation_lr"] * 0.1  # 旋转学习率降低为原来的10%
        #         elif group["name"] == "scaling":
        #             group["lr"] = original_params["scaling_lr"] * 0.1  # 缩放学习率降低为原来的10%
        #         elif group["name"] == "opacity":
        #             group["lr"] = original_params["opacity_lr"] * 0.1  # 不透明度学习率降低为原来的10%
        #         elif group["name"] == "wave":
        #             group["lr"] = original_params["wave_lr"] * 6.0  # 将波向量学习率提高从10倍改为6倍
            
        #     # 增强雅各比权重，降低波向量正则化
        #     lambda_jacobian_wave = original_params["lambda_jacobian_wave"] * 5.0  # 雅各比权重提高为原来的5倍
        #     lambda_wave_reg = original_params["lambda_wave_reg"] * 0.1  # 波向量正则化降低为原来的10%
            
        # elif iteration == stage2_end + 1 and current_stage == 2:
        #     current_stage = 3
        #     print(f"\n[训练阶段] 进入第三阶段：整体精细调整 (迭代 {iteration}/{total_iters})")
            
        #     # 第三阶段参数调整：恢复正常学习率，微调整体效果
        #     for group in gaussians.optimizer.param_groups:
        #         if group["name"] == "xyz":
        #             group["lr"] = original_params["xyz_lr"] * 0.5  # 位置学习率恢复为原来的50%
        #         elif group["name"] == "f_dc" or group["name"] == "f_rest":
        #             group["lr"] = original_params["feature_lr"] * 0.5  # 特征学习率恢复为原来的50%
        #         elif group["name"] == "rotation":
        #             group["lr"] = original_params["rotation_lr"] * 0.5  # 旋转学习率恢复为原来的50%
        #         elif group["name"] == "scaling":
        #             group["lr"] = original_params["scaling_lr"] * 0.5  # 缩放学习率恢复为原来的50%
        #         elif group["name"] == "opacity":
        #             group["lr"] = original_params["opacity_lr"] * 0.5  # 不透明度学习率恢复为原来的50%
        #         elif group["name"] == "wave":
        #             group["lr"] = original_params["wave_lr"] * 3.0  # 波向量学习率保持较高但低于第二阶段
            
        #     # 调整权重，但保持较高的雅各比权重
        #     lambda_jacobian_wave = original_params["lambda_jacobian_wave"] * 2.0  # 雅各比权重保持较高
        #     lambda_wave_reg = original_params["lambda_wave_reg"] * 0.5  # 波向量正则化适中
        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # 每1000次迭代增加SH的程度直到最大值
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择一个相机视角
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # 实现非线性波向量频率增长策略
        freq = (iteration / opt.iterations) * 100
        
        # 非线性频率曲线调整
        # 使用平滑的S型曲线，0-100的输出范围
        # 在早期增长较慢，中期加速，后期放缓
        if iteration < opt.iterations * 0.3:  # 前30%训练时间
            # 慢速启动阶段
            normalized_iter = (iteration / (opt.iterations * 0.3))
            freq = 20 * (normalized_iter ** 2)  # 二次曲线，最大值为20
        elif iteration < opt.iterations * 0.7:  # 30%-70%训练时间
            # 快速增长阶段
            normalized_iter = ((iteration - opt.iterations * 0.3) / (opt.iterations * 0.4))
            freq = 20 + 60 * normalized_iter  # 线性增长，从20到80
        else:  # 最后30%训练时间
            # 饱和阶段
            normalized_iter = ((iteration - opt.iterations * 0.7) / (opt.iterations * 0.3))
            freq = 80 + 20 * (1 - (1 - normalized_iter) ** 2)  # 逆二次曲线，从80到100
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
            
        # 测量渲染时间
        render_time_start.record()
        # 统一使用快速分裂渲染 - 高效且无雅可比计算开销
        
        # 计算当前迭代的分裂预算
        current_split_budget = compute_adaptive_split_budget(
            iteration, opt.iterations, initial_gaussian_count, 
            max_split_ratio=getattr(args, 'max_split_ratio', 0.05)  # 使用用户指定的分裂比例
        )
        
        # 检测高频区域（每10次迭代进行一次，减少计算开销）
        high_freq_mask = None
        if iteration % 10 == 0:
            try:
                # 先进行一次渲染获取当前图像（不分裂）
                quick_render = render(viewpoint_cam, gaussians, pipe, background, scaling_modifier=1.0)
                quick_image = quick_render["render"]
                gt_image_for_freq = viewpoint_cam.original_image.cuda()
                high_freq_mask, freq_score = detect_high_frequency_regions(quick_image, gt_image_for_freq, threshold=0.2)
                
                if iteration % 100 == 0:
                    print(f"[ITER {iteration}] 高频分数: {freq_score:.4f}, 分裂预算: {current_split_budget}")
            except Exception as e:
                if iteration % 100 == 0:
                    print(f"[警告] 高频区域检测失败: {e}")
                high_freq_mask = None
        
        # 执行渲染（禁用分裂，强制使用原始render函数）
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, scaling_modifier=1.0)
        render_time_end.record()
        
        # 等待CUDA操作完成
        render_time_end.synchronize()
        render_time_ms = render_time_start.elapsed_time(render_time_end)
        render_times.append(render_time_ms)
        
        # 计算FPS (1000ms / 渲染时间ms)
        current_fps = 1000.0 / max(render_time_ms, 1.0)  # 避免除以零
        
        # 保持最近render_time_window次迭代的平均值
        if len(render_times) > render_time_window:
            render_times.pop(0)
        avg_render_time = sum(render_times) / len(render_times)
        avg_fps = 1000.0 / max(avg_render_time, 1.0)
        
        # 每100次迭代输出一次渲染性能
        if iteration % 100 == 0:
            print(f"\n[ITER {iteration:05d}] 渲染性能: {avg_render_time:.2f}ms/帧 ({avg_fps:.2f} FPS)")

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # 获取高斯球统计（快速渲染模式）
        current_gaussian_stats = render_pkg.get("gaussian_stats")

        # 快速模式：不使用雅可比梯度，使用替代方案
        split_jacobians_info = None  # 明确设置为None

        # 获取真实图像并计算L1损失
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)

        # --- Frequency-based loss (DoG Mask) ---
        mask_loss = torch.tensor(0.0, device="cuda")
        # 注释DoG与拉普拉斯部分
        #if lambda_im_laplace > 0:
        #    mask = apply_dog_filter(image.unsqueeze(0), freq=freq, scale_factor=opt.im_laplace_scale_factor).squeeze(0)
        #    mask_loss = l1_loss(image * mask, gt_image * mask)
        
        # --- Image Gradient Difference Loss ---
        grad_loss = torch.tensor(0.0, device="cuda")
        # 恢复图像梯度差异损失
        if lambda_grad_diff > 0:
            # Add batch dimension for conv2d
            grad_image = compute_image_gradients(image.unsqueeze(0))
            grad_gt = compute_image_gradients(gt_image.unsqueeze(0))
            grad_loss = l1_loss(grad_image, grad_gt)
            
            # # 在强制分裂阶段，使用梯度信息来识别高频区域 (已注释掉)
            # if iteration < first_iter + 300:
            #     # 计算梯度图的大小，用于识别高频区域
            #     grad_magnitude = grad_gt.squeeze()
            #     # 识别高梯度区域（高频区域）
            #     high_freq_mask = grad_magnitude > 0.3
            #     if high_freq_mask.any():
            #         # 转换图像上的点到3D空间
            #         # 首先计算图像坐标和视点相机内外参之间的对应关系
            #         # 这里我们只需要标记哪些高斯点对应高频区域
            #         if isinstance(gaussians, SplitGaussianModel):
            #             # 使用可见性过滤器或投影关系来确定高频区域的3D点
            #             # 简化实现：增加visible高斯球的波向量
            #             if iteration % 100 == 0 or iteration == first_iter:  # 从50改为100，减少频率
            #                 print(f"[强制分裂] 正在强制高频区域的高斯球分裂，迭代{iteration}/{first_iter+300}")
            #             
            #             # 获取可见点的索引
            #             visible_indices = torch.where(visibility_filter)[0]
            #             
            #             # 大幅减少强制分裂的比例：从20%减少到5%
            #             num_to_force = int(visible_indices.shape[0] * 0.05)
            #             if num_to_force > 0:
            #                 force_indices = visible_indices[torch.randperm(visible_indices.shape[0])[:num_to_force]]
            #                 
            #                 # 将选定点的波向量增强
            #                 if hasattr(gaussians, '_wave') and gaussians._wave is not None:
            #                     # 分析现有波向量
            #                     current_wave = gaussians._wave.data[force_indices]
            #                     wave_norm = torch.norm(current_wave, dim=1, keepdim=True)
            #                     
            #                     # 对于波向量太小的点，增加其大小，但强度更小
            #                     small_wave_mask = wave_norm < 0.05  # 从0.1增加到0.05，更严格
            #                     if small_wave_mask.any():
            #                         # 为小波向量创建随机方向
            #                         random_dirs = torch.randn(small_wave_mask.sum().item(), 3, device="cuda")
            #                         random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)
            #                         
            #                         # 减小放大强度：从0.2减小到0.05
            #                         enhanced_small_waves = random_dirs * 0.05
            #                         
            #                         # 更新对应点的波向量
            #                         small_indices = force_indices[small_wave_mask.squeeze()]
            #                         gaussians._wave.data[small_indices] = enhanced_small_waves
            #                         
            #                         if iteration % 100 == 0:  # 从50改为100
            #                             print(f"[强制分裂] 增强了{small_indices.shape[0]}个高频区域点的波向量")

        # --- Wave Regularization Loss ---
        wave_reg_loss = torch.tensor(0.0, device="cuda")
        if lambda_wave_reg > 0:
             # L2 regularization on wave norm (penalize large wave vectors)
             wave_reg_loss = torch.mean(torch.sum(gaussians._wave**2, dim=1)) 
        
        # --- Frequency-based Wave Learning Loss ---
        freq_wave_loss = torch.tensor(0.0, device="cuda")
        if lambda_freq_wave > 0:
            freq_wave_loss = compute_frequency_wave_loss(image, gt_image, gaussians, visibility_filter)
             
        # --- Combined Loss ---
        loss = Ll1
        if lambda_dssim > 0:
            loss += lambda_dssim * (1.0 - ssim(image, gt_image))
        # 恢复图像梯度差异损失
        if lambda_grad_diff > 0:
            loss += lambda_grad_diff * grad_loss
        if lambda_wave_reg > 0:
            loss += lambda_wave_reg * wave_reg_loss
        if lambda_freq_wave > 0:
            loss += lambda_freq_wave * freq_wave_loss
        # 注释结构余弦相似度损失
        #if lambda_scs > 0:
        #    loss += lambda_scs * scs_loss
               
        loss.backward() # Compute gradients for all terms EXCEPT jacobian term

        iter_end.record()

        # 密度统计（统一处理，无需区分快速或完整模式）
        if iteration < opt.densify_until_iter:
            # 快速渲染模式：统一处理所有点的密度统计
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, use_fast_render=True)

        with torch.no_grad():
            # --- Jacobian-based Wave Gradient Contribution --- 
            if split_jacobians_info is not None: 
                # Check if gradients exist for viewspace points (should exist after loss.backward)
                # 安全检查：确保张量有梯度且不会产生警告
                viewspace_has_grad = False
                try:
                    # 尝试访问梯度，如果是非叶子张量且没有retain_grad，这会返回None
                    if hasattr(viewspace_point_tensor, 'grad') and viewspace_point_tensor.grad is not None:
                        viewspace_has_grad = True
                except Exception as e:
                    print(f"[梯度检查异常 ITER {iteration}] {e}")
                    viewspace_has_grad = False
                
                if viewspace_has_grad:
                    # Calculate accumulated gradient contribution for wave
                    accumulated_wave_grad = torch.zeros_like(gaussians._wave.data)
                    
                    screen_space_grads = viewspace_point_tensor.grad # Shape [N_render, 3]
                    
                    # Iterate through the recorded splits and their Jacobians
                    original_indices = split_jacobians_info["original_indices"]
                    split_indices_pos = split_jacobians_info["split_indices_pos"]
                    split_indices_neg = split_jacobians_info["split_indices_neg"]
                    jacobians_pos = split_jacobians_info["jacobians_pos"]
                    jacobians_neg = split_jacobians_info["jacobians_neg"]
                    
                    # 添加调试输出
                    if iteration % 100 == 0:
                        print(f"\n[DEBUG 雅各比梯度 ITER {iteration}] "
                              f"视点梯度存在: {viewspace_has_grad}, "
                              f"记录的分裂点数: {len(original_indices)}")
                        
                        # 打印当前参数设置
                        print(f"[参数设置] 波向量学习率: {gaussians.optimizer.param_groups[-1]['lr']:.6f}, "
                              f"雅各比权重: {lambda_jacobian_wave:.6f}, "
                              f"波向量正则化: {lambda_wave_reg:.6f}")
                        
                        # 检查屏幕空间梯度
                        if screen_space_grads is not None:
                            ss_grad_norm = torch.norm(screen_space_grads, dim=1)
                            print(f"屏幕空间梯度统计: 最大值={ss_grad_norm.max().item():.6f}, "
                                  f"平均值={ss_grad_norm.mean().item():.6f}, "
                                  f"非零比例={((ss_grad_norm > 1e-10).sum().item() / ss_grad_norm.shape[0]):.2%}")
                    
                    # Ensure Jacobians are tensors
                    # Check if lists are empty before trying to stack
                    if len(jacobians_pos) > 0 and len(jacobians_neg) > 0:
                        jacobians_pos_tensor = torch.stack(jacobians_pos) # [N_splits, 3, 3]
                        jacobians_neg_tensor = torch.stack(jacobians_neg) # [N_splits, 3, 3]
                        original_indices_tensor = torch.tensor(original_indices, device=accumulated_wave_grad.device) # [N_splits]
                        split_indices_pos_tensor = torch.tensor(split_indices_pos, device=accumulated_wave_grad.device) # [N_splits]
                        split_indices_neg_tensor = torch.tensor(split_indices_neg, device=accumulated_wave_grad.device) # [N_splits]

                        # Get grads for relevant split points, handle potential OOB
                        max_idx = max(split_indices_pos_tensor.max(), split_indices_neg_tensor.max()) if len(original_indices)>0 else -1
                        valid_pos_mask = split_indices_pos_tensor < screen_space_grads.shape[0]
                        valid_neg_mask = split_indices_neg_tensor < screen_space_grads.shape[0]

                        # Calculate contributions only for valid indices
                        valid_orig_idx_pos = original_indices_tensor[valid_pos_mask]
                        valid_split_idx_pos = split_indices_pos_tensor[valid_pos_mask]
                        valid_jac_pos = jacobians_pos_tensor[valid_pos_mask]
                        grads_pos = screen_space_grads[valid_split_idx_pos, :3] # [N_valid_pos, 3]
                        contrib_pos = torch.bmm(valid_jac_pos.transpose(1, 2), grads_pos.unsqueeze(2)).squeeze(2) # [N_valid_pos, 3]
                        
                        valid_orig_idx_neg = original_indices_tensor[valid_neg_mask]
                        valid_split_idx_neg = split_indices_neg_tensor[valid_neg_mask]
                        valid_jac_neg = jacobians_neg_tensor[valid_neg_mask]
                        grads_neg = screen_space_grads[valid_split_idx_neg, :3] # [N_valid_neg, 3]
                        contrib_neg = torch.bmm(valid_jac_neg.transpose(1, 2), grads_neg.unsqueeze(2)).squeeze(2) # [N_valid_neg, 3]
                        
                        # Use index_add_ to accumulate contributions
                        accumulated_wave_grad.index_add_(0, valid_orig_idx_pos, contrib_pos)
                        accumulated_wave_grad.index_add_(0, valid_orig_idx_neg, contrib_neg)

                        # Add the accumulated Jacobian gradient contribution to the existing wave gradient
                        if gaussians._wave.grad is None:
                            gaussians._wave.grad = torch.zeros_like(gaussians._wave.data)
                        # Scale by lambda and add
                        gaussians._wave.grad.add_(accumulated_wave_grad * lambda_jacobian_wave)
                        
                        # 添加波向量梯度监控
                        if iteration % 100 == 0:
                            wave_norm = torch.norm(gaussians._wave.data, dim=1)
                            if gaussians._wave.grad is not None:
                                wave_grad_norm = torch.norm(gaussians._wave.grad.data, dim=1)
                                print(f"[波向量梯度统计] 波向量梯度最大值={wave_grad_norm.max().item():.6f}, "
                                      f"平均值={wave_grad_norm.mean().item():.6f}, "
                                      f"非零比例={((wave_grad_norm > 1e-6).sum().item() / wave_grad_norm.shape[0]):.2%}")
                                print(f"[波向量统计] 波向量最大值={wave_norm.max().item():.6f}, "
                                      f"平均值={wave_norm.mean().item():.6f}, "
                                      f"活跃Wave向量(>0.001)比例={((wave_norm > 0.001).sum().item() / wave_norm.shape[0]):.2%}")
                    # else: No jacobians were computed/returned, skip gradient addition
                        
                else:
                    if iteration % 100 == 0:  # 减少警告频率
                        print(f"[Warning Iter {iteration}] viewspace_point_tensor梯度不可用，无法应用雅各比波向量梯度。")
                    # 尝试在渲染张量上调用retain_grad以修复未来的迭代
                    try:
                        viewspace_point_tensor.retain_grad()
                    except:
                        pass

            # Progress bar update
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
                
            # 每100次迭代输出一次等效高斯球数量
            if iteration % 100 == 0 and current_gaussian_stats:
                stats = current_gaussian_stats
                total_gaussians = stats['n_original'] + stats['n_split']
                effective_gaussians = stats.get('effective_render_count', total_gaussians)
                print(f"\n[ITER {iteration:05d}] 等效高斯球: {effective_gaussians:,} (原始: {stats['n_original']:,}, 分裂: {stats['n_split']:,}, 比例: {stats['split_ratio']:.2f}倍)")
                print(f"渲染性能: {avg_render_time:.2f}ms/帧 ({avg_fps:.2f} FPS)")
                
                # 输出频率域wave损失信息
                if lambda_freq_wave > 0:
                    print(f"频率域Wave损失: {freq_wave_loss.item():.6f}")
                
                if "active_waves" in stats:
                    active_waves = stats["active_waves"]
                    print(f"活跃波向量(>0.01): {active_waves:,} ({100*active_waves/stats['n_original']:.1f}% 的高斯球)")
                    
                # 显示不同阈值的活跃波向量
                if "active_waves_001" in stats:
                    print(f"轻微活跃波向量(>0.001): {stats['active_waves_001']:,} ({100*stats['active_waves_001']/stats['n_original']:.1f}%)")
                if "active_waves_05" in stats:
                    print(f"高度活跃波向量(>0.05): {stats['active_waves_05']:,} ({100*stats['active_waves_05']/stats['n_original']:.1f}%)")
                
                # 显示分裂统计
                if "all_splitting_gaussians" in stats:
                    print(f"产生分裂的高斯球: {stats['all_splitting_gaussians']:,} ({100*stats['all_splitting_gaussians']/stats['n_original']:.1f}%)")
                if "avg_splits_all" in stats and stats["avg_splits_all"] > 0:
                    print(f"每个分裂高斯平均分裂数: {stats['avg_splits_all']:.2f}")
                if "max_splits_per_gaussian" in stats:
                    print(f"单个高斯最多分裂数: {stats['max_splits_per_gaussian']}")
                    
            # 波向量统计（新的简化版本）
            if iteration % 100 == 0 and hasattr(gaussians, '_wave') and gaussians._wave is not None:
                wave_norm = torch.norm(gaussians._wave.data, dim=1)
                if gaussians._wave.grad is not None:
                    wave_grad_norm = torch.norm(gaussians._wave.grad.data, dim=1)
                    print(f"[无雅可比Wave统计 ITER {iteration}] Wave梯度最大值={wave_grad_norm.max().item():.6f}, "
                          f"平均值={wave_grad_norm.mean().item():.6f}")
                print(f"Wave向量最大值={wave_norm.max().item():.6f}, "
                      f"平均值={wave_norm.mean().item():.6f}, "
                      f"活跃Wave向量(>0.001)比例={((wave_norm > 0.001).sum().item() / wave_norm.shape[0]):.2%}")

            # Log and save - 统一使用快速分裂渲染进行评估
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, mask_loss, grad_loss, wave_reg_loss, 
                           iter_start.elapsed_time(iter_end), testing_iterations, scene, render_fast_split, (pipe, background), 
                           scs_loss=None, render_time=avg_render_time, fps=avg_fps, 
                           gaussian_stats=current_gaussian_stats, freq_wave_loss=freq_wave_loss)
            
            if (iteration in saving_iterations):
                print(f"\n[ITER {iteration}] Saving Gaussians")
                scene.save(iteration)

            # 简化的密集化处理
            if iteration < opt.densify_until_iter:
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, size_threshold)
                
                # Reset opacity at specified intervals
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步进
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] 保存检查点".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        # Ensure output path is relative to the project directory
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("输出文件夹: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 只使用 Tensorboard，完全移除 wandb
    tb_writer = None
    if TENSORBOARD_FOUND:
        # Log directory should be the model path
        try: # Try-catch for SummaryWriter, path might not exist if dry run or unusual setup
            tb_writer = SummaryWriter(args.model_path)
        except Exception as e:
            print(f"Error creating SummaryWriter at {args.model_path}: {e}")
            print("Tensorboard logging will be disabled.")
            pass # tb_writer remains None
    else:
        print("Tensorboard不可用：不记录进度")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, mask_loss, grad_loss, wave_reg_loss, elapsed, 
                   testing_iterations, scene : Scene, renderFunc, renderArgs, scs_loss=None, render_time=0.0, fps=0.0,
                   gaussian_stats=None, freq_wave_loss=0.0): # Added gaussian_stats and freq_wave_loss
    if tb_writer:
        # Log main losses
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        # Log specific loss components if their weights > 0
        if mask_loss.item() > 0:
            tb_writer.add_scalar('train_loss_patches/mask_loss', mask_loss.item(), iteration)
        if grad_loss.item() > 0:
             tb_writer.add_scalar('train_loss_patches/grad_loss', grad_loss.item(), iteration)
        if wave_reg_loss.item() > 0:
             tb_writer.add_scalar('train_loss_patches/wave_reg_loss', wave_reg_loss.item(), iteration)
        if freq_wave_loss > 0:
             tb_writer.add_scalar('train_loss_patches/freq_wave_loss', freq_wave_loss, iteration)
        if scs_loss is not None and scs_loss.item() > 0:
             tb_writer.add_scalar('train_loss_patches/scs_loss', scs_loss.item(), iteration)
             
        tb_writer.add_scalar('iter_time', elapsed, iteration)

        # Log parameter statistics
        gaussians = scene.gaussians
        tb_writer.add_scalar('params/xyz_norm', gaussians._xyz.data.norm(), iteration)
        tb_writer.add_scalar('params/scaling_mean', gaussians.get_scaling.mean(), iteration)
        tb_writer.add_scalar('params/rotation_norm', gaussians._rotation.data.norm(), iteration)
        tb_writer.add_scalar('params/opacity_mean', gaussians.get_opacity.mean(), iteration)
        
        # Log shape params if model has them
        if hasattr(gaussians, '_shape') and gaussians._shape.numel() > 0:
            tb_writer.add_scalar('params/shape_mean_abs', gaussians._shape.data.abs().mean(), iteration)
            tb_writer.add_scalar('params/shape_norm', gaussians._shape.data.norm(), iteration)
            if gaussians._shape.grad is not None:
                 tb_writer.add_scalar('params/shape_grad_norm', gaussians._shape.grad.data.norm(), iteration)

        if hasattr(gaussians, '_wave') and gaussians._wave.numel() > 0:
            tb_writer.add_scalar('params/wave_norm', gaussians._wave.data.norm(), iteration)
            tb_writer.add_scalar('params/wave_mean_abs', gaussians._wave.data.abs().mean(), iteration)
            
            # 添加更详细的wave统计数据
            wave_norms = torch.norm(gaussians._wave.data, dim=1)
            tb_writer.add_scalar('params/wave_min_norm', wave_norms.min().item(), iteration)
            tb_writer.add_scalar('params/wave_max_norm', wave_norms.max().item(), iteration)
            # 统计接近零的wave数量 (小于等于不同阈值)
            near_zero_wave_1e3 = (wave_norms <= 1e-3).sum().item()
            near_zero_wave_1e4 = (wave_norms <= 1e-4).sum().item()
            near_zero_wave_1e5 = (wave_norms <= 1e-5).sum().item()
            tb_writer.add_scalar('params/near_zero_wave_1e3', near_zero_wave_1e3, iteration)
            tb_writer.add_scalar('params/near_zero_wave_1e4', near_zero_wave_1e4, iteration)
            tb_writer.add_scalar('params/near_zero_wave_1e5', near_zero_wave_1e5, iteration)
            
            # 波向量梯度统计
            if gaussians._wave.grad is not None:
                wave_grad_norms = torch.norm(gaussians._wave.grad.data, dim=1)
                tb_writer.add_scalar('params/wave_grad_mean', wave_grad_norms.mean().item(), iteration)
                tb_writer.add_scalar('params/wave_grad_max', wave_grad_norms.max().item(), iteration)
                # 记录梯度为零的波向量数量
                zero_grad_waves = (wave_grad_norms <= 1e-10).sum().item()
                tb_writer.add_scalar('params/zero_grad_waves', zero_grad_waves, iteration)
                
        # Log feature norms
        tb_writer.add_scalar('params/f_dc_norm', gaussians._features_dc.data.norm(), iteration)
        tb_writer.add_scalar('params/f_rest_norm', gaussians._features_rest.data.norm(), iteration)

        tb_writer.add_scalar('scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # 记录等效高斯球数量（如果可用）
        # render_pkg = renderFunc(scene.getTrainCameras()[0], gaussians, *renderArgs) # This can be slow
        # Instead, use passed gaussian_stats if available
        if gaussian_stats:
            stats = gaussian_stats
            total_gaussians_stat = stats['n_original'] + stats['n_split']
            effective_gaussians_stat = stats.get('effective_render_count', total_gaussians_stat)
            
            tb_writer.add_scalar('scene/effective_gaussians', effective_gaussians_stat, iteration)
            tb_writer.add_scalar('scene/original_gaussians', stats['n_original'], iteration) # Log n_original
            tb_writer.add_scalar('scene/split_gaussians', stats['n_split'], iteration)
            tb_writer.add_scalar('scene/split_ratio', stats['split_ratio'], iteration)
            
            # 记录更详细的波向量活跃度统计
            if "active_waves" in stats:
                active_waves = stats["active_waves"]
                active_ratio = active_waves / stats['n_original'] if stats['n_original'] > 0 else 0
                tb_writer.add_scalar('scene/active_waves', active_waves, iteration)
                tb_writer.add_scalar('scene/active_waves_ratio', active_ratio, iteration)
                
                # 记录不同阈值的活跃波向量
                if "active_waves_001" in stats:
                    tb_writer.add_scalar('scene/active_waves_001', stats["active_waves_001"], iteration)
                    tb_writer.add_scalar('scene/active_waves_001_ratio', 
                                        stats["active_waves_001"]/stats['n_original'] if stats['n_original'] > 0 else 0, 
                                        iteration)
                if "active_waves_05" in stats:
                    tb_writer.add_scalar('scene/active_waves_05', stats["active_waves_05"], iteration)
                    tb_writer.add_scalar('scene/active_waves_05_ratio', 
                                        stats["active_waves_05"]/stats['n_original'] if stats['n_original'] > 0 else 0, 
                                        iteration)
            
            # 分裂统计 (from gaussian_stats if available)
            if "max_splits_per_gaussian" in stats:
                tb_writer.add_scalar('scene/max_splits_per_gaussian', stats['max_splits_per_gaussian'], iteration)
            if "all_splitting_gaussians" in stats:
                tb_writer.add_scalar('scene/all_splitting_gaussians', stats['all_splitting_gaussians'], iteration)
                tb_writer.add_scalar('scene/all_splitting_ratio', 
                                    stats['all_splitting_gaussians']/stats['n_original'] if stats['n_original'] > 0 else 0, 
                                    iteration)
            if "avg_splits_all" in stats:
                tb_writer.add_scalar('scene/avg_splits_all', stats['avg_splits_all'], iteration)
        
        # Keep existing logging for shape, opacity, wave etc. (using direct model access)
        # This part is somewhat redundant if already covered by gaussian_stats and params/ above
        # but provides histogram and specific grad norms.
        if hasattr(scene.gaussians, 'get_shape') and scene.gaussians._shape.numel() > 0:  
            tb_writer.add_scalar('scene/average_shape_val', scene.gaussians.get_shape.mean().item(), iteration) # from get_shape
            tb_writer.add_histogram("scene/shape_param_hist", scene.gaussians._shape.data, iteration) # raw _shape param
            
            # The following were from train_ges.py, may or may not be useful for SplitLaplacian
            # tb_writer.add_scalar('scene/small_points', (scene.gaussians.get_shape < 0.5).sum().item(), iteration)
            # if (scene.gaussians.get_shape >= 1.0).sum().item() > 0:
            #     tb_writer.add_scalar('scene/large_shapes', scene.gaussians.get_shape[scene.gaussians.get_shape >= 1.0].mean().item(), iteration)
            # if (scene.gaussians.get_shape < 1.0).sum().item() > 0:
            #     tb_writer.add_scalar('scene/small_shapes', scene.gaussians.get_shape[scene.gaussians.get_shape < 1.0].mean().item(), iteration)
            
            # Redundant if already logged under params/
            # if scene.gaussians._shape.grad is not None:
            #     tb_writer.add_scalar('scene/shape_grads', scene.gaussians._shape.grad.data.norm(2).item(), iteration)
        
        if scene.gaussians._opacity.grad is not None:
            tb_writer.add_scalar('scene/opacity_grads', scene.gaussians._opacity.grad.data.norm(2).item(), iteration)
        
        # Add splitting specific metrics if enabled
        if hasattr(scene.gaussians, 'use_splitting') and scene.gaussians.use_splitting:
            # Log mean wave norm
            wave_norms = scene.gaussians.get_wave.norm(dim=1)
            if wave_norms.numel() > 0:
                 tb_writer.add_scalar('scene/wave_norm_mean', wave_norms.mean().item(), iteration)
                 tb_writer.add_scalar('scene/wave_norm_max', wave_norms.max().item(), iteration)
            # Log wave gradient norm 
            if scene.gaussians._wave.grad is not None:
                tb_writer.add_scalar('scene/wave_grads_norm', scene.gaussians._wave.grad.data.norm(2).item(), iteration)
            # Log number of active splitting points
            if hasattr(scene.gaussians, '_wave'):
                 active_splitters = (scene.gaussians._wave.norm(dim=1) > 1e-6).sum().item()
                 tb_writer.add_scalar('scene/active_splitters', active_splitters, iteration)

        # 添加形状统计信息（如果有）
        if hasattr(scene.gaussians, 'get_shape_aware_stats'):
            shape_stats = scene.gaussians.get_shape_aware_stats()
            if 'avg_shape_factor' in shape_stats:
                tb_writer.add_scalar('scene/shape_factor_mean', shape_stats['avg_shape_factor'], iteration)
            if 'std_shape_factor' in shape_stats:
                tb_writer.add_scalar('scene/shape_factor_std', shape_stats['std_shape_factor'], iteration)
            if 'nonuniform_ratio' in shape_stats:
                tb_writer.add_scalar('scene/nonuniform_ratio', shape_stats['nonuniform_ratio'], iteration)
                
        # 添加分裂统计信息（如果有）
        if hasattr(scene.gaussians, 'get_splitting_stats'):
            split_stats = scene.gaussians.get_splitting_stats()
            if 'wave_norm_mean' in split_stats:
                tb_writer.add_scalar('scene/wave_norm_mean', split_stats['wave_norm_mean'], iteration)
            if 'wave_norm_std' in split_stats:
                tb_writer.add_scalar('scene/wave_norm_std', split_stats['wave_norm_std'], iteration)
            if 'active_splitters' in split_stats:
                tb_writer.add_scalar('scene/active_splitters', split_stats['active_splitters'], iteration)

        # 记录渲染性能指标
        if render_time > 0:
            tb_writer.add_scalar('performance/render_time_ms', render_time, iteration)
            tb_writer.add_scalar('performance/fps', fps, iteration)

        # 测试集和训练集采样评估
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                 {'name': 'train', 'cameras' : scene.getTrainCameras()})
            
            # 评估集渲染性能测量
            eval_render_times = []
            
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0
                    psnr_test = 0
                    # 添加高斯球统计变量
                    total_original = 0
                    total_split = 0
                    total_active_waves = 0
                    max_splits = 0
                    
                    for idx, viewpoint in enumerate(config['cameras']):
                        # 测量评估渲染时间
                        eval_start = torch.cuda.Event(enable_timing=True)
                        eval_end = torch.cuda.Event(enable_timing=True)
                        
                        eval_start.record()
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        eval_end.record()
                        eval_end.synchronize()
                        
                        eval_render_times.append(eval_start.elapsed_time(eval_end))
                        
                        image = render_pkg["render"]
                        
                        # 收集高斯球统计信息
                        if "gaussian_stats" in render_pkg:
                            stats = render_pkg["gaussian_stats"]
                            total_original += stats["n_original"]
                            total_split += stats["n_split"]
                            total_active_waves += stats.get("active_waves", 0)
                            current_max_splits = stats.get("max_splits_per_gaussian", 0)
                            max_splits = max(max_splits, current_max_splits)
                        
                        gt_image = viewpoint.original_image.cuda()
                        if gt_image.shape[0] == 4:
                            gt_image = gt_image[:3]
                        l1_test += l1_loss(image, gt_image).cpu().numpy()
                        psnr_test += psnr(image, gt_image).cpu().numpy()
                    
                    # 计算平均高斯球统计信息
                    n_views = len(config['cameras'])
                    avg_original = total_original / n_views
                    avg_split = total_split / n_views
                    avg_total = avg_original + avg_split
                    avg_active_waves = total_active_waves / n_views
                    avg_ratio = avg_total / avg_original if avg_original > 0 else 1.0
                    
                    l1_test /= n_views
                    psnr_test /= n_views
                    # 修复: 确保是标量值而不是numpy数组
                    try:
                        # 优先尝试作为标量处理
                        if isinstance(l1_test, (int, float)):
                            l1_test_val = l1_test
                        elif hasattr(l1_test, 'size') and l1_test.size == 1:
                            # NumPy数组且大小为1
                            l1_test_val = l1_test.item()
                        else:
                            # 其他情况取第一个元素
                            l1_test_val = float(l1_test.flatten()[0]) if hasattr(l1_test, 'flatten') and l1_test.size > 0 else 0.0
                        
                        if isinstance(psnr_test, (int, float)):
                            psnr_test_val = psnr_test
                        elif hasattr(psnr_test, 'size') and psnr_test.size == 1:
                            # NumPy数组且大小为1
                            psnr_test_val = psnr_test.item()
                        else:
                            # 其他情况取第一个元素
                            psnr_test_val = float(psnr_test.flatten()[0]) if hasattr(psnr_test, 'flatten') and psnr_test.size > 0 else 0.0
                    except Exception as e:
                        print(f"警告: 无法转换验证指标为标量: {e}")
                        l1_test_val = 0.0
                        psnr_test_val = 0.0
                    
                    # 计算评估渲染性能
                    if eval_render_times:
                        eval_avg_time = sum(eval_render_times) / len(eval_render_times)
                        eval_avg_fps = 1000.0 / max(eval_avg_time, 1.0)
                        
                        # 打印扩展的评估信息，包括渲染性能    
                        print("\n[ITER {}] 验证 {}: L1 = {:.6f}, PSNR = {:.3f}".format(iteration, config['name'], l1_test_val, psnr_test_val))
                        print(f"等效高斯球平均数量: {avg_total:.0f} (原始: {avg_original:.0f}, 分裂: {avg_split:.0f}, 比例: {avg_ratio:.2f}倍)")
                        print(f"[验证渲染性能] {eval_avg_time:.2f}ms/帧 ({eval_avg_fps:.2f} FPS)")
                        
                        if avg_active_waves > 0 and avg_original > 0:
                            print(f"活跃波向量平均数量: {avg_active_waves:.0f} ({100*avg_active_waves/avg_original:.1f}% 的高斯球有活跃波向量)")
                        if max_splits > 0:
                            print(f"单个高斯最多分裂数: {max_splits}")
                    
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test_val, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test_val, iteration)
                    # 记录高斯球统计信息
                    tb_writer.add_scalar(config['name'] + '/gaussian_stats - original', avg_original, iteration)
                    tb_writer.add_scalar(config['name'] + '/gaussian_stats - split', avg_split, iteration)
                    tb_writer.add_scalar(config['name'] + '/gaussian_stats - total', avg_total, iteration)
                    tb_writer.add_scalar(config['name'] + '/gaussian_stats - ratio', avg_ratio, iteration)
                    if avg_active_waves > 0 and avg_original > 0:
                        tb_writer.add_scalar(config['name'] + '/gaussian_stats - active_waves', avg_active_waves, iteration)
                        tb_writer.add_scalar(config['name'] + '/gaussian_stats - active_ratio', avg_active_waves/avg_original, iteration)
                    if max_splits > 0:
                        tb_writer.add_scalar(config['name'] + '/gaussian_stats - max_splits', max_splits, iteration)
                    
                    # 记录渲染性能数据
                    if eval_render_times:
                        tb_writer.add_scalar(config['name'] + '/performance - render_time_ms', eval_avg_time, iteration)
                        tb_writer.add_scalar(config['name'] + '/performance - fps', eval_avg_fps, iteration)

def force_initialize_wave_vectors(scene, min_magnitude=0.01, max_magnitude=0.05):
    """
    强制初始化所有高斯球的波向量，确保它们有足够的大小以产生有效分裂。
    
    Args:
        scene: 场景对象，包含高斯球模型
        min_magnitude: 最小波向量大小
        max_magnitude: 最大波向量大小
    """
    gaussians = scene.gaussians
    
    if not hasattr(gaussians, '_wave'):
        print("高斯模型没有wave属性，无法初始化波向量")
        return False
        
    # 获取当前波向量
    waves = gaussians._wave.data
    n_points = waves.shape[0]
    
    # 计算当前向量的范数
    wave_norms = torch.norm(waves, dim=1)
    
    # 统计当前波向量状态
    small_waves = (wave_norms < min_magnitude).sum().item()
    print(f"当前波向量状态: 总数={n_points}, 过小波向量(<{min_magnitude})={small_waves} ({small_waves/n_points:.1%})")
    
    # 对于每个波向量，如果范数太小，则放大到所需范围
    needs_update = wave_norms < min_magnitude
    n_update = needs_update.sum().item()
    
    if n_update > 0:
        # 对于小的波向量，保持方向但调整大小
        unit_vectors = waves.clone()
        
        # 对于接近零的波向量(无法归一化)，创建随机方向
        very_small = wave_norms < 1e-6
        n_very_small = very_small.sum().item()
        
        # 对非零波向量，保持方向只调整大小
        normalize_mask = (~very_small) & needs_update
        if normalize_mask.sum() > 0:
            unit_vectors[normalize_mask] = unit_vectors[normalize_mask] / wave_norms[normalize_mask].unsqueeze(1)
        
        # 对接近零的波向量，创建随机方向
        if n_very_small > 0:
            random_dirs = torch.randn(n_very_small, 3, device=waves.device)
            random_dirs = random_dirs / torch.norm(random_dirs, dim=1, keepdim=True)
            unit_vectors[very_small] = random_dirs
        
        # 为每个需要更新的波向量创建一个随机幅度
        if min_magnitude < max_magnitude:
            rand_magnitudes = min_magnitude + torch.rand(n_update, device=waves.device) * (max_magnitude - min_magnitude)
        else:
            rand_magnitudes = torch.ones(n_update, device=waves.device) * min_magnitude
        
        # 应用新的波向量
        waves[needs_update] = unit_vectors[needs_update] * rand_magnitudes.unsqueeze(1)
        
        # 确保requires_grad仍然为True
        gaussians._wave.requires_grad_(True)
        
        # 获取更新后的统计信息
        wave_norms_after = torch.norm(waves, dim=1)
        
        print(f"波向量初始化完成: 已更新{n_update}个波向量 ({n_update/n_points:.1%})")
        print(f"更新后状态: 平均范数={wave_norms_after.mean().item():.4f}, 最小范数={wave_norms_after.min().item():.4f}")
        
        # 统计波向量方向的分布
        if n_update > 1000:
            # 随机抽样显示方向分布
            sample_size = min(1000, n_update)
            sample_indices = torch.where(needs_update)[0][torch.randperm(n_update)[:sample_size]]
            sample_vecs = waves[sample_indices]
            sample_norms = torch.norm(sample_vecs, dim=1)
            print(f"波向量方向样本统计 (随机{sample_size}个点):")
            print(f"  X方向均值: {sample_vecs[:, 0].mean().item():.4f}, 标准差: {sample_vecs[:, 0].std().item():.4f}")
            print(f"  Y方向均值: {sample_vecs[:, 1].mean().item():.4f}, 标准差: {sample_vecs[:, 1].std().item():.4f}")
            print(f"  Z方向均值: {sample_vecs[:, 2].mean().item():.4f}, 标准差: {sample_vecs[:, 2].std().item():.4f}")
        
        return True
    
    print("所有波向量都已经有足够大小，无需更新")
    return False

def compute_frequency_wave_loss(image, gt_image, gaussians, visibility_filter):
    """
    计算基于频率域的wave损失，鼓励wave向量在需要高频细节的区域更活跃
    """
    # 转换为灰度图像进行频率分析
    if image.shape[0] == 3:
        image_gray = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
        gt_gray = 0.299 * gt_image[0] + 0.587 * gt_image[1] + 0.114 * gt_image[2]
    else:
        image_gray = image[0]
        gt_gray = gt_image[0]
    
    # 计算频率域差异
    image_fft = torch.fft.fft2(image_gray)
    gt_fft = torch.fft.fft2(gt_gray)
    freq_magnitude_diff = torch.abs(torch.abs(image_fft) - torch.abs(gt_fft))
    
    # 识别高频区域 (去除DC分量)
    h, w = freq_magnitude_diff.shape
    center_h, center_w = h // 2, w // 2
    
    # 创建高频掩码 (距离中心较远的频率)
    y_coords, x_coords = torch.meshgrid(torch.arange(h, device=image.device), 
                                        torch.arange(w, device=image.device), indexing='ij')
    freq_distance = torch.sqrt((y_coords - center_h)**2 + (x_coords - center_w)**2)
    high_freq_mask = freq_distance > min(h, w) * 0.1  # 高于10%频率半径的区域
    
    # 计算高频区域的差异
    high_freq_error = (freq_magnitude_diff * high_freq_mask).mean()
    
    # 获取可见高斯球的wave向量范数
    if hasattr(gaussians, '_wave') and gaussians._wave is not None:
        visible_wave_norms = torch.norm(gaussians._wave[visibility_filter], dim=1)
        
        # 鼓励在高频误差大时有更活跃的wave向量
        # 如果高频误差大，wave向量应该更活跃；如果高频误差小，可以更平稳
        target_wave_activity = torch.clamp(high_freq_error * 10.0, 0.001, 0.1)  # 动态目标
        
        # 计算wave活跃度与目标之间的差异
        current_wave_activity = visible_wave_norms.mean() if len(visible_wave_norms) > 0 else torch.tensor(0.0, device=image.device)
        wave_activity_loss = torch.abs(current_wave_activity - target_wave_activity)
        
        return wave_activity_loss
    
    return torch.tensor(0.0, device=image.device)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="训练分裂高斯模型")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    
    # 增加额外的命令行参数
    # 添加SCS损失权重参数 - 默认设为0禁用
    parser.add_argument('--lambda_scs', type=float, default=0.0,
                        help='结构余弦相似度损失权重 (默认0.0，禁用)')
    # 添加频率域wave损失权重参数
    parser.add_argument('--lambda_freq_wave', type=float, default=0.01,
                        help='频率域wave学习损失权重 (默认0.01)')
    # 添加点云下采样参数以减少初始高斯球数量，加速训练
    parser.add_argument('--downsample_factor', type=float, default=1.0,
                        help='点云下采样因子 (0.1-1.0)，1.0表示不下采样，0.5表示保留50%的点 (默认1.0)')
    parser.add_argument('--max_split_ratio', type=float, default=0.05,
                        help='最大分裂比例，控制分裂预算 (默认0.05，即5%)')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # For SplitGaussianModel, ensure splitting is active unless explicitly disabled?
    # The default=True handles this for now.

    # 初始化系统状态(RNG)
    # 统一：无论是否eval，都直接进入训练主循环
    if hasattr(args, 'eval') and args.eval:
        print("评估模式：使用原始模型路径 " + args.model_path)
    else:
        # 训练模式下添加时间戳
        exp_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        if args.model_path is None or args.model_path == "": # Check if path is empty
            args.model_path = "output"
        args.model_path = args.model_path + "/" + exp_id
        print("训练模式：优化分裂高斯模型 " + args.model_path)
        setup = vars(args)
        setup["exp_id"] = exp_id

    safe_state(args.quiet, args.seed)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # 完成
    print("\n分裂高斯模型训练完成。") 