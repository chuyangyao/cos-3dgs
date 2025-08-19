#!/usr/bin/env python3
"""
train_optimized.py
渐进式频率训练策略：先低频后高频
修复autocast错误并加入wave感知的密集化剪枝机制
"""

import os
import sys
import time
import torch
import torch.nn.functional as F
import numpy as np
import gc
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser, Namespace

# 导入必要模块
from scene import Scene
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams
from utils.general_utils import safe_state
from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr

# 混合精度训练
from torch.cuda.amp import autocast, GradScaler

# TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

# ============================================================
# 渐进式损失函数
# ============================================================
def compute_progressive_loss(image, gt_image, iteration, max_iteration=40000, 
                            wave_vectors=None, phase="low_freq"):
    """
    渐进式损失函数，根据训练阶段调整权重
    
    Args:
        image: 渲染图像
        gt_image: 真实图像
        iteration: 当前迭代
        max_iteration: 最大迭代
        wave_vectors: Wave参数
        phase: 训练阶段 ("low_freq", "mid_freq", "high_freq")
    """
    
    # 基础L1损失
    l1 = l1_loss(image, gt_image)
    
    # SSIM损失（正确计算）
    ssim_val = ssim(image.unsqueeze(0) if image.dim() == 3 else image,
                    gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image)
    ssim_loss = 1.0 - ssim_val
    
    # 梯度损失（高频信息）
    def compute_gradient_loss(img1, img2):
        # X方向梯度
        dx1 = img1[:, :, 1:] - img1[:, :, :-1]
        dx2 = img2[:, :, 1:] - img2[:, :, :-1]
        # Y方向梯度
        dy1 = img1[:, 1:, :] - img1[:, :-1, :]
        dy2 = img2[:, 1:, :] - img2[:, :-1, :]
        
        grad_loss = F.l1_loss(dx1, dx2) + F.l1_loss(dy1, dy2)
        return grad_loss
    
    grad_loss = compute_gradient_loss(image, gt_image)
    
    # 根据训练阶段调整权重
    if phase == "low_freq":
        # 阶段1：专注低频重建
        # 高L1和SSIM权重，低梯度权重
        w_l1 = 0.8
        w_ssim = 0.2
        w_grad = 0.0  # 不关注高频
        w_wave_reg = 0.0  # 不使用wave
        
    elif phase == "mid_freq":
        # 阶段2：引入中频
        # 平衡的权重，开始关注梯度
        progress = (iteration - 10000) / 15000  # 0 -> 1
        w_l1 = 0.6
        w_ssim = 0.2
        w_grad = 0.2 * progress  # 逐渐增加梯度权重
        w_wave_reg = 0.001  # 轻微的wave正则化
        
    else:  # high_freq
        # 阶段3：精细化高频
        # 更关注细节
        w_l1 = 0.4
        w_ssim = 0.2
        w_grad = 0.4  # 高梯度权重
        w_wave_reg = 0.0005  # 减少正则化
    
    # 计算总损失
    total_loss = w_l1 * l1 + w_ssim * ssim_loss + w_grad * grad_loss
    
    # Wave正则化（如果有）
    if wave_vectors is not None and w_wave_reg > 0:
        wave_norms = torch.norm(wave_vectors, dim=1)
        # 软约束：鼓励wave在合理范围内
        wave_reg = wave_norms.mean() * w_wave_reg
        
        # 防止wave爆炸
        if wave_norms.max() > 10.0:
            wave_reg += torch.relu(wave_norms - 10.0).mean() * 0.1
        
        total_loss += wave_reg
    else:
        wave_reg = 0.0
    
    # 返回损失和详细信息
    loss_dict = {
        'total': total_loss.item(),
        'l1': l1.item(),
        'ssim': ssim_loss.item(),
        'gradient': grad_loss.item(),
        'wave_reg': wave_reg.item() if torch.is_tensor(wave_reg) else wave_reg,
        'phase': phase
    }
    
    return total_loss, loss_dict

# ============================================================
# 渲染函数
# ============================================================
def render_optimized(viewpoint_camera, gaussians, pipe, bg, 
                     iteration=0, max_iteration=40000, use_amp=False):
    """优化的渲染函数"""
    
    # 设置分裂和wave状态
    if hasattr(gaussians, 'use_splitting'):
        use_split = gaussians.use_splitting
    else:
        use_split = False
    
    if hasattr(gaussians, 'use_wave'):
        use_wave = gaussians.use_wave
    else:
        use_wave = False
    
    # 调用基础渲染
    render_pkg = render(viewpoint_camera, gaussians, pipe, bg)
    
    return render_pkg

# ============================================================
# Wave初始化
# ============================================================
def initialize_wave_progressive(gaussians, iteration):
    """渐进式wave初始化"""
    with torch.no_grad():
        N = gaussians._xyz.shape[0]
        device = gaussians._xyz.device
        
        # 基于梯度累积初始化
        if hasattr(gaussians, 'xyz_gradient_accum') and gaussians.xyz_gradient_accum is not None:
            grad_norms = torch.norm(gaussians.xyz_gradient_accum, dim=1)
            grad_norms = grad_norms / (grad_norms.max() + 1e-8)
            
            # 高梯度区域给予更大的初始wave
            wave_init = torch.randn(N, 3, device=device) * 0.1
            wave_init *= (1.0 + grad_norms.unsqueeze(1) * 2.0)
        else:
            # 随机初始化
            wave_init = torch.randn(N, 3, device=device) * 0.1
        
        gaussians._wave.data = wave_init
        print(f"[Wave Init] Initialized {N} wave vectors at iteration {iteration}")

# ============================================================
# 训练主函数
# ============================================================
def training_optimized(dataset, opt, pipe, testing_iterations, saving_iterations, 
                       checkpoint_iterations, checkpoint, debug_from):
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    # 检查点加载
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    # 背景设置
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 混合精度
    use_amp = False  # 暂时禁用，确保稳定性
    scaler = GradScaler(enabled=use_amp)
    
    # ========== 训练阶段定义 ==========
    PHASE1_END = 10000   # 低频阶段结束
    PHASE2_END = 25000   # 中频阶段结束
    
    # ========== 训练循环 ==========
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training")
    
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        progress_bar.update(1)
        
        iter_start.record()
        
        # ========== 确定训练阶段 ==========
        if iteration <= PHASE1_END:
            phase = "low_freq"
            gaussians.use_wave = False
            gaussians.use_splitting = False
        elif iteration <= PHASE2_END:
            phase = "mid_freq"
            gaussians.use_wave = True
            gaussians.use_splitting = False
            # 在阶段2开始时初始化wave
            if iteration == PHASE1_END + 1:
                initialize_wave_progressive(gaussians, iteration)
        else:
            phase = "high_freq"
            gaussians.use_wave = True
            gaussians.use_splitting = True
        
        # ========== 学习率调度 ==========
        gaussians.update_learning_rate(iteration)
        
        # Wave学习率调度（阶段2和3）
        if phase != "low_freq":
            for param_group in gaussians.optimizer.param_groups:
                if param_group["name"] == "wave":
                    # 渐进式学习率
                    if phase == "mid_freq":
                        base_lr = opt.wave_lr * 0.5  # 中频阶段使用较小学习率
                    else:
                        base_lr = opt.wave_lr  # 高频阶段使用完整学习率
                    
                    # Cosine衰减
                    progress = (iteration - PHASE1_END) / (opt.iterations - PHASE1_END)
                    lr = base_lr * (0.5 * (1 + np.cos(np.pi * progress)))
                    param_group['lr'] = lr
        
        # SH degree增加
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # ========== 选择相机 ==========
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # ========== 渲染 ==========
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 修复autocast使用方式（兼容旧版PyTorch）
        with autocast(enabled=use_amp):
            render_pkg = render_optimized(
                viewpoint_cam, gaussians, pipe, background,
                iteration=iteration,
                max_iteration=opt.iterations,
                use_amp=use_amp
            )
            
            image = render_pkg["render"]
            viewspace_point_tensor = render_pkg["viewspace_points"]
            visibility_filter = render_pkg["visibility_filter"]
            radii = render_pkg["radii"]
        
        # ========== 计算损失 ==========
        gt_image = viewpoint_cam.original_image.cuda()
        
        # 使用渐进式损失函数
        wave_vectors = gaussians._wave if hasattr(gaussians, '_wave') and phase != "low_freq" else None
        Ll1 = l1_loss(image, gt_image)
        
        loss, loss_dict = compute_progressive_loss(
            image, gt_image, 
            iteration=iteration,
            max_iteration=opt.iterations,
            wave_vectors=wave_vectors,
            phase=phase
        )
        
        # ========== 反向传播 ==========
        loss.backward()
        iter_end.record()
        
        with torch.no_grad():
            # ========== 梯度裁剪 ==========
            if phase != "low_freq" and hasattr(gaussians, '_wave'):
                if gaussians._wave.grad is not None:
                    # 梯度裁剪
                    grad_norm = gaussians._wave.grad.norm()
                    if grad_norm > 1.0:
                        gaussians._wave.grad *= 1.0 / grad_norm
            
            # ========== 更新进度条 ==========
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.4f}",
                    "Phase": phase,
                    "Points": f"{gaussians._xyz.shape[0]:,}"
                })
            
            # ========== Wave感知的密集化（Densification） ==========
            if iteration < opt.densify_until_iter:
                # 记录梯度统计
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                
                # 执行密集化
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    # 根据阶段调整密集化阈值
                    if phase == "low_freq":
                        grad_threshold = opt.densify_grad_threshold
                    elif phase == "mid_freq":
                        grad_threshold = opt.densify_grad_threshold * 0.8
                    else:
                        grad_threshold = opt.densify_grad_threshold * 0.6
                    
                    # 使用wave感知的密集化和剪枝
                    if hasattr(gaussians, 'densify_and_prune_wave_aware'):
                        gaussians.densify_and_prune_wave_aware(
                            grad_threshold,
                            opt.prune_opacity_threshold,
                            scene.cameras_extent,
                            size_threshold,
                            iteration
                        )
                    else:
                        gaussians.densify_and_prune(
                            grad_threshold,
                            opt.prune_opacity_threshold,
                            scene.cameras_extent,
                            size_threshold
                        )
                
                # 重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            # ========== 优化器步骤 ==========
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
            
            # ========== 保存检查点 ==========
            if (iteration in checkpoint_iterations):
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                torch.save((gaussians.capture(), iteration), 
                          scene.model_path + "/chkpnt" + str(iteration) + ".pth")
            
            # ========== 评估和保存 ==========
            if iteration in testing_iterations:
                evaluate_and_save(scene, gaussians, background, iteration, pipe)
            
            if iteration in saving_iterations:
                save_gaussians(scene, gaussians, iteration)
        
        # 定期清理内存
        if iteration % 1000 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    print("\n[Training complete]")
    return gaussians

# ============================================================
# 辅助函数
# ============================================================
def prepare_output_and_logger(args):
    """准备输出目录和日志"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    
    return tb_writer

def evaluate_and_save(scene, gaussians, background, iteration, pipe):
    """评估并保存结果"""
    torch.cuda.empty_cache()
    validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                         {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] 
                                                        for idx in range(5, 30, 5)]})

    for config in validation_configs:
        if config['cameras'] and len(config['cameras']) > 0:
            l1_test = 0.0
            psnr_test = 0.0
            
            for idx, viewpoint in enumerate(config['cameras']):
                render_pkg = render(viewpoint, gaussians, pipe, background)
                image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                
                l1_test += l1_loss(image, gt_image).mean().double()
                psnr_test += psnr(image, gt_image).mean().double()
            
            psnr_test /= len(config['cameras'])
            l1_test /= len(config['cameras'])
            
            print(f"\n[ITER {iteration}] Evaluating {config['name']}: L1 {l1_test:.5f} PSNR {psnr_test:.5f}")

def save_gaussians(scene, gaussians, iteration):
    """保存高斯模型"""
    point_cloud_path = os.path.join(scene.model_path, f"point_cloud/iteration_{iteration}")
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 设置参数解析器
    parser = ArgumentParser(description="Training optimized")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    
    # 设置随机种子
    safe_state(args.quiet)
    
    # 网络GUI
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 开始训练
    training_optimized(lp.extract(args), op.extract(args), pp.extract(args),
                      args.test_iterations, args.save_iterations,
                      args.checkpoint_iterations, args.start_checkpoint, 
                      args.debug_from)
    
    print("\nTraining complete.")