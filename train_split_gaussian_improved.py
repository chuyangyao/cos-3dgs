from __future__ import annotations
import os
import torch
import gc
import uuid
import sys
from random import randint
from utils.loss_utils import l1_loss, ssim, compute_comprehensive_frequency_loss
from gaussian_renderer import render, network_gui
from scene import Scene
from scene.new_model import SplitGaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def initialize_wave_by_gradient(gaussians, scene, noise_level=0.1):
    """
    基于梯度信息初始化wave向量，避免全零问题
    在Phase 2开始时调用
    """
    with torch.no_grad():
        device = gaussians._wave.device
        N = gaussians._wave.shape[0]
        
        # 步骤1：添加基础随机噪声，确保没有完全为0的wave
        base_noise = torch.randn(N, 3, device=device) * noise_level
        
        # 步骤2：如果有位置梯度，使用梯度信息
        if gaussians._xyz.grad is not None and gaussians._xyz.grad.norm() > 0:
            # 计算梯度的模长和方向
            grad_norms = torch.norm(gaussians._xyz.grad, dim=1, keepdim=True)
            grad_mean = grad_norms.mean()
            grad_std = grad_norms.std()
            
            # 识别高梯度区域（需要更多细节的区域）
            high_grad_mask = grad_norms > (grad_mean + 0.5 * grad_std)
            
            # 高梯度区域给予更大的初始wave
            wave_init = base_noise.clone()
            wave_init[high_grad_mask.squeeze()] *= 3.0
            
            # 使用梯度方向作为wave的初始方向（部分）
            grad_directions = gaussians._xyz.grad / (grad_norms + 1e-8)
            wave_init += grad_directions * noise_level * 2.0
            
        else:
            # 没有梯度信息，使用纯随机初始化
            wave_init = base_noise
            
            # 基于高斯的尺度来调整wave
            scales = gaussians.get_scaling
            scale_norms = torch.norm(scales, dim=1)
            
            # 大的高斯给予更大的初始wave（可能需要更多分裂）
            large_gaussian_mask = scale_norms > scale_norms.mean()
            wave_init[large_gaussian_mask] *= 2.0
        
        # 步骤3：基于空间分布调整
        # 计算每个高斯到其最近邻的距离
        xyz = gaussians._xyz.data
        
        # 简化的密度估计：随机采样计算局部密度
        sample_size = min(100, N)
        sample_indices = torch.randperm(N, device=device)[:sample_size]
        
        for i in range(0, N, 1000):  # 批处理以节省内存
            batch_end = min(i + 1000, N)
            batch_xyz = xyz[i:batch_end]
            
            # 计算到采样点的距离
            distances = torch.cdist(batch_xyz, xyz[sample_indices])
            min_distances = distances.min(dim=1)[0]
            
            # 密集区域（距离小）给予更大的wave
            density_factor = 1.0 / (min_distances + 0.1)
            density_factor = density_factor / density_factor.mean()  # 归一化
            
            wave_init[i:batch_end] *= density_factor.unsqueeze(1)
        
        # 步骤4：限制wave的最大初始值
        max_init_norm = 0.5  # 最大初始wave模长
        wave_norms = torch.norm(wave_init, dim=1, keepdim=True)
        over_limit = wave_norms > max_init_norm
        wave_init[over_limit.squeeze()] *= (max_init_norm / wave_norms[over_limit.squeeze()])
        
        # 步骤5：确保没有NaN或Inf
        wave_init = torch.nan_to_num(wave_init, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 应用初始化
        gaussians._wave.data = wave_init
        
        # 打印统计信息
        final_norms = torch.norm(wave_init, dim=1)
        print(f"[Wave Init] Statistics:")
        print(f"  Mean norm: {final_norms.mean().item():.4f}")
        print(f"  Max norm: {final_norms.max().item():.4f}")
        print(f"  Min norm: {final_norms.min().item():.4f}")
        print(f"  Active (>0.01): {(final_norms > 0.01).sum().item()}/{N} ({(final_norms > 0.01).sum().item()/N*100:.1f}%)")
        
        return wave_init


# 修改Phase 2进入时的初始化调用
def handle_phase_transition(gaussians, scene, phase, iteration, opt):
    """处理阶段转换"""
    if phase == 2:
        # 进入阶段2：初始化wave
        print("\n" + "="*50)
        print(f"[Phase 2] Initializing wave vectors at iteration {iteration}")
        print("="*50)
        
        # 使用改进的初始化
        initialize_wave_by_gradient(gaussians, scene, noise_level=opt.wave_init_noise)
        
        # 清理缓存
        torch.cuda.empty_cache()
        gaussians.invalidate_split_cache()
        
    elif phase == 3:
        # 进入阶段3：准备分裂
        print("\n" + "="*50)
        print(f"[Phase 3] Enabling splitting at iteration {iteration}")
        print("="*50)
        
        # 检查wave状态
        wave_norms = torch.norm(gaussians._wave, dim=1)
        active_ratio = (wave_norms > 0.01).sum().item() / wave_norms.shape[0]
        
        print(f"[Phase 3] Wave statistics:")
        print(f"  Active gaussians: {active_ratio:.1%}")
        print(f"  Mean wave norm: {wave_norms.mean().item():.4f}")
        print(f"  Max wave norm: {wave_norms.max().item():.4f}")
        
        # 如果wave太少，重新初始化
        if active_ratio < 0.1:
            print(f"[Warning] Only {active_ratio:.1%} gaussians have active wave")
            print("Reinitializing wave vectors...")
            initialize_wave_by_gradient(gaussians, scene, noise_level=opt.wave_init_noise * 2)
        
        gaussians.use_splitting = True
        
        # 降低密集化频率
        opt.densification_interval = int(opt.densification_interval * 1.5)
        
        # 清理缓存
        torch.cuda.empty_cache()
        gaussians.invalidate_split_cache()

# ==========================
# 内存管理设置函数
# ==========================
def setup_memory_management():
    """
    设置PyTorch内存管理优化
    """
    # 1. 设置CUDA内存分配器
    if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    
    # 2. 启用cudnn benchmark以提高性能
    torch.backends.cudnn.benchmark = True
    
    # 3. 设置内存分配策略
    torch.cuda.set_per_process_memory_fraction(0.95)  # 使用95%的GPU内存
    
    # 4. 清理初始内存
    gc.collect()
    torch.cuda.empty_cache()
    
    # 5. 打印内存状态
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU Device: {torch.cuda.get_device_name(device)}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
        print(f"Reserved Memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
        print(f"Allocated Memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")

def log_memory_usage(iteration, phase="Training"):
    """
    记录当前内存使用情况
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = torch.cuda.get_device_properties(0).total_memory / 1024**3 - reserved
        
        if iteration % 1000 == 0 or allocated > 40:  # 每1000步或内存使用超过40GB时记录
            print(f"[{phase}] Iter {iteration} - Memory: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={free:.2f}GB")
        
        # 如果内存使用超过阈值，触发清理
        if allocated > 45:  # 45GB阈值
            print(f"[WARNING] High memory usage detected ({allocated:.2f}GB), triggering cleanup...")
            torch.cuda.empty_cache()
            gc.collect()
            
            # 重新检查
            new_allocated = torch.cuda.memory_allocated() / 1024**3
            print(f"[INFO] Memory after cleanup: {new_allocated:.2f}GB")
            
            return True  # 返回需要采取进一步行动
    
    return False

# ==========================
# 训练阶段控制函数
# ==========================
def get_training_phase(iteration, max_iteration):
    """
    确定当前训练阶段
    
    三阶段策略：
    阶段1 (0-37.5%): 基础高斯训练，无wave，无分裂
    阶段2 (37.5%-75%): Wave学习，无分裂
    阶段3 (75%-100%): 启用分裂，精细调整
    """
    progress = iteration / max_iteration
    
    if progress < 0.375:  # 0-15k iterations (40k total)
        return 1, progress / 0.375
    elif progress < 0.75:  # 15k-30k iterations
        return 2, (progress - 0.375) / 0.375
    else:  # 30k-40k iterations
        return 3, (progress - 0.75) / 0.25

def adjust_learning_rates_by_phase(gaussians, optimizer, iteration, phase, phase_progress, opt):
    """根据训练阶段调整学习率"""
    
    # 基础学习率调度（原始3DGS）
    gaussians.update_learning_rate(iteration)
    
    # 阶段特定调整
    if phase == 1:
        # 阶段1：正常学习率，wave不更新
        for param_group in optimizer.param_groups:
            if param_group["name"] == "wave":
                param_group['lr'] = 0.0
    
    elif phase == 2:
        # 阶段2：引入wave学习
        for param_group in optimizer.param_groups:
            if param_group["name"] == "wave":
                # 渐进式增加wave学习率
                base_wave_lr = opt.wave_lr
                param_group['lr'] = base_wave_lr * phase_progress
            elif param_group["name"] in ["xyz", "scaling", "rotation"]:
                # 略微降低其他参数的学习率
                param_group['lr'] *= 0.8
    
    elif phase == 3:
        # 阶段3：精细调整，所有学习率降低
        decay_factor = 1.0 - 0.7 * phase_progress  # 从1.0降到0.3
        for param_group in optimizer.param_groups:
            if param_group["name"] == "wave":
                param_group['lr'] = opt.wave_lr * 0.5 * decay_factor
            else:
                param_group['lr'] *= decay_factor

def initialize_wave_by_gradient(gaussians, scene):
    """基于梯度信息初始化wave参数"""
    print("Initializing wave parameters based on gradient information...")
    
    with torch.no_grad():
        if gaussians.xyz_gradient_accum is not None and gaussians.denom.sum() > 0:
            # 计算平均梯度
            avg_grads = gaussians.xyz_gradient_accum / (gaussians.denom + 1e-8)
            avg_grads = avg_grads.squeeze()
            
            # 归一化到[0, 1]
            if avg_grads.max() > 0:
                normalized_grads = avg_grads / avg_grads.max()
            else:
                normalized_grads = torch.zeros_like(avg_grads)
            
            # 高梯度区域需要更大的wave
            wave_magnitude = normalized_grads * 0.5  # 最大初始wave为0.5
            
            # 随机方向
            random_directions = torch.randn_like(gaussians._wave)
            random_directions = random_directions / (torch.norm(random_directions, dim=1, keepdim=True) + 1e-8)
            
            # 设置wave
            gaussians._wave.data = random_directions * wave_magnitude.unsqueeze(1)
            
            # 稀疏化：只在高梯度区域设置wave
            low_grad_mask = normalized_grads < 0.1
            gaussians._wave.data[low_grad_mask] = 0.0
            
            print(f"Wave initialized: {(wave_magnitude > 0.01).sum().item()}/{gaussians._wave.shape[0]} active")
        else:
            # 如果没有梯度信息，使用小随机值
            gaussians._wave.data = torch.randn_like(gaussians._wave) * 0.01

def prepare_output_and_logger(args):    
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

# ==========================
# 主训练函数
# ==========================
def training(dataset, opt, pipe, testing_iterations, saving_iterations, 
             checkpoint_iterations, checkpoint, debug_from):
    """三阶段训练主函数"""
    
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = SplitGaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 记录当前阶段和阶段转换点
    current_phase = 0
    phase_1_end = int(opt.iterations * 0.375)  # 15k for 40k total
    phase_2_end = int(opt.iterations * 0.75)    # 30k for 40k total
    current_phase_name = "Phase 1"
    
    for iteration in range(first_iter, opt.iterations + 1):
        # GUI连接处理
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
        
        # 获取当前训练阶段
        phase, phase_progress = get_training_phase(iteration, opt.iterations)
        
        # ==========================
        # 阶段转换处理（包含内存清理）
        # ==========================
        if phase != current_phase:
            print(f"\n{'='*50}")
            print(f"[Phase Transition] Entering Phase {phase} at iteration {iteration}")
            print(f"{'='*50}")
            
            # 清理内存
            gaussians.invalidate_split_cache()
            torch.cuda.empty_cache()
            gc.collect()
            
            current_phase = phase
            
            if phase == 1:
                current_phase_name = "Phase 1"
            elif phase == 2:
                current_phase_name = "Phase 2"
                # 进入阶段2：初始化wave
                initialize_wave_by_gradient(gaussians, scene)
                # 清理缓存
                gaussians.invalidate_split_cache()
            elif phase == 3:
                current_phase_name = "Phase 3"
                # 进入阶段3：准备分裂
                print("Phase 3: Splitting enabled")
                gaussians.use_splitting = True
                # 降低密集化频率
                opt.densification_interval = opt.densification_interval * 2
                # 清理内存以准备分裂计算
                torch.cuda.empty_cache()
                gc.collect()

        # 调整学习率
        adjust_learning_rates_by_phase(gaussians, gaussians.optimizer, iteration, phase, phase_progress, opt)

        # 每1000次迭代增加SH degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 随机选择相机
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        # 根据阶段决定渲染策略
        if phase < 3:
            # 阶段1和2：不使用分裂
            render_iteration = 0
        else:
            # 阶段3：使用分裂
            render_iteration = iteration
        
        # ==========================
        # Phase 3内存优化：每隔一定迭代清理缓存
        # ==========================
        if current_phase == 3 and iteration % 500 == 0:
            gaussians.invalidate_split_cache()
            torch.cuda.empty_cache()
        
        # 渲染
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, 
                          iteration=render_iteration, max_iteration=opt.iterations)
        
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        # 保存split_data的引用（Phase 3需要）
        split_data = render_pkg.get('split_data', None)
        
        # GT图像
        gt_image = viewpoint_cam.original_image.cuda()
        
        # ==========================
        # 计算损失（根据阶段调整）
        # ==========================
        if phase == 1:
            # 阶段1：简单的L1 + SSIM
            Ll1 = l1_loss(image, gt_image)
            ssim_val = ssim(image.unsqueeze(0) if image.dim() == 3 else image, 
                          gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
            
            loss_dict = {
                'l1': Ll1.item(),
                'ssim': (1.0 - ssim_val).item(),
                'total': loss.item()
            }
            
        elif phase == 2:
            # 阶段2：Wave训练（不分裂）
            loss, loss_dict = compute_comprehensive_frequency_loss(
                rendered=image,
                gt=gt_image,
                wave_vectors=gaussians._wave,
                visibility_filter=visibility_filter,  # 此时维度匹配
                iteration=iteration,
                max_iter=opt.iterations
            )
            
        elif phase == 3:
            # 阶段3：分裂阶段 - 特殊处理wave_vectors
            if split_data is not None and 'original_indices' in split_data:
                # 创建扩展的wave_vectors以匹配分裂后的高斯
                original_indices = split_data['original_indices']
                expanded_wave_vectors = gaussians._wave[original_indices]
                
                # 现在维度匹配，可以安全使用
                loss, loss_dict = compute_comprehensive_frequency_loss(
                    rendered=image,
                    gt=gt_image,
                    wave_vectors=expanded_wave_vectors,  # 扩展后匹配visibility_filter维度
                    visibility_filter=visibility_filter,
                    iteration=iteration,
                    max_iter=opt.iterations
                )
            else:
                # 降级到基础损失（不应该发生）
                print(f"[Warning] No split data available at iteration {iteration}, using basic loss")
                Ll1 = l1_loss(image, gt_image)
                ssim_val = ssim(image.unsqueeze(0) if image.dim() == 3 else image, 
                              gt_image.unsqueeze(0) if gt_image.dim() == 3 else gt_image)
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)
                
                loss_dict = {
                    'l1': Ll1.item(),
                    'ssim': (1.0 - ssim_val).item(),
                    'total': loss.item()
                }
        
        # 及时删除split_data引用以节省内存（在使用后）
        if split_data is not None:
            del split_data
        
        # ==========================
        # 反向传播
        # ==========================
        loss.backward()
        
        # ==========================
        # 处理分裂高斯的梯度聚合（阶段3）
        # ==========================
        if phase == 3 and "split_viewspace_points" in render_pkg and render_pkg["split_viewspace_points"] is not None:
            split_viewspace_points = render_pkg["split_viewspace_points"]
            split_info = render_pkg["split_info"]
            
            if split_viewspace_points.grad is not None:
                # 创建聚合梯度tensor
                aggregated_grad = torch.zeros((split_info['n_original'], 3), 
                                            device='cuda', dtype=split_viewspace_points.grad.dtype)
                
                # 使用scatter_add聚合梯度
                aggregated_grad = aggregated_grad.scatter_add(0,
                    split_info['original_indices'].unsqueeze(1).expand(-1, 3),
                    split_viewspace_points.grad[:, :3])
                
                # 将聚合的梯度传回原始viewspace_points
                if viewspace_point_tensor.grad is None:
                    viewspace_point_tensor.grad = torch.zeros_like(viewspace_point_tensor)
                viewspace_point_tensor.grad[:, :3] = aggregated_grad

        # ==========================
        # 优化器步进和其他处理
        # ==========================
        iter_end.record()

        with torch.no_grad():
            # 进度条更新
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            
            # 密集化
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], 
                    radii[visibility_filter]
                )
                
                # 添加密集化统计
                # 注意：在Phase 3时，viewspace_point_tensor是原始高斯的
                # visibility_filter是分裂后的，所以需要特殊处理
                if phase < 3:
                    # Phase 1和2：正常处理
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
                else:
                    # Phase 3：使用原始高斯的visibility
                    # 创建原始高斯的visibility_filter
                    if 'split_info' in render_pkg and render_pkg['split_info'] is not None:
                        split_info = render_pkg['split_info']
                        n_original = split_info['n_original']
                        original_indices = split_info['original_indices']
                        
                        # 创建原始高斯的visibility
                        original_visibility = torch.zeros(n_original, dtype=torch.bool, device='cuda')
                        visible_split_indices = torch.where(visibility_filter)[0]
                        for idx in visible_split_indices:
                            orig_idx = original_indices[idx]
                            original_visibility[orig_idx] = True
                        
                        gaussians.add_densification_stats(viewspace_point_tensor, original_visibility)
                    else:
                        # 降级处理
                        print(f"[Warning] Cannot compute densification stats at iteration {iteration}")

                # 根据阶段调整密集化策略
                if phase == 1:
                    # 阶段1：正常密集化
                    if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, 
                                                   scene.cameras_extent, size_threshold)
                elif phase == 2:
                    # 阶段2：降低密集化频率
                    if iteration > opt.densify_from_iter and iteration % (opt.densification_interval * 2) == 0:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        gaussians.densify_and_prune(opt.densify_grad_threshold * 1.5, 0.01, 
                                                   scene.cameras_extent, size_threshold)
                elif phase == 3:
                    # 阶段3：最小化密集化
                    if iteration % (opt.densification_interval * 3) == 0:
                        gaussians.densify_and_prune(opt.densify_grad_threshold * 2.0, 0.02, 
                                                   scene.cameras_extent, 20)
                
                # 重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or \
                   (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
                # 清理分裂缓存
                if phase == 3:
                    gaussians.invalidate_split_cache()

            # 优化器步进
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            # 记录到TensorBoard
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('train/loss', ema_loss_for_log, iteration)
                tb_writer.add_scalar('train/phase', phase, iteration)
                tb_writer.add_scalar('train/num_points', gaussians._xyz.shape[0], iteration)
                
                for key, value in loss_dict.items():
                    tb_writer.add_scalar(f'loss/{key}', value, iteration)
                
                if phase >= 2:
                    wave_norms = torch.norm(gaussians._wave, dim=1)
                    tb_writer.add_scalar('wave/mean', wave_norms.mean().item(), iteration)
                    tb_writer.add_scalar('wave/max', wave_norms.max().item(), iteration)
                    tb_writer.add_scalar('wave/active_ratio', 
                                       (wave_norms > 0.01).sum().item() / wave_norms.shape[0], 
                                       iteration)

            # ==========================
            # 保存检查点（包含内存清理）
            # ==========================
            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving checkpoint")
                scene.save(iteration)
                torch.cuda.empty_cache()  # 保存后清理内存

            # ==========================
            # 测试评估（包含内存清理）
            # ==========================
            if iteration in testing_iterations:
                print(f"\n[ITER {iteration}] Evaluating")
                torch.cuda.empty_cache()
                
                validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                                     {'name': 'train', 'cameras': scene.getTrainCameras()})
                
                for config in validation_configs:
                    if config['cameras'] and len(config['cameras']) > 0:
                        l1_test = 0
                        psnr_test = 0
                        
                        for idx, viewpoint in enumerate(config['cameras']):
                            # 测试时使用当前阶段的设置
                            test_render = render(viewpoint, gaussians, pipe, background,
                                               iteration=render_iteration, 
                                               max_iteration=opt.iterations)
                            test_image = test_render["render"]
                            test_gt = viewpoint.original_image.cuda()
                            
                            l1_test += l1_loss(test_image, test_gt).item()
                            psnr_test += psnr(test_image, test_gt).mean().item()
                        
                        l1_test /= len(config['cameras'])
                        psnr_test /= len(config['cameras'])
                        
                        print(f"  {config['name']}: L1={l1_test:.5f}, PSNR={psnr_test:.2f}")
                        
                        if tb_writer:
                            tb_writer.add_scalar(f"{config['name']}/l1", l1_test, iteration)
                            tb_writer.add_scalar(f"{config['name']}/psnr", psnr_test, iteration)
                
                # 评估后清理内存
                torch.cuda.empty_cache()

if __name__ == "__main__":
    # 设置内存管理
    setup_memory_management()
    
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script with three-stage strategy")
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
    
    # 添加特殊的checkpoint选项
    parser.add_argument("--resume_from_phase3", action="store_true", 
                       help="Resume from iteration 29990 checkpoint for phase 3 debugging")
    parser.add_argument("--custom_checkpoint", type=str, default=None,
                       help="Path to a specific checkpoint file to resume from")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    # 自动添加29990轮的保存点（如果训练会到达这个点）
    if args.iterations >= 30000 and 29990 not in args.save_iterations:
        args.save_iterations.append(29990)
        print(f"[Info] Added checkpoint at iteration 29990 for phase 3 debugging")
    
    # 处理从Phase 3恢复的情况
    if args.resume_from_phase3:
        checkpoint_path = os.path.join(args.model_path, "checkpoints", "chkpnt29990.pth")
        if os.path.exists(checkpoint_path):
            args.start_checkpoint = checkpoint_path
            print(f"[Resume] Loading checkpoint from iteration 29990 for Phase 3 debugging")
        else:
            print(f"[Warning] Checkpoint at 29990 not found. Starting from beginning.")
            print(f"         Expected path: {checkpoint_path}")
    
    # 处理自定义checkpoint
    if args.custom_checkpoint:
        if os.path.exists(args.custom_checkpoint):
            args.start_checkpoint = args.custom_checkpoint
            print(f"[Resume] Loading custom checkpoint from {args.custom_checkpoint}")
        else:
            print(f"[Error] Custom checkpoint not found: {args.custom_checkpoint}")
            exit(1)
    
    print("Optimizing " + args.model_path)
    
    # 初始化系统状态
    safe_state(args.quiet)

    # 配置
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # 开始训练
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # 完成
    print("\nTraining complete.")
    
    # 最终内存清理
    torch.cuda.empty_cache()
    gc.collect()
