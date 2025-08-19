from __future__ import annotations
import os
import datetime
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, apply_dog_filter
from utils.extra_utils import random_id
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, log_to_wandb=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="训练进度")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
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

        # 渐进式频率调整 - 从低频逐渐关注高频
        freq = (iteration / opt.iterations) * 100
        
        # 渲染
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # 修改后的Loss计算 - 添加高频敏感性
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        
        # 应用差分高斯滤波器生成高频掩码
        mask = apply_dog_filter(image.unsqueeze(0), freq=freq, scale_factor=opt.im_laplace_scale_factor).squeeze(0)
        mask_loss = l1_loss(image * mask, gt_image * mask)
        
        # 综合损失函数
        loss = (1.0 - opt.lambda_dssim - opt.lambda_im_laplace) * Ll1 + \
               opt.lambda_dssim * (1.0 - ssim(image, gt_image)) + \
               opt.lambda_im_laplace * mask_loss
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # 记录和保存
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, mask_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] 保存高斯点云".format(iteration))
                scene.save(iteration)

            # 密度调整
            if iteration < opt.densify_until_iter:
                # 跟踪图像空间中最大半径以进行剪枝
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.prune_opacity_threshold, scene.cameras_extent, size_threshold)
                
                # 添加形状剪枝功能，如果模型支持
                if hasattr(opt, 'shape_pruning_interval') and hasattr(gaussians, 'size_prune') and iteration > opt.densify_from_iter and iteration % opt.shape_pruning_interval == 0:
                    if hasattr(opt, 'prune_shape_threshold'):
                        gaussians.size_prune(opt.prune_shape_threshold)
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                
                # 添加形状重置功能，如果模型支持
                if hasattr(opt, 'shape_reset_interval') and hasattr(gaussians, 'reset_shape') and iteration % opt.shape_reset_interval == 0:
                    gaussians.reset_shape()

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
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("输出文件夹: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 只使用 Tensorboard，完全移除 wandb
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard不可用：不记录进度")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, mask_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        # 使用 tensorboard 记录日志
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/mask_loss', mask_loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('scene/total_points', scene.gaussians.get_xyz.shape[0], iteration)
        
        # 添加与train_ges.py相同的评估指标
        if hasattr(scene.gaussians, 'get_shape'):  
            tb_writer.add_scalar('scene/small_points', (scene.gaussians.get_shape < 0.5).sum().item(), iteration)
            tb_writer.add_scalar('scene/average_shape', scene.gaussians.get_shape.mean().item(), iteration)
            
            if (scene.gaussians.get_shape >= 1.0).sum().item() > 0:
                tb_writer.add_scalar('scene/large_shapes', scene.gaussians.get_shape[scene.gaussians.get_shape >= 1.0].mean().item(), iteration)
            
            if (scene.gaussians.get_shape < 1.0).sum().item() > 0:
                tb_writer.add_scalar('scene/small_shapes', scene.gaussians.get_shape[scene.gaussians.get_shape < 1.0].mean().item(), iteration)
            
            if scene.gaussians._shape.grad is not None:
                tb_writer.add_scalar('scene/shape_grads', scene.gaussians._shape.grad.data.norm(2).item(), iteration)
        
        if scene.gaussians._opacity.grad is not None:
            tb_writer.add_scalar('scene/opacity_grads', scene.gaussians._opacity.grad.data.norm(2).item(), iteration)
        
        if scene.gaussians.use_splitting:
            tb_writer.add_scalar('scene/wave_norm', scene.gaussians.get_wave.norm(dim=1).mean().item(), iteration)
            if scene.gaussians._wave.grad is not None:
                tb_writer.add_scalar('scene/wave_grads', scene.gaussians._wave.grad.data.norm(2).item(), iteration)

        # 测试集和训练集采样评估
        if iteration in testing_iterations:
            torch.cuda.empty_cache()
            validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                                 {'name': 'train', 'cameras' : scene.getTrainCameras()})
            
            for config in validation_configs:
                if config['cameras'] and len(config['cameras']) > 0:
                    l1_test = 0
                    psnr_test = 0
                    for idx, viewpoint in enumerate(config['cameras']):
                        render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                        image = render_pkg["render"]
                        gt_image = viewpoint.original_image.cuda()
                        l1_test += l1_loss(image, gt_image).cpu().numpy()
                        psnr_test += psnr(image, gt_image).cpu().numpy()
                    l1_test /= len(config['cameras'])
                    psnr_test /= len(config['cameras'])
                    print("\n[ITER {}] 验证 {}: L1 = {:.6f}, PSNR = {:.3f}".format(iteration, config['name'], l1_test, psnr_test))
                    
                    if tb_writer:
                        tb_writer.add_scalar("metrics/" + config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                        tb_writer.add_scalar("metrics/" + config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

            if tb_writer:
                # 使用 tensorboard 记录直方图
                tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
                tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
                
                # 添加形状直方图（如果存在）
                if hasattr(scene.gaussians, 'get_shape'):
                    tb_writer.add_histogram("scene/shape_histogram", scene.gaussians.get_shape, iteration)
                
                # 如果使用分裂机制，记录wave信息
                if scene.gaussians.use_splitting:
                    tb_writer.add_histogram("scene/wave_histogram", scene.gaussians.get_wave.norm(dim=1), iteration)
            
            torch.cuda.empty_cache()

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="训练脚本参数")
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
    parser.add_argument("--nowandb", action="store_false", dest='wandb')
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument('--split_factor', type=float, default=-1.0,
                       help='分裂高斯的协方差缩放因子。-1表示使用理论值')
    parser.add_argument("--wandb", action="store_true", default=False,
                      help="启用wandb日志记录（默认禁用）")
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    # 初始化系统状态(RNG)
    if args.eval:
        # 评估模式下不添加时间戳和附加信息
        print("评估模式：使用原始模型路径 " + args.model_path)
    else:
        # 训练模式下添加时间戳
        exp_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        args.model_path = args.model_path + "_" + args.exp_set + "_" +  exp_id
        print("训练模式：优化 " + args.model_path)
        setup = vars(args)
        setup["exp_id"] = exp_id

    safe_state(args.quiet, args.seed)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, False)

    # 完成
    print("\n训练完成。")
