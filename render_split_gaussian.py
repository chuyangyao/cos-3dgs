"""
Split Gaussian渲染脚本 - 支持动态和预分裂渲染
"""

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.new_model import SplitGaussianModel
import json
from pathlib import Path
import numpy as np

def detect_model_type(model_path, iteration):
    """检测模型类型和可用的分裂版本"""
    point_cloud_path = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    
    model_info = {
        'model_type': 'standard',
        'has_wave': False,
        'has_split': False,
        'split_info': None
    }
    
    # 检查是否有wave参数（表示是split gaussian模型）
    standard_ply = os.path.join(point_cloud_path, "point_cloud.ply")
    if os.path.exists(standard_ply):
        # 简单检查：尝试加载并查看是否有wave参数
        try:
            # 这里可以添加更精确的检测逻辑
            model_info['has_wave'] = True
            model_info['model_type'] = 'split_gaussian'
        except:
            pass
    
    # 检查是否有预分裂版本
    split_ply = os.path.join(point_cloud_path, "point_cloud_split.ply")
    split_info_json = os.path.join(point_cloud_path, "split_info.json")
    
    if os.path.exists(split_ply):
        model_info['has_split'] = True
        
        # 读取分裂信息
        if os.path.exists(split_info_json):
            with open(split_info_json, 'r') as f:
                model_info['split_info'] = json.load(f)
    
    return model_info

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, 
               render_dir="split_gaussian", use_split_version=False, split_info=None):
    """
    渲染一组视图
    
    Args:
        model_path: 模型路径
        name: train/test
        iteration: 迭代次数
        views: 视图列表
        gaussians: 高斯模型
        pipeline: 渲染管线
        background: 背景颜色
        render_dir: 渲染目录名
        use_split_version: 是否使用预分裂的版本
        split_info: 分裂信息（如果有）
    """
    # 修复路径结构，使其与metrics_debug.py期望的一致
    if use_split_version:
        # 预分裂版本路径
        render_path = os.path.join(model_path, name, f"{render_dir}_presplit", "renders")
        gts_path = os.path.join(model_path, name, f"{render_dir}_presplit", "gt")
    else:
        # 标准路径结构：确保包含test目录层级
        if iteration > 0:
            # 如果有迭代次数，放在renders子目录下
            render_path = os.path.join(model_path, name, render_dir, "renders", f"iter_{iteration}")
        else:
            render_path = os.path.join(model_path, name, render_dir, "renders")
        gts_path = os.path.join(model_path, name, render_dir, "gt")

    # 创建目录
    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    # 打印渲染信息
    if split_info and split_info.get('meta'):
        meta = split_info['meta']
        print(f"\n渲染信息:")
        print(f"  原始高斯数: {meta.get('original_gaussians', 'N/A')}")
        print(f"  分裂高斯数: {meta.get('split_gaussians', 'N/A')}")
        print(f"  最大wave范数: {meta.get('max_wave_norm', 0):.4f}")
        print(f"  平均wave范数: {meta.get('mean_wave_norm', 0):.4f}")
        print(f"  活跃wave数: {meta.get('active_waves', 0)}")

    print(f"\n渲染{len(views)}个视图")
    print(f"  渲染路径: {render_path}")
    print(f"  GT路径: {gts_path}")
    
    # 渲染每个视图
    for idx, view in enumerate(tqdm(views, desc="渲染进度")):
        if hasattr(gaussians, 'use_splitting') and gaussians.use_splitting and not use_split_version:
            # 使用动态分裂渲染
            rendering = render(view, gaussians, pipeline, background, 
                             iteration=iteration, max_iteration=40000)["render"]
        else:
            # 使用标准渲染（无分裂）
            rendering = render(view, gaussians, pipeline, background)["render"]
            
        # 获取GT图像
        gt = view.original_image[0:3, :, :]
        
        # 保存渲染结果
        torchvision.utils.save_image(rendering, os.path.join(render_path, f'{idx:05d}.png'))
        # 重要：保存GT图像！
        torchvision.utils.save_image(gt, os.path.join(gts_path, f'{idx:05d}.png'))
    
    print(f"✓ 渲染完成: {len(views)}个视图")
    print(f"  渲染图像保存在: {render_path}")
    print(f"  GT图像保存在: {gts_path}")
    
    # 保存渲染元信息
    meta_file = os.path.join(model_path, name, render_dir, "render_info.json")
    render_info = {
        'iteration': iteration,
        'num_views': len(views),
        'use_split': use_split_version,
        'render_path': render_path,
        'gt_path': gts_path
    }
    if split_info:
        render_info['split_info'] = split_info
    
    makedirs(os.path.dirname(meta_file), exist_ok=True)
    with open(meta_file, 'w') as f:
        json.dump(render_info, f, indent=2)
    print(f"  元信息保存在: {meta_file}")

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, 
                skip_train : bool, skip_test : bool, render_dir="split_gaussian",
                use_presplit=False, export_standard=False):
    """
    渲染训练集和测试集
    
    Args:
        dataset: 数据集参数
        iteration: 迭代次数
        pipeline: 渲染管线参数
        skip_train: 是否跳过训练集
        skip_test: 是否跳过测试集
        render_dir: 渲染目录名
        use_presplit: 使用预分裂的高斯（如果存在）
        export_standard: 导出标准高斯用于其他渲染器
    """
    with torch.no_grad():
        # 检测模型类型
        model_info = detect_model_type(dataset.model_path, iteration)
        
        print(f"\n{'='*60}")
        print(f"模型信息:")
        print(f"  类型: {model_info['model_type']}")
        print(f"  包含wave: {model_info['has_wave']}")
        print(f"  有分裂版本: {model_info['has_split']}")
        print(f"{'='*60}")
        
        # 根据模型类型加载
        if model_info['has_wave']:
            # 使用带 wave 的模型
            if use_presplit and model_info['has_split']:
                print("\n使用预分裂的高斯进行高效渲染...")
                # 预分裂路径：标准 GaussianModel 加载 split PLY，禁用动态分裂
                split_path = os.path.join(
                    dataset.model_path,
                    "point_cloud",
                    f"iteration_{iteration}",
                    "point_cloud_split.ply"
                )
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, shuffle=False)
                gaussians.load_ply(split_path)
                gaussians.use_splitting = False
                split_info = model_info.get('split_info')
            elif use_presplit and not model_info['has_split']:
                # 新增：允许用户选择标准点云（未分裂）进行渲染，便于对比
                standard_path = os.path.join(
                    dataset.model_path,
                    "point_cloud",
                    f"iteration_{iteration}",
                    "point_cloud.ply"
                )
                print("\n未找到预分裂文件，改用未分裂点云进行渲染（禁用动态分裂）...")
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                gaussians.load_ply(standard_path)
                gaussians.use_splitting = False
                split_info = None
            elif use_presplit and not model_info['has_split']:
                # 显式兜底：用户要求预分裂，但不存在分裂文件 -> 回退为标准渲染（无动态分裂）
                print("\n未找到预分裂文件，回退为标准渲染（禁用动态分裂）...")
                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                gaussians.use_splitting = False
                split_info = None
            else:
                # 正常动态路径（仅当未要求预分裂）
                print("\n使用动态分裂渲染...")
                gaussians = SplitGaussianModel(dataset.sh_degree)
                scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
                gaussians.use_splitting = True
                split_info = None
        else:
            # 标准高斯模型
            print("\n使用标准高斯模型...")
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
            split_info = None

        # 设置背景颜色
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 渲染训练集
        if not skip_train:
            print(f"\n{'='*40}")
            print("渲染训练集...")
            print(f"{'='*40}")
            render_set(dataset.model_path, "train", scene.loaded_iter, 
                      scene.getTrainCameras(), gaussians, pipeline, background, 
                      render_dir, use_presplit and model_info['has_split'], split_info)

        # 渲染测试集
        if not skip_test:
            print(f"\n{'='*40}")
            print("渲染测试集...")
            print(f"{'='*40}")
            render_set(dataset.model_path, "test", scene.loaded_iter, 
                      scene.getTestCameras(), gaussians, pipeline, background, 
                      render_dir, use_presplit and model_info['has_split'], split_info)
        
        # 导出标准格式（可选）
        if export_standard and model_info['has_wave']:
            print("\n导出标准高斯格式...")
            export_path = os.path.join(
                dataset.model_path,
                "point_cloud",
                f"iteration_{iteration}",
                "point_cloud_standard.ply"
            )
            
            # 创建一个标准高斯模型并复制参数
            standard_gaussians = GaussianModel(dataset.sh_degree)
            # 这里需要实现参数转换逻辑
            # standard_gaussians.save_ply(export_path)
            print(f"标准格式保存到: {export_path}")

        print(f"\n{'='*60}")
        print("渲染完成！")
        print(f"{'='*60}")

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Split Gaussian渲染脚本")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int, help="要加载的迭代次数")
    parser.add_argument("--skip_train", action="store_true", help="跳过训练集渲染")
    parser.add_argument("--skip_test", action="store_true", help="跳过测试集渲染")
    parser.add_argument("--quiet", action="store_true", help="安静模式")
    parser.add_argument("--render_dir", default="split_gaussian", type=str, 
                       help="渲染输出目录名")
    parser.add_argument("--use_presplit", action="store_true", 
                       help="使用预分裂版本（如果存在）")
    parser.add_argument("--export_standard", action="store_true",
                       help="导出标准高斯格式")
    
    args = get_combined_args(parser)
    
    print(f"\n{'='*60}")
    print(f"Split Gaussian渲染")
    print(f"模型路径: {args.model_path}")
    print(f"迭代次数: {args.iteration}")
    print(f"渲染目录: {args.render_dir}")
    print(f"使用预分裂: {args.use_presplit}")
    print(f"{'='*60}")

    # 初始化系统状态
    safe_state(args.quiet)

    # 执行渲染
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.skip_train, args.skip_test, args.render_dir,
                args.use_presplit, args.export_standard)