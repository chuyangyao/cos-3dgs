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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_laplacian, render_new
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.new_model import SplitLaplacianModel, SplitGaussianModel
from scene.gaussian_model import GaussianModel
from scene.laplacian_model import LaplacianModel
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, model_type="split_gaussian", render_dir=None):
    """
    渲染视图集合
    
    Args:
        model_path: 模型路径
        name: 渲染名称
        iteration: 迭代次数
        views: 视图列表
        gaussians: 高斯模型
        pipeline: 渲染管线参数
        background: 背景颜色
        model_type: 模型类型("split_gaussian", "ges", "split_laplacian")
        render_dir: 渲染输出目录名，不指定时根据model_type自动选择
    """
    # 自动确定渲染目录名称
    if render_dir is None:
        render_dir = model_type  # 直接使用模型类型作为目录名
    
    # 创建迭代次数子目录
    iter_dir = f"iter_{iteration}" if iteration > 0 else ""
    
    base_render_path = os.path.join(model_path, "test", render_dir)
    render_path = os.path.join(base_render_path, "renders", iter_dir)
    gts_path = os.path.join(base_render_path, "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    print(f"渲染{len(views)}个视图到: {render_path}")
    
    # 检查wave参数是否存在
    if model_type in ["split_gaussian", "split_laplacian"]:
        if not hasattr(gaussians, '_wave') or gaussians._wave is None:
            raise RuntimeError("错误：模型缺少wave参数！请确保使用的是支持分裂的模型。")
    
    # 计时开始
    start_time = time.time()
    
    for idx, view in enumerate(tqdm(views, desc="渲染进度")):
        # 根据模型类型选择不同的渲染函数
        if model_type == "split_gaussian":
            # 分裂高斯方法使用render函数，传递iteration和max_iteration参数
            rendering = render(view, gaussians, pipeline, background, 
                             iteration=iteration, max_iteration=40000)["render"]
        elif model_type == "ges":
            # GES方法（拉普拉斯）使用render_laplacian函数
            rendering = render_laplacian(view, gaussians, pipeline, background)["render"]
        elif model_type == "split_laplacian":
            # 分裂拉普拉斯方法（融合方案）使用render_new函数，传递iteration和max_iteration参数
            rendering = render_new(view, gaussians, pipeline, background,
                                 iteration=iteration, max_iteration=40000)["render"]
        else:
            # 默认使用render函数，传递iteration和max_iteration参数
            rendering = render(view, gaussians, pipeline, background,
                             iteration=iteration, max_iteration=40000)["render"]
            
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    # 计时结束并计算FPS
    end_time = time.time()
    total_time = end_time - start_time
    fps = len(views) / total_time
    print(f"渲染完成！总时间: {total_time:.2f}秒, FPS: {fps:.2f}")

def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool, model_type="split_gaussian", render_dir=None):
    """
    渲染训练集和测试集
    
    Args:
        dataset: 模型参数
        iteration: 迭代次数
        pipeline: 渲染管线参数
        skip_train: 是否跳过训练集渲染
        skip_test: 是否跳过测试集渲染
        model_type: 模型类型("split_gaussian", "ges", "split_laplacian")
        render_dir: 渲染输出目录名，不指定时根据model_type自动选择
    """
    with torch.no_grad():
        # 根据模型类型加载不同的模型
        if model_type == "ges":
            # GES是拉普拉斯模型
            gaussians = LaplacianModel(dataset.sh_degree)
        elif model_type == "split_laplacian":
            # 分裂拉普拉斯模型（融合方案）
            gaussians = SplitLaplacianModel(dataset.sh_degree)
        else:
            # 默认或split_gaussian使用分裂高斯模型
            gaussians = SplitGaussianModel(dataset.sh_degree)
            
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # 如果没有指定迭代次数，使用场景加载的迭代次数
        if iteration == -1:
            iteration = scene.loaded_iter

        if not skip_train:
             render_set(dataset.model_path, "train", iteration, scene.getTrainCameras(), gaussians, pipeline, background, model_type, render_dir)

        if not skip_test:
             render_set(dataset.model_path, "test", iteration, scene.getTestCameras(), gaussians, pipeline, background, model_type, render_dir)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="测试脚本参数")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model_type", type=str, default="split_gaussian", 
                        choices=["split_gaussian", "ges", "split_laplacian"], 
                        help="模型类型: split_gaussian, ges, split_laplacian")
    parser.add_argument("--render_dir", type=str, default=None, 
                        help="渲染目录名称，默认与模型类型相同")
    args = get_combined_args(parser)
    print("渲染 " + args.model_path + f" 使用 {args.model_type} 模型")

    # 初始化系统状态 (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.model_type, args.render_dir)