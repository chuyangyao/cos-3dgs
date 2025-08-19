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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.laplacian_model import LaplacianModel
from scene.new_model import SplitGaussianModel, SplitLaplacianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch

class Scene:

    gaussians : GaussianModel | LaplacianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel | LaplacianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, save_split_version=False):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        
        # 保存分裂版本
        if save_split_version and hasattr(self.gaussians, 'use_splitting') and self.gaussians.use_splitting:
            print(f"[Save] Creating split version at iteration {iteration}...")
            split_data = self.gaussians.get_split_data(iteration, 40000)  # 假设最大迭代为40000
            
            if split_data is not None:
                split_path = os.path.join(point_cloud_path, "point_cloud_split.ply")
                
                # 创建临时的高斯模型来保存分裂版本
                split_gaussians = type(self.gaussians)(self.gaussians.max_sh_degree)
                
                print(f"[Export] Converting {self.gaussians._xyz.shape[0]} gaussians to {split_data['split_xyz'].shape[0]} split gaussians")
                
                # 设置分裂后的参数
                split_gaussians._xyz = nn.Parameter(split_data['split_xyz'])
                split_gaussians._features_dc = nn.Parameter(split_data['split_features_dc'])
                split_gaussians._features_rest = nn.Parameter(split_data['split_features_rest'])
                split_gaussians._opacity = nn.Parameter(split_data['split_opacity'])
                split_gaussians._scaling = nn.Parameter(split_data['split_scaling'])
                split_gaussians._rotation = nn.Parameter(split_data['split_rotation'])
                
                # 预分裂版本不需要wave参数（已经是分裂后的结果）
                split_gaussians._wave = nn.Parameter(torch.zeros((split_data['split_xyz'].shape[0], 3), device='cuda'))
                
                # 保存分裂版本
                split_gaussians.save_ply(split_path)
                print(f"[Save] Split version saved to {split_path}")
    def save_checkpoint(self, iteration):
        """
        保存完整的检查点，包括优化器状态
        """
        print(f"[Checkpoint] Saving checkpoint at iteration {iteration}")
        
        # 保存点云
        self.save(iteration, save_split_version=True)
        
        # 保存模型和优化器状态
        checkpoint_path = os.path.join(self.model_path, f"chkpnt{iteration}.pth")
        torch.save((self.gaussians.capture(), iteration), checkpoint_path)
        
        print(f"[Checkpoint] Saved to {checkpoint_path}")

    def export_for_standard_rendering(self, iteration=None):
        """
        导出用于标准3DGS渲染的点云（无wave参数）
        
        Args:
            iteration: 指定迭代次数，如果为None则使用最后保存的
        
        Returns:
            导出的文件路径
        """
        if iteration is None:
            iteration = self.loaded_iter if self.loaded_iter else 0
        
        print(f"[Export] Exporting standard gaussians for iteration {iteration}")
        
        # 创建导出目录
        export_path = os.path.join(self.model_path, "export", f"iteration_{iteration}")
        os.makedirs(export_path, exist_ok=True)
        
        # 如果是带wave的模型，转换为标准高斯
        if isinstance(self.gaussians, (SplitGaussianModel, SplitLaplacianModel)):
            from gaussian_renderer import export_splits_to_standard_gaussians
            standard_gaussians = export_splits_to_standard_gaussians(
                self.gaussians,
                iteration=iteration,
                max_iteration=40000
            )
        else:
            standard_gaussians = self.gaussians
        
        # 保存标准高斯
        output_file = os.path.join(export_path, "point_cloud.ply")
        standard_gaussians.save_ply(output_file)
        
        # 保存相机参数（复制）
        cameras_file = os.path.join(self.model_path, "cameras.json")
        if os.path.exists(cameras_file):
            import shutil
            shutil.copy(cameras_file, os.path.join(export_path, "cameras.json"))
        
        # 创建渲染脚本
        render_script = os.path.join(export_path, "render_standard.py")
        with open(render_script, 'w') as f:
            f.write("""#!/usr/bin/env python
# 标准3DGS渲染脚本（无需wave参数）

import torch
from scene import Scene
from gaussian_renderer import render
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams
from argparse import ArgumentParser

def render_standard():
    parser = ArgumentParser(description="Standard 3DGS rendering")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    args = parser.parse_args()
    
    # 加载标准高斯模型
    gaussians = GaussianModel(args.sh_degree)
    scene = Scene(args, gaussians, load_iteration=args.iteration)
    
    # 设置背景
    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # 渲染测试视图
    views = scene.getTestCameras()
    for idx, view in enumerate(views):
        rendering = render(view, gaussians, pipeline.extract(args), background)["render"]
        # 保存渲染结果...
        print(f"Rendered view {idx+1}/{len(views)}")

if __name__ == "__main__":
    render_standard()
""")
        
        print(f"[Export] Exported to: {output_file}")
        print(f"[Export] Original: {self.gaussians._xyz.shape[0]} gaussians")
        if isinstance(standard_gaussians, GaussianModel) and standard_gaussians != self.gaussians:
            print(f"[Export] Split version: {standard_gaussians._xyz.shape[0]} gaussians")
        
        return output_file

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]