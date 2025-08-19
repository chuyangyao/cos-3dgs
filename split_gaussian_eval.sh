#!/bin/bash

# 模型训练评估
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/bicycle -i images_4 -m ../autodl-tmp/eval/split_gaussian_eval/bicycle --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/flowers -i images_4 -m ../autodl-tmp/eval/split_gaussian_eval/flowers --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/garden -i images_4 -m ../autodl-tmp/eval/split_gaussian_eval/garden --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/stump -i images_4 -m ../autodl-tmp/eval/split_gaussian_eval/stump --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/treehill -i images_4 -m ../autodl-tmp/eval/split_gaussian_eval/treehill --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/room -i images_2 -m ../autodl-tmp/eval/split_gaussian_eval/room --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/counter -i images_2 -m ../autodl-tmp/eval/split_gaussian_eval/counter --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/kitchen -i images_2 -m ../autodl-tmp/eval/split_gaussian_eval/kitchen --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/nerf_360/bonsai -i images_2 -m ../autodl-tmp/eval/split_gaussian_eval/bonsai --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/tandt_db/tandt/truck -m ../autodl-tmp/eval/split_gaussian_eval/truck --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/tandt_db/tandt/train -m ../autodl-tmp/eval/split_gaussian_eval/train --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/tandt_db/db/drjohnson -m ../autodl-tmp/eval/split_gaussian_eval/drjohnson --quiet --eval --test_iterations -1
python train_split_gaussian.py -s ../autodl-tmp/tandt_db/db/playroom -m ../autodl-tmp/eval/split_gaussian_eval/playroom --quiet --eval --test_iterations -1

# 中间迭代结果和最终结果渲染
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/bicycle -m ../autodl-tmp/eval/split_gaussian_eval/bicycle --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/bicycle -m ../autodl-tmp/eval/split_gaussian_eval/bicycle --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/flowers -m ../autodl-tmp/eval/split_gaussian_eval/flowers --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/flowers -m ../autodl-tmp/eval/split_gaussian_eval/flowers --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/garden -m ../autodl-tmp/eval/split_gaussian_eval/garden --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/garden -m ../autodl-tmp/eval/split_gaussian_eval/garden --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/stump -m ../autodl-tmp/eval/split_gaussian_eval/stump --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/stump -m ../autodl-tmp/eval/split_gaussian_eval/stump --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/treehill -m ../autodl-tmp/eval/split_gaussian_eval/treehill --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/treehill -m ../autodl-tmp/eval/split_gaussian_eval/treehill --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/room -m ../autodl-tmp/eval/split_gaussian_eval/room --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/room -m ../autodl-tmp/eval/split_gaussian_eval/room --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/counter -m ../autodl-tmp/eval/split_gaussian_eval/counter --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/counter -m ../autodl-tmp/eval/split_gaussian_eval/counter --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/kitchen -m ../autodl-tmp/eval/split_gaussian_eval/kitchen --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/kitchen -m ../autodl-tmp/eval/split_gaussian_eval/kitchen --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/nerf_360/bonsai -m ../autodl-tmp/eval/split_gaussian_eval/bonsai --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/nerf_360/bonsai -m ../autodl-tmp/eval/split_gaussian_eval/bonsai --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/tandt_db/tandt/truck -m ../autodl-tmp/eval/split_gaussian_eval/truck --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/tandt_db/tandt/truck -m ../autodl-tmp/eval/split_gaussian_eval/truck --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/tandt_db/tandt/train -m ../autodl-tmp/eval/split_gaussian_eval/train --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/tandt_db/tandt/train -m ../autodl-tmp/eval/split_gaussian_eval/train --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/tandt_db/db/drjohnson -m ../autodl-tmp/eval/split_gaussian_eval/drjohnson --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/tandt_db/db/drjohnson -m ../autodl-tmp/eval/split_gaussian_eval/drjohnson --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 7000 -s ../autodl-tmp/tandt_db/db/playroom -m ../autodl-tmp/eval/split_gaussian_eval/playroom --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian
python render_new.py --iteration 40000 -s ../autodl-tmp/tandt_db/db/playroom -m ../autodl-tmp/eval/split_gaussian_eval/playroom --quiet --eval --skip_train --model_type split_gaussian --render_dir split_gaussian

# 计算评估指标
python metrics_debug.py -m "../autodl-tmp/eval/split_gaussian_eval/bicycle" "../autodl-tmp/eval/split_gaussian_eval/flowers" "../autodl-tmp/eval/split_gaussian_eval/garden" "../autodl-tmp/eval/split_gaussian_eval/stump" "../autodl-tmp/eval/split_gaussian_eval/treehill" "../autodl-tmp/eval/split_gaussian_eval/room" "../autodl-tmp/eval/split_gaussian_eval/counter" "../autodl-tmp/eval/split_gaussian_eval/kitchen" "../autodl-tmp/eval/split_gaussian_eval/bonsai" "../autodl-tmp/eval/split_gaussian_eval/truck" "../autodl-tmp/eval/split_gaussian_eval/train" "../autodl-tmp/eval/split_gaussian_eval/drjohnson" "../autodl-tmp/eval/split_gaussian_eval/playroom" --render_name split_gaussian
#python metrics_debug.py -m "../autodl-tmp/eval/split_gaussian_eval/playroom" --render_name split_gaussian