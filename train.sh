#!/bin/bash

# 三阶段训练启动脚本
# 使用方法: ./train.sh <数据路径> [其他参数]

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <data_path> [additional_args]"
    echo "Example: $0 data/nerf/lego --eval"
    exit 1
fi

DATA_PATH=$1
shift  # 移除第一个参数，保留其余参数

# 设置输出路径
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_PATH="output/split_gaussian_${TIMESTAMP}"

echo "=========================================="
echo "Three-Stage Split Gaussian Training"
echo "=========================================="
echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_PATH}"
echo "=========================================="

# 基础参数
BASE_ARGS="--source_path ${DATA_PATH} \
           --model_path ${OUTPUT_PATH} \
           --iterations 40000 \
           --eval"

# 优化参数（三阶段策略）
OPT_ARGS="--densify_from_iter 500 \
          --densify_until_iter 15000 \
          --densify_grad_threshold 0.0002 \
          --densification_interval 100 \
          --opacity_reset_interval 3000 \
          --lambda_dssim 0.2 \
          --percent_dense 0.01"

# Wave相关参数
WAVE_ARGS="--wave_lr 0.01 \
           --max_splits 5 \
           --lambda_wave_reg 0.01"

# 测试和保存参数
TEST_ARGS="--test_iterations 7000 15000 30000 \
           --save_iterations 7000 15000 30000 40000 \
           --checkpoint_iterations 15000 30000"

# 合并所有参数
ALL_ARGS="${BASE_ARGS} ${OPT_ARGS} ${WAVE_ARGS} ${TEST_ARGS} $@"

# 打印最终命令
echo "Running command:"
echo "python train_split_gaussian_improved.py ${ALL_ARGS}"
echo "=========================================="

# 执行训练
python train_split_gaussian_improved.py ${ALL_ARGS}

# 训练完成后的处理
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Results saved to: ${OUTPUT_PATH}"
    
    # 可选：运行评估
    read -p "Run evaluation? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Running evaluation..."
        python render.py --model_path ${OUTPUT_PATH} --skip_train
        python metrics.py --model_path ${OUTPUT_PATH}
    fi
else
    echo "Training failed!"
    exit 1
fi