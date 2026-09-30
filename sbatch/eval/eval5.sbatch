#!/bin/bash
#SBATCH -J eval5
#SBATCH -p h01                 # h800 队列
#SBATCH -N 1                   # 单节点
#SBATCH --gres=gpu:4           # 申请 4 张 GPU
#SBATCH --ntasks-per-node=8    # CPU:GPU = 2:1（可改为4表示1:1）
#SBATCH -o stdout.%j
#SBATCH -e stderr.%j
#SBATCH --no-requeue
# #SBATCH -x g01               # 如需排除故障节点，取消注释
#SBATCH --chdir=/home/fit/lijuanzi1/WORK/dz/SAERL

set -euxo pipefail

# ===== NCCL / 线程（可选）=====
export NCCL_IB_HCA=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_6,mlx5_7,mlx5_8
export NCCL_DEBUG=INFO
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=2

# 只用 CUDA，禁用 ROCm 变量
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
unset HSA_VISIBLE_DEVICES

# ===== Ray：不连外部集群，走默认本地；临时目录放到大盘 =====
unset RAY_ADDRESS RAY_HEAD_IP RAY_HEAD_NODE RAY_CLIENT_MODE
export RAY_TMPDIR=/home/fit/lijuanzi1/WORK/dz/ray_tmp
mkdir -p "${RAY_TMPDIR}"

# vLLM 单机，不用 Ray
export VLLM_USE_RAY=0

# （可选）若不在启动脚本里激活，可在此激活环境：
source /home/fit/lijuanzi1/WORK/jy/miniconda3/etc/profile.d/conda.sh
conda activate verl_dz

# 启动脚本
chmod +x /home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/eval5.sh
bash /home/fit/lijuanzi1/WORK/dz/SAERL/evaluation/eval5.sh