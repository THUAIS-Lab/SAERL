## 1. nvcc -V无反应

ls /usr/local/ | grep cuda

`彻底修改当前会话的路径`

export CUDA_HOME=/usr/local/cuda-12.5

export PATH=$CUDA_HOME/bin:$PATH

export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

`检查输出，确保显示的是 12.5 而不是 13.1`

nvcc -V

## 2. GLIBC_2.32 not found

IMPORT FAILED: ImportError /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.32' not found (required by /home/dz/miniconda3/envs/verl/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so)

<https://github.com/Dao-AILab/flash-attention/issues/1708>

pip install flash_attn==2.7.4.post1 --no-build-isolation

`检查`

python - << 'EOF'
try:
    import flash_attn
    print("flash_attn import OK")
except Exception as e:
    print("IMPORT FAILED:", type(e).__name__, e)
EOF

## 3.
\math数据集 parquet格式有7500个，但是json和pt文件都只有7474条，所以processed_file中的id与训练时使用的parquet不能严格对应，所以会影响训练。
启用了data.filter_overlong_prompts=True 后，代码会将parquet里数据filter，所以实际上的dataset和dataloader可能小于原本数据个数 而processed_file与原本数据又是对应的，所以可能出错

## 4.
deepscaler数据集的reward function用错了，现在复用了huawei实现的函数（v2之后）
检查aime amc的reward function记得

## 5.
生成的difficulty，相同难度之间排序要重新排

## 6.
gsm8k数据集要使用boxed，不然正确率极低
先不管gsm8k

## 7.
qwen3-1.7b有一个best practice, v2(including)之前都没有使用，difficulty计算也没有使用，
<!-- 从diffficulty baseline开始进行使用best practice -->

## 8.
rewardmanager
看abstract是啥用法，为啥就naive了

## 9.
"math8k", "GAIR/LIMR": same reward score func as math
deepscaler, nlile/NuminaMath-1.5-RL-Verifiable: same as deepmath