## ==================== Full dataset training pipeline ====================
## Suggested full training pipeline (Dense, 4x GPU)
# 阶段1：稠密模型，t2a文本转音频，从llm底座开始训练，学习率5e‑4，6轮完整数据集
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a.parquet --epochs 6 --batch_size 32 --use_compile 1 --from_weight llm --save_weight sft_omni --use_wandb --use_moe 0 # epochs * 45min
# 阶段2：a2a音频‑音频，只训练audio_proj音频投影层，接续sft_omni权重
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --mode audio_proj --use_wandb --use_moe 0 # epochs * 25min
# 阶段3：降低学习率5e‑5，a2a数据集全模型微调，不冻结投影层
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_a2a.parquet --epochs 3 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 0 # epochs * 25min
# 阶段4：i2t图像‑文本，只训练vision_proj图像投影层，接入图像能力
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 0 # epochs * 45min
# 阶段5：进一步降学习率5e‑6，i2t数据集全模型微调
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --use_wandb --use_moe 0 # epochs * 45min
# 阶段6：极小学习率，a2a音频数据集做精细微调
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 0 # epochs * 25min
# 阶段7：极小学习率，i2t图像数据集，仅训练图像投影层收尾
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 0 # epochs * 45min


## Suggested full training pipeline (MoE, 4x GPU)
# MoE版本仅把每条命令 --use_moe 0 修改为 --use_moe 1，整套流水线保持不变，权重不可交叉混用
# MoE混合专家版本，阶段1：t2a文本‑音频，从llm底座开始训练，开启use_moe=1
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_t2a.parquet --epochs 6 --batch_size 32 --use_compile 1 --from_weight llm --save_weight sft_omni --use_wandb --use_moe 1 # epochs * 45min
# MoE阶段2：a2a音频‑音频，只训练audio_proj音频投影层
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-4 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --mode audio_proj --use_wandb --use_moe 1 # epochs * 25min
# MoE阶段3：学习率5e‑5，a2a数据集全模型微调
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_a2a.parquet --epochs 3 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 1 # epochs * 25min
# MoE阶段4：i2t图像‑文本，只训练vision_proj图像投影层
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-5 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 1 # epochs * 45min
# MoE阶段5：降学习率5e‑6，i2t数据集全模型微调
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --use_wandb --use_moe 1 # epochs * 45min
# MoE阶段6：极小学习率，a2a音频数据集精细微调
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_a2a.parquet --epochs 1 --batch_size 32 --use_compile 0 --from_weight sft_omni --save_weight sft_omni --max_seq_len 1024 --use_wandb --use_moe 1 # epochs * 25min
# MoE阶段7：极小学习率，i2t图像数据集，图像投影层收尾训练
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port 29560 --nproc_per_node 4 train_sft_omni.py --learning_rate 5e-6 --data_path ../dataset/sft_i2t.parquet --epochs 1 --batch_size 32 --use_compile 1 --from_weight sft_omni --save_weight sft_omni --max_seq_len 768 --mode vision_proj --use_wandb --use_moe 1 # epochs * 45min


# ==================== (Recommend) Mini dataset training pipeline ====================
# 环境：单卡训练，示例耗时参考 1x RTX 3090，实际耗时受GPU性能影响仅供参考
# 【公共参数说明，从上到下顺序解读，仅解释一次】
# CUDA_VISIBLE_DEVICES=0                  # 指定可见GPU设备编号
# torchrun                                # PyTorch分布式训练启动器
# --master_port 29560                     # 分布式进程通信端口，端口冲突更换数字
# --nproc_per_node 1                      # 单机启动进程数，单卡固定写1
# train_sft_omni.py                       # SFT监督微调训练脚本入口
# --learning_rate                         # 模型训练学习率
# --data_path                             # 训练数据集parquet文件路径
# --epochs                                # 数据集完整遍历轮数
# --batch_size                            # 训练批次大小
# --use_compile 1/0                       # 开启/关闭torch.compile模型编译加速
# --from_weight                           # 加载的初始权重，接续训练时填上一轮输出权重名
# --save_weight                           # 本轮训练保存权重的名字
# --max_seq_len                           # 输入输出最大序列长度
# --use_wandb                             # 开启wandb训练日志可视化，不需要直接删掉该参数
# --use_moe 0                             # 0=Dense稠密模型；MoE整套命令改为--use_moe 1；稠密/MoE权重结构不互通，不可交叉接续训练
# --mode audio_proj                       # 仅训练音频投影层，主干大模型冻结；不写该参数代表全模型所有参数参与更新


# 阶段1：t2a 文本转音频，从llm底座开始训练，输出 sft_zero
CUDA_VISIBLE_DEVICES=0 torchrun \
--master_port 29560 \
--nproc_per_node 1 \
train_sft_omni.py \
--learning_rate 5e-4 \
--data_path ../dataset/sft_t2a_mini.parquet \
--epochs 1 \
--batch_size 40 \
--use_compile 1 \
--from_weight llm \
--save_weight sft_zero \
--max_seq_len 512 \
--use_wandb \
--use_moe 0
# epochs * 60min

# 阶段2：a2a 音频输入输出，训练audio_proj音频投影层，接续 sft_zero，输出覆盖 sft_zero
CUDA_VISIBLE_DEVICES=0 torchrun \
--master_port 29560 \
--nproc_per_node 1 \
train_sft_omni.py \
--learning_rate 5e-4 \
--data_path ../dataset/sft_a2a_mini.parquet \
--epochs 1 \
--batch_size 40 \
--use_compile 0 \
--from_weight sft_zero \
--save_weight sft_zero \
--max_seq_len 640 \
--mode audio_proj \
--use_wandb \
--use_moe 0
# epochs * 15min

# 阶段3：降低学习率，a2a数据集全模型微调，接续 sft_zero
CUDA_VISIBLE_DEVICES=0 torchrun \
--master_port 29560 \
--nproc_per_node 1 \
train_sft_omni.py \
--learning_rate 2e-5 \
--data_path ../dataset/sft_a2a_mini.parquet \
--epochs 1 \
--batch_size 16 \
--use_compile 0 \
--from_weight sft_zero \
--save_weight sft_zero \
--max_seq_len 768 \
--use_wandb \
--use_moe 0
# epochs * 15min