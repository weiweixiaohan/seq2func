# DeepSEA 课程报告实验

本项目基于 PyTorch 版 DeepSEA，设计了两组实验用于深度学习课程报告。

## 实验设计

### 方向 5：序列上下文长度对性能的影响

**核心问题**：模型需要多长的 DNA 序列上下文？原始的 1000bp 是最优的吗？

| 实验 | `--seq_len` | 预期感受野 | 说明 |
|------|-------------|------------|------|
| Short | 200 | ~200 bp | 仅覆盖核心 motif 区域 |
| Medium | 500 | ~500 bp | 中等上下文 |
| Full (原版) | 1000 | ~1000 bp | 全上下文 |

**可期待的分析角度**：
- 不同长度下三类特征（DNase/TF/Histone）的 AUC 差异
- 感受野大小对各特征类型的重要性（TF 可能需要较短上下文，组蛋白可能需要较长上下文）
- 参数量的变化（越短序列 → FC 层越小 → 参数量越少）

### 方向 4：架构改进 —  Attention

**核心问题**：2015 年的纯 CNN 架构，加入现代设计能提升吗？

| 实验 | `--variant` | 改动 |
|------|-------------|------|
| Original | `original` | 3 层纯 CNN + FC |
| +Attention | `attention` | Conv3 后加多头自注意力 + 位置编码 |

## 目录结构

```
deepsea_report/
├── model.py              # 模型（支持 --seq_len 和 --variant）
├── dataset.py            # 数据加载（支持中心裁剪）
├── train.py              # 训练脚本
├── eval.py               # 评估脚本
├── compute_aucs.py       # 逐特征 AUC 计算
├── run_experiments.sh    # 批量实验运行脚本
├── README.md
├── checkpoints/          # 模型 checkpoint（按实验分目录）
│   ├── seq200_original/
│   ├── seq500_original/
│   ├── seq1000_original/
│   └── seq1000_attention/
└── eval_results/         # 评估结果
```

## 快速开始

### 1. 确保数据已准备

```bash
# 如果还没下载数据，先到 pytorch_deepsea 目录执行
cd ../pytorch_deepsea
python build_data.py
cd ../deepsea_report
```

### 2. 单个实验运行

```bash
# 训练一个序列长度实验
python train.py --seq_len 200 --variant original

# 训练一个架构实验
python train.py --seq_len 1000 --variant residual

# 监控评估（另开终端）
python eval.py --seq_len 200 --variant original

# 测试集评估
python eval.py --seq_len 200 --variant original \
    --split test --run_once --save_predictions --global_step 100000

# 计算 AUC
python compute_aucs.py --seq_len 200 --variant original --global_step 100000
```

### 3. 批量运行所有实验

```bash
chmod +x run_experiments.sh

# 依次训练所有 6 组实验（3 长度 + 3 架构）
./run_experiments.sh train

# 评估所有实验（替换 <step> 为实际 checkpoint 步数）
./run_experiments.sh eval <step>

# 计算所有实验的逐特征 AUC
./run_experiments.sh auc <step>
```

如果只想快速预览（减少训练时间），可以限制 epoch：

```bash
MAX_EPOCHS=10 ./run_experiments.sh train
```

### 4. 用 TensorBoard 对比实验

```bash
tensorboard --logdir=checkpoints/tensorboard
```

所有实验的曲线在同一个 TensorBoard 中，可直接对比 loss 和 learning rate。
