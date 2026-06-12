#!/bin/bash
# ============================================================
# DeepSEA Report — Batch Experiment Runner
#
# 一键运行多个 (seq_len × variant) 组合的实验。
# 根据需要修改下面的 COMBO 列表。
#
# 用法:
#   chmod +x run_experiments.sh
#   ./run_experiments.sh train          # 依次训练所有组合
#   ./run_experiments.sh eval           # 评估所有已完成训练的 checkpoint
#   ./run_experiments.sh auc STEP       # 计算所有组合在 test 集的 AUC
# ============================================================

set -e

DEVICE="${DEVICE:-cuda}"          # 默认 GPU，CPU 训练用: DEVICE=cpu ./run_experiments.sh train
MAX_EPOCHS="${MAX_EPOCHS:-100}"   # 最大 epoch 数（方向5 可减半以加快实验）
MAX_STEPS="${MAX_STEPS:-0}"       # 0 = 由 max_epochs 决定

DATA_DIR="${DATA_DIR:-../pytorch_deepsea/data}"
TRAIN_DIR="${TRAIN_DIR:-checkpoints}"
EVAL_DIR="${EVAL_DIR:-eval_results}"

has_ckpts() {
    local tag="$1"
    local dir="$TRAIN_DIR/$tag"
    if [ -d "$dir" ] && ls "$dir"/ckpt-*.pth >/dev/null 2>&1; then
        return 0
    fi
    return 1
}

# ---- 实验组合定义 ----
# 方向5：序列长度实验
SEQ_LEN_EXPS=(
    "200"
    "500"
    "1000"
)

# 方向4：架构变体实验（统一用 1000bp）
ARCH_EXPS=(
    "original"
    "residual"
    "attention"
)

# ---- 解析子命令 ----
CMD="${1:-train}"
GSTEP="${2:-}"

case "$CMD" in
    train)
        echo "=========================================="
        echo " Phase 1: Sequence Length Experiments"
        echo "=========================================="
        for sl in "${SEQ_LEN_EXPS[@]}"; do
            echo ""
            echo ">>> Training: seq_len=$sl, variant=original <<<"
            if [ "$sl" = "1000" ] && has_ckpts "seq1000_original"; then
                echo "Skip: seq1000_original already has checkpoints"
                continue
            fi
            python train.py \
                --data_dir "$DATA_DIR" \
                --train_dir "$TRAIN_DIR" \
                --seq_len "$sl" \
                --variant original \
                --device "$DEVICE" \
                --max_epochs "$MAX_EPOCHS" \
                --max_steps "$MAX_STEPS"
        done

        echo ""
        echo "=========================================="
        echo " Phase 2: Architecture Variant Experiments"
        echo "=========================================="
        for va in "${ARCH_EXPS[@]}"; do
            echo ""
            echo ">>> Training: seq_len=1000, variant=$va <<<"
            if [ "$va" = "original" ] && has_ckpts "seq1000_original"; then
                echo "Skip: seq1000_original already has checkpoints"
                continue
            fi
            python train.py \
                --data_dir "$DATA_DIR" \
                --train_dir "$TRAIN_DIR" \
                --seq_len 1000 \
                --variant "$va" \
                --device "$DEVICE" \
                --max_epochs "$MAX_EPOCHS" \
                --max_steps "$MAX_STEPS"
        done
        echo ""
        echo "All training experiments completed!"
        ;;

    eval)
        echo "Running evaluation for all combination folders..."
        for combo_dir in "$TRAIN_DIR"/seq*_*; do
            [ -d "$combo_dir" ] || continue
            tag=$(basename "$combo_dir")
            sl=${tag#seq}; sl=${sl%%_*}
            va=${tag#*_}
            echo ""
            echo ">>> Eval: $tag <<<"
            python eval.py \
                --data_dir "$DATA_DIR" \
                --train_dir "$TRAIN_DIR" \
                --eval_dir "$EVAL_DIR" \
                --seq_len "$sl" \
                --variant "$va" \
                --device "$DEVICE" \
                --split test \
                --run_once \
                --report_progress \
                --save_predictions \
                --global_step "$GSTEP"
        done
        ;;

    auc)
        if [ -z "$GSTEP" ]; then
            echo "Usage: ./run_experiments.sh auc <global_step>"
            exit 1
        fi
        for combo_dir in "$TRAIN