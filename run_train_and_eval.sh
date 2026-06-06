#!/bin/bash
# Train all 3 roles then evaluate collector checkpoint.
# Uses fast training config (llama-3.2-3b, 50 episodes/role, ~4h total).

set -e
cd "$(dirname "$0")"
source renv/bin/activate 2>/dev/null || true

echo "========================================"
echo "  REMARL Training + Evaluation Pipeline"
echo "  $(date)"
echo "========================================"

# ── Phase 1: Train all 3 roles ────────────────────────────────
echo ""
echo "[1/2] Training collector, modeler, checker (50 eps each)..."
python train.py --role all --config configs/remarl_train_fast.yaml
echo "[1/2] Training complete: $(date)"

# ── Phase 2: Evaluate collector checkpoint ────────────────────
CKPT="data/checkpoints/collector_final"
if [ -f "${CKPT}.zip" ]; then
    echo ""
    echo "[2/2] Evaluating collector checkpoint (20 episodes)..."
    python evaluate.py \
        --checkpoint "$CKPT" \
        --config configs/remarl_train_fast.yaml \
        --n 20
    echo "[2/2] Evaluation complete: $(date)"
    echo ""
    echo "Benchmark outputs saved to data/benchmarks/"
    ls -lt data/benchmarks/ | head -5
else
    echo "WARNING: checkpoint not found at $CKPT — skipping eval"
fi

echo ""
echo "All done: $(date)"
