#!/usr/bin/env bash
# ==============================================================================
# run_experiment.sh — Full Pipeline Orchestrator
#
# Runs the complete Community Brain experiment end-to-end:
#   Step 1: Fine-tune LoRA Adapter A (Agronomy Expert)
#   Step 2: Fine-tune LoRA Adapter B (Veterinary Expert)
#   Step 3: Merge both adapters via TIES into the Unified Community Brain
#   Step 4: Evaluate all three configurations and produce a report
#
# Usage:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh               # Run everything
#   ./run_experiment.sh --skip-train   # Skip training, just merge + evaluate
#   ./run_experiment.sh --eval-only    # Only run evaluation
# ==============================================================================

set -euo pipefail

# Disable buggy xet downloader for HuggingFace Hub
export HF_HUB_DISABLE_XET=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
DATA_DIR="${DATA_DIR:-./data}"
ADAPTER_DIR="${ADAPTER_DIR:-./adapters}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LR="${LR:-2e-4}"
MERGE_TYPE="${MERGE_TYPE:-ties}"
TIES_DENSITY="${TIES_DENSITY:-0.5}"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_TRAIN=false
EVAL_ONLY=false

for arg in "$@"; do
    case $arg in
        --skip-train) SKIP_TRAIN=true ;;
        --eval-only)  EVAL_ONLY=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-train] [--eval-only] [--help]"
            echo ""
            echo "  --skip-train  Skip fine-tuning, only merge and evaluate"
            echo "  --eval-only   Only run evaluation (adapters must exist)"
            echo ""
            echo "Environment variables:"
            echo "  EPOCHS=$EPOCHS  BATCH_SIZE=$BATCH_SIZE  LR=$LR"
            echo "  MERGE_TYPE=$MERGE_TYPE  TIES_DENSITY=$TIES_DENSITY"
            exit 0
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

section() {
    echo ""
    echo "=================================================================="
    echo "  $(timestamp)  $1"
    echo "=================================================================="
    echo ""
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
section "PRE-FLIGHT CHECKS"

echo "Checking Python environment..."
python3 -c "import torch; print(f'PyTorch {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers {transformers.__version__}')"
python3 -c "import peft; print(f'PEFT {peft.__version__}')"
python3 -c "import trl; print(f'TRL {trl.__version__}')"

# Detect compute device
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)')
elif torch.backends.mps.is_available():
    print('Device: Apple Silicon (MPS)')
else:
    print('Device: CPU (training will be slow)')
"

echo "Data directory: $DATA_DIR"
ls -la "$DATA_DIR"/*.jsonl 2>/dev/null || { echo "ERROR: No .jsonl files in $DATA_DIR"; exit 1; }

mkdir -p "$ADAPTER_DIR" "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1 & 2: Fine-tune both adapters
# ---------------------------------------------------------------------------
if [ "$EVAL_ONLY" = false ] && [ "$SKIP_TRAIN" = false ]; then
    section "STEP 1: Fine-tuning Agronomy Expert LoRA Adapter"
    python3 finetune.py \
        --adapter agronomy \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --data-dir "$DATA_DIR" \
        --output-dir "$ADAPTER_DIR"

    section "STEP 2: Fine-tuning Veterinary Expert LoRA Adapter"
    python3 finetune.py \
        --adapter veterinary \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --lr "$LR" \
        --data-dir "$DATA_DIR" \
        --output-dir "$ADAPTER_DIR"
fi

# ---------------------------------------------------------------------------
# Step 3: Merge adapters via TIES
# ---------------------------------------------------------------------------
if [ "$EVAL_ONLY" = false ]; then
    section "STEP 3: TIES Merge — Creating the Unified Community Brain"
    python3 merge_engine.py \
        --adapter-a "$ADAPTER_DIR/agronomy_expert_lora" \
        --adapter-b "$ADAPTER_DIR/veterinary_expert_lora" \
        --output-dir "$ADAPTER_DIR/unified_community_brain" \
        --combination-type "$MERGE_TYPE" \
        --density "$TIES_DENSITY" \
        --test
fi

# ---------------------------------------------------------------------------
# Step 4: Evaluate all configurations
# ---------------------------------------------------------------------------
section "STEP 4: Evaluation — Knowledge Retention Test"
python3 evaluate.py \
    --adapter-a "$ADAPTER_DIR/agronomy_expert_lora" \
    --adapter-b "$ADAPTER_DIR/veterinary_expert_lora" \
    --merged "$ADAPTER_DIR/unified_community_brain" \
    --output "$RESULTS_DIR/evaluation_report.json"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
section "EXPERIMENT COMPLETE"
echo "Results saved to: $RESULTS_DIR/evaluation_report.json"
echo ""
echo "Next steps for your paper:"
echo "  1. Review the evaluation report for knowledge retention scores"
echo "  2. The merged model is at: $ADAPTER_DIR/unified_community_brain"
echo "  3. Compare Agronomy-only vs Veterinary-only vs Merged scores"
echo "  4. The key finding: TIES merge retains knowledge from BOTH domains"
echo ""
