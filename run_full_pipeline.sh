#!/bin/bash
# ================================================================
#  REMARL Full Pipeline — Train → Benchmark → Paired Eval → Stats
# ================================================================
#
#  LAUNCH (runs in background, returns your terminal immediately):
#
#    bash run_full_pipeline.sh
#
#  MONITOR while it runs:
#
#    bash check_status.sh           # overall progress
#    tail -f data/logs/current/train.log          # watch training
#    tail -f data/logs/current/paired_eval.log    # watch paired eval
#
#  STOP if needed:
#
#    bash stop_pipeline.sh
# ================================================================

# ── Settings ────────────────────────────────────────────────────
CONFIG="configs/remarl_train_fast.yaml"
CKPT_DIR="data/checkpoints"
N_EVAL=20      # episodes per role in quick benchmark
N_PAIRED=22    # scenarios for rigorous paired eval
SEED=42
ROLES=("collector" "modeler" "checker")

# ── Directories ──────────────────────────────────────────────────
RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="data/logs/run_${RUN_ID}"
CURRENT_LINK="data/logs/current"     # symlink always points to latest run

mkdir -p "${LOG_DIR}"
mkdir -p "${CKPT_DIR}"
mkdir -p "data/benchmarks"
mkdir -p "data/logs"

# Symlink so check_status.sh always finds latest run without knowing timestamp
ln -sfn "run_${RUN_ID}" "${CURRENT_LINK}"
echo "${LOG_DIR}" > data/logs/latest_run_dir.txt

# ── Logging helpers ───────────────────────────────────────────────
MASTER="${LOG_DIR}/master.log"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log() {
    echo "[$(ts)] $*" | tee -a "${MASTER}"
}

log_banner() {
    local bar="══════════════════════════════════════════════════════"
    {
        echo ""
        echo "${bar}"
        printf "  %s\n" "$*"
        echo "${bar}"
    } | tee -a "${MASTER}"
}

# Record start time for elapsed-time reporting
stage_start=0
start_stage() {
    stage_start=$(date +%s)
    log "▶ START   $1"
}

end_stage() {
    local name="$1" exit_code="$2"
    local elapsed=$(( $(date +%s) - stage_start ))
    local mm=$(( elapsed / 60 ))
    local ss=$(( elapsed % 60 ))
    if [ "${exit_code}" -eq 0 ]; then
        log "✓ DONE    ${name}  (${mm}m ${ss}s)"
    else
        log "✗ FAILED  ${name}  (${mm}m ${ss}s, exit ${exit_code})"
    fi
}

# run_stage <display_name> <logfile_basename> <command...>
# Streams command output to <LOG_DIR>/<logfile_basename>
# Returns the command's exit code
run_stage() {
    local name="$1"
    local logfile="${LOG_DIR}/$2"
    shift 2

    start_stage "${name}"

    # Write the exact command to the top of the log so it's reproducible
    {
        echo "# ── $(ts) ──────────────────────────────────────────"
        echo "# CMD: $*"
        echo "# LOG: ${logfile}"
        echo "# ────────────────────────────────────────────────────"
        echo ""
    } >> "${logfile}"

    # Run — stdout+stderr both go to the stage log file
    "$@" >> "${logfile}" 2>&1
    local exit_code=$?

    end_stage "${name}" "${exit_code}"
    return ${exit_code}
}

# ── Self-background: re-launch under nohup if not already ────────
#   First call: PIPELINE_STARTED is unset → launch under nohup → exit
#   Second call (the nohup'd one): PIPELINE_STARTED=1 → continue
if [ -z "${PIPELINE_STARTED:-}" ]; then
    echo ""
    echo "  Launching pipeline in background…"
    echo "  Log directory: ${LOG_DIR}"
    echo ""
    PIPELINE_STARTED=1 \
        nohup bash "$0" >> "${LOG_DIR}/nohup_stdout.log" 2>&1 &
    BGPID=$!
    echo "${BGPID}" > data/logs/pipeline.pid
    echo "  PID: ${BGPID}"
    echo "  Symlink: ${CURRENT_LINK}/ → ${LOG_DIR}/"
    echo ""
    echo "  Monitor:"
    echo "    bash check_status.sh"
    echo "    tail -f ${CURRENT_LINK}/train.log"
    echo ""
    exit 0
fi

# ── Pipeline body (runs inside nohup) ────────────────────────────
echo $$ > data/logs/pipeline.pid   # update to the real background PID

log_banner "REMARL PIPELINE  run_id=${RUN_ID}"
log "Config   : ${CONFIG}"
log "Log dir  : ${LOG_DIR}"
log "PID      : $$"
log "N_eval   : ${N_EVAL}  per role"
log "N_paired : ${N_PAIRED}  scenarios"
log ""
log "Estimated time:"
log "  Phase 1  Training all 3 roles  → ~2.5 h  (fast config)"
log "  Phase 2  Quick benchmark       → ~1 h    (${N_EVAL} eps × 3 roles)"
log "  Phase 3  Paired eval           → ~3 h    (${N_PAIRED} scenarios × 3 policies)"
log "  Phase 4  Statistical analysis  → ~5 min"
log "  Phase 5  MARE-style P/R/F1     → ~5 min"
log "  ─────────────────────────────────────────"
log "  TOTAL                          → ~7 h"

# ════════════════════════════════════════════════════════════════
#  PHASE 1 — TRAINING
# ════════════════════════════════════════════════════════════════
log_banner "PHASE 1 — TRAINING  (all 3 roles, fast config)"
log "Trainer output: tail -f ${LOG_DIR}/train.log"

if ! run_stage \
        "train --role all" \
        "train.log" \
        python train.py --role all --config "${CONFIG}"
then
    log "Training failed. Aborting pipeline — check ${LOG_DIR}/train.log"
    exit 1
fi

# ════════════════════════════════════════════════════════════════
#  PHASE 2 — QUICK BENCHMARK  (one evaluate.py call per role)
# ════════════════════════════════════════════════════════════════
log_banner "PHASE 2 — QUICK BENCHMARK  (${N_EVAL} episodes per role)"

for role in "${ROLES[@]}"; do
    ckpt="${CKPT_DIR}/${role}_final"
    if [ ! -f "${ckpt}.zip" ]; then
        log "  WARNING: ${ckpt}.zip not found — skipping ${role}"
        continue
    fi
    log "  Evaluating ${role} → tail -f ${LOG_DIR}/eval_${role}.log"

    run_stage \
        "benchmark ${role}" \
        "eval_${role}.log" \
        python evaluate.py \
            --checkpoint "${ckpt}" \
            --config     "${CONFIG}" \
            --role       "${role}" \
            --n_eval     "${N_EVAL}" \
    || log "  WARNING: benchmark for ${role} failed (continuing)"
done

# ════════════════════════════════════════════════════════════════
#  PHASE 3 — RIGOROUS PAIRED EVALUATION  (collector)
# ════════════════════════════════════════════════════════════════
log_banner "PHASE 3 — PAIRED EVAL  (${N_PAIRED} scenarios × 3 policies)"

COLL_CKPT="${CKPT_DIR}/collector_final.zip"
if [ ! -f "${COLL_CKPT}" ]; then
    log "ERROR: ${COLL_CKPT} not found. Aborting."
    exit 1
fi
log "Paired eval output: tail -f ${LOG_DIR}/paired_eval.log"

if ! run_stage \
        "paired eval (collector, n=${N_PAIRED})" \
        "paired_eval.log" \
        python eval/run_paired_eval.py \
            --checkpoint "${COLL_CKPT}" \
            --config     "${CONFIG}" \
            --role       collector \
            --n          "${N_PAIRED}" \
            --seed       "${SEED}"
then
    log "Paired eval failed. Aborting — check ${LOG_DIR}/paired_eval.log"
    exit 1
fi

# Find the JSON that was just written
PAIRED_JSON=$(ls -t data/benchmarks/paired_eval_*.json 2>/dev/null | head -1)
if [ -z "${PAIRED_JSON}" ]; then
    log "ERROR: Could not find paired_eval JSON in data/benchmarks/ — aborting"
    exit 1
fi
log "Paired eval JSON: ${PAIRED_JSON}"
echo "${PAIRED_JSON}" > "${LOG_DIR}/paired_eval_json.txt"

# ════════════════════════════════════════════════════════════════
#  PHASE 4 — STATISTICAL ANALYSIS
# ════════════════════════════════════════════════════════════════
log_banner "PHASE 4 — STATISTICAL ANALYSIS"

for comparison in remarl_vs_baseline remarl_vs_random random_vs_baseline; do
    run_stage \
        "analyze: ${comparison}" \
        "analysis_${comparison}.log" \
        python eval/analyze_paired.py \
            --input      "${PAIRED_JSON}" \
            --comparison "${comparison}" \
    || log "  WARNING: analysis ${comparison} failed (continuing)"
done

# ════════════════════════════════════════════════════════════════
#  PHASE 5 — MARE-STYLE P / R / F1
# ════════════════════════════════════════════════════════════════
log_banner "PHASE 5 — MARE-STYLE P/R/F1"
log "MARE eval output: tail -f ${LOG_DIR}/mare_style_eval.log"

run_stage \
    "mare-style F1 evaluation" \
    "mare_style_eval.log" \
    python -m eval.mare_style_eval \
        --episodes "${PAIRED_JSON}" \
        --model    all-MiniLM-L6-v2 \
|| log "  WARNING: mare_style_eval failed (check log)"

# ════════════════════════════════════════════════════════════════
#  DONE
# ════════════════════════════════════════════════════════════════
log_banner "PIPELINE COMPLETE  run_id=${RUN_ID}"
log ""
log "All benchmark outputs:"
ls data/benchmarks/ 2>/dev/null | sed 's/^/    /' | tee -a "${MASTER}"
log ""
log "Log files:"
ls "${LOG_DIR}"/*.log 2>/dev/null | sed 's/^/    /' | tee -a "${MASTER}"
log ""
log "Quick results:"
log "  cat ${LOG_DIR}/analysis_remarl_vs_baseline.log"
log "  cat ${LOG_DIR}/mare_style_eval.log"
log ""
log "LaTeX table (if generated):"
ls data/benchmarks/*/table.tex 2>/dev/null | head -3 | sed 's/^/    /' | tee -a "${MASTER}"