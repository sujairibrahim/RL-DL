#!/bin/bash
# ================================================================
#  REMARL Pipeline Status Monitor
# ================================================================
#  Usage:
#    bash check_status.sh           → full overview
#    bash check_status.sh tail      → tail the currently active log
#    bash check_status.sh train     → tail training log
#    bash check_status.sh paired    → tail paired eval log
#    bash check_status.sh results   → print final results
# ================================================================

# ── Find current run ─────────────────────────────────────────────
CURRENT_LINK="data/logs/current"
PID_FILE="data/logs/pipeline.pid"

if [ ! -L "${CURRENT_LINK}" ] && [ ! -d "${CURRENT_LINK}" ]; then
    echo ""
    echo "  No pipeline run found."
    echo "  Start one with:  bash run_full_pipeline.sh"
    echo ""
    exit 1
fi

LOG_DIR=$(readlink -f "${CURRENT_LINK}" 2>/dev/null || echo "${CURRENT_LINK}")
MASTER="${LOG_DIR}/master.log"

# ── Helper: horizontal bar ────────────────────────────────────────
bar() { printf '  %s\n' "────────────────────────────────────────────────────"; }
bigbar() { printf '  %s\n' "════════════════════════════════════════════════════"; }

# ── Sub-command: tail a specific log ─────────────────────────────
if [ "${1:-}" = "train" ]; then
    echo ""; echo "  [tail] Training log — Ctrl-C to stop"; bar
    tail -f "${LOG_DIR}/train.log" 2>/dev/null || echo "  train.log not found yet."
    exit 0
fi
if [ "${1:-}" = "paired" ]; then
    echo ""; echo "  [tail] Paired eval log — Ctrl-C to stop"; bar
    tail -f "${LOG_DIR}/paired_eval.log" 2>/dev/null || echo "  paired_eval.log not found yet."
    exit 0
fi
if [ "${1:-}" = "results" ]; then
    echo ""
    for f in analysis_remarl_vs_baseline analysis_remarl_vs_random mare_style_eval; do
        logf="${LOG_DIR}/${f}.log"
        if [ -f "${logf}" ]; then
            echo ""; bigbar; echo "  ${f}"; bigbar
            cat "${logf}"
        fi
    done
    exit 0
fi

# ── Main status overview ──────────────────────────────────────────
echo ""
bigbar
echo "  REMARL Pipeline Status"
bigbar

# PID / running check
if [ -f "${PID_FILE}" ]; then
    PID=$(cat "${PID_FILE}")
    if kill -0 "${PID}" 2>/dev/null; then
        echo "  Status   : ▶  RUNNING  (PID ${PID})"
    else
        # Check if master log says complete
        if grep -q "PIPELINE COMPLETE" "${MASTER}" 2>/dev/null; then
            echo "  Status   : ✓  COMPLETE"
        else
            echo "  Status   : ✗  STOPPED / FAILED  (PID ${PID} not running)"
        fi
    fi
else
    echo "  Status   : ?  Unknown (no PID file)"
fi

echo "  Log dir  : ${LOG_DIR}"
echo ""

# ── Progress from master log ──────────────────────────────────────
if [ -f "${MASTER}" ]; then
    bar
    echo "  Progress"
    bar
    # Show phase banners + start/done/failed lines
    grep -E "(PHASE|▶ START|✓ DONE|✗ FAILED|PIPELINE COMPLETE|Estimated|TOTAL)" \
        "${MASTER}" 2>/dev/null \
        | sed 's/\[.*\] /  /' \
        | head -60
    echo ""

    # Count summary
    DONE=$(grep -c "✓ DONE"  "${MASTER}" 2>/dev/null || true)
    FAIL=$(grep -c "✗ FAILED" "${MASTER}" 2>/dev/null || true)
    DONE=${DONE:-0}; FAIL=${FAIL:-0}
    bar
    echo "  Stages done: ${DONE}   Failed: ${FAIL}"
fi

# ── Last 3 lines from master (most recent activity) ───────────────
echo ""
bar
echo "  Most recent log entry"
bar
tail -5 "${MASTER}" 2>/dev/null | sed 's/^/  /'

# ── Available log files ───────────────────────────────────────────
echo ""
bar
echo "  Log files  (size | lines | last-modified)"
bar
for f in \
    train.log \
    eval_collector.log \
    eval_modeler.log \
    eval_checker.log \
    paired_eval.log \
    analysis_remarl_vs_baseline.log \
    analysis_remarl_vs_random.log \
    analysis_random_vs_baseline.log \
    mare_style_eval.log; do

    fpath="${LOG_DIR}/${f}"
    if [ -f "${fpath}" ]; then
        sz=$(du -h "${fpath}" | cut -f1)
        ln=$(wc -l < "${fpath}")
        mod=$(date -r "${fpath}" '+%H:%M:%S' 2>/dev/null || stat -c '%y' "${fpath}" 2>/dev/null | cut -c12-19)
        # Mark active (modified in last 90 seconds)
        now=$(date +%s)
        mtime=$(date -r "${fpath}" +%s 2>/dev/null || stat -c '%Y' "${fpath}" 2>/dev/null)
        age=$(( now - ${mtime:-0} ))
        marker="   "
        [ "${age}" -lt 90 ] && marker="◀ ACTIVE"
        printf "  %-42s  %5s  %5d lines  %s  %s\n" \
            "${f}" "${sz}" "${ln}" "${mod}" "${marker}"
    else
        printf "  %-42s  (not started)\n" "${f}"
    fi
done

# ── Paired eval JSON ─────────────────────────────────────────────
PAIRED_TXT="${LOG_DIR}/paired_eval_json.txt"
if [ -f "${PAIRED_TXT}" ]; then
    PJSON=$(cat "${PAIRED_TXT}")
    echo ""
    bar
    echo "  Paired eval JSON"
    bar
    echo "  ${PJSON}"
fi

# ── Commands reference ────────────────────────────────────────────
echo ""
bigbar
echo "  Commands"
bigbar
echo "  Watch training live:"
echo "    tail -f ${LOG_DIR}/train.log"
echo ""
echo "  Watch paired eval live:"
echo "    tail -f ${LOG_DIR}/paired_eval.log"
echo ""
echo "  Print final results:"
echo "    bash check_status.sh results"
echo ""
echo "  Stop the pipeline:"
echo "    bash stop_pipeline.sh"
echo ""
echo "  Master log:"
echo "    cat ${MASTER}"
bigbar
echo ""