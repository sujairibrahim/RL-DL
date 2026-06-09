#!/bin/bash
# ================================================================
#  Stop the REMARL pipeline gracefully
# ================================================================
#  Usage: bash stop_pipeline.sh
# ================================================================

PID_FILE="data/logs/pipeline.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "  No PID file found at ${PID_FILE}."
    echo "  Pipeline may not be running."
    exit 0
fi

PID=$(cat "${PID_FILE}")

if ! kill -0 "${PID}" 2>/dev/null; then
    echo "  PID ${PID} is not running — pipeline already stopped."
    exit 0
fi

echo ""
echo "  Stopping REMARL pipeline (PID ${PID})…"
echo "  Note: the current LLM call will finish before the process exits."
echo "  If you need to force-kill immediately: kill -9 ${PID}"
echo ""

# SIGTERM — lets Python clean up open files
kill "${PID}"

# Wait up to 30 seconds for graceful exit
for i in $(seq 1 30); do
    sleep 1
    if ! kill -0 "${PID}" 2>/dev/null; then
        echo "  Pipeline stopped cleanly."
        echo ""
        echo "  To resume from where it left off, re-run:"
        echo "    bash run_full_pipeline.sh"
        echo "  (already-completed stages will run again, but checkpoints are safe)"
        exit 0
    fi
    echo "  Waiting… (${i}/30)"
done

echo ""
echo "  Process did not stop within 30 s."
echo "  Force-kill with:  kill -9 ${PID}"