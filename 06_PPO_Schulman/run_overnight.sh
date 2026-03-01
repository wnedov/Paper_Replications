#!/bin/bash
# run_overnight.sh — Train PPO on an Atari and a continuous env overnight.
# Results are logged to runs/ for TensorBoard.
#
# Usage:  ./run_overnight.sh
#         tensorboard --logdir runs/    (in another terminal to monitor)

if [ -z "$CAFFEINATED" ]; then
    export CAFFEINATED=1
    exec caffeinate -i "$0" "$@"
fi

set -e
cd "$(dirname "$0")"

ATARI_ENV="ALE/Breakout-v5"
ATARI_STEPS=10000000
CTS_ENV="HalfCheetah-v5"
CTS_STEPS=1000000

echo "=== PPO Overnight Training ==="
echo "Started at $(date)"
echo "Monitor:  tensorboard --logdir runs/"
echo ""

echo "[1/2] Atari  — ${ATARI_ENV} (${ATARI_STEPS} steps)"
python main.py \
    --env-type atari \
    --env-id "$ATARI_ENV" \
    --run-name breakout_overnight \
    --total-timesteps $ATARI_STEPS &
PID1=$!

echo "[2/2] Continuous — ${CTS_ENV} (${CTS_STEPS} steps)"
python main.py \
    --env-type continuous \
    --env-id "$CTS_ENV" \
    --run-name halfcheetah_overnight \
    --total-timesteps $CTS_STEPS &
PID2=$!

echo ""
echo "PIDs: Atari=$PID1  Continuous=$PID2"
echo "Waiting for both to finish..."

wait $PID1 $PID2

echo ""
echo "=== All runs complete at $(date) ==="
