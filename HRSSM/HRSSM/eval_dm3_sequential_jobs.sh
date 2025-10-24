#!/bin/bash

# Check if a port argument is provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <carla_port> <gpu_device> [additional_training_parameters]"
    exit 1
fi

CARLA_PORT=$1
GPU_DEVICE=$2
CHECKPOINT_PATH=$3
MODE=$4
TASK=$5

DIR_NAME=$(basename "$(dirname "$CHECKPOINT_PATH")")
LOG_FILE="logdir/evals_finegrained/eval_log_${CARLA_PORT}_${MODE}.log"
LOG_DIR="logdir/evals_finegrained/${DIR_NAME}_${TASK}_${MODE}"

CARLA_SERVER_COMMAND="$CARLA_ROOT/CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT -benchmark -fps=10"
TRAINING_SCRIPT="dreamer.py"
COMMON_PARAMS="--checkpoint $CHECKPOINT_PATH --mode $MODE --logdir $LOG_DIR --task $TASK --configs carla"
TRAINING_COMMAND="python -u $TRAINING_SCRIPT $COMMON_PARAMS"

> $LOG_FILE  # Clear log

log_with_timestamp() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> $LOG_FILE
}

launch_carla() {
    if ! pgrep -f "CarlaUE4.sh -RenderOffScreen -carla-port=$CARLA_PORT" > /dev/null; then
        log_with_timestamp "CARLA not running on port $CARLA_PORT. Starting..."
        fuser -k ${CARLA_PORT}/tcp

        CUDA_VISIBLE_DEVICES=$GPU_DEVICE $CARLA_SERVER_COMMAND &
        CARLA_PID=$!

        MAX_WAIT_TIME=30  # seconds
        WAIT_INTERVAL=2
        ELAPSED=0

        while ! nc -z localhost $CARLA_PORT; do
            if (( ELAPSED >= MAX_WAIT_TIME )); then
                log_with_timestamp "ERROR: CARLA failed to start within ${MAX_WAIT_TIME}s. Giving up."
                kill -TERM $CARLA_PID >/dev/null 2>&1
                wait $CARLA_PID >/dev/null 2>&1
                cleanup
                exit 1
            fi
            log_with_timestamp "Waiting for CARLA on port $CARLA_PORT... (${ELAPSED}s elapsed)"
            sleep $WAIT_INTERVAL
            (( ELAPSED += WAIT_INTERVAL ))
        done

        log_with_timestamp "CARLA running on port $CARLA_PORT (started in ${ELAPSED}s)."
    fi
}

cleanup() {
    log_with_timestamp "Cleaning up and releasing resources..."
    fuser -k ${CARLA_PORT}/tcp
    if [ -n "$TRAINING_PID" ]; then
        kill -TERM $TRAINING_PID >/dev/null 2>&1
        wait $TRAINING_PID >/dev/null 2>&1
    fi
    log_with_timestamp "Cleanup done. Exiting job."
    # If under SLURM, optionally release the job:
    if [ -n "$SLURM_JOB_ID" ]; then
        log_with_timestamp "Releasing SLURM job $SLURM_JOB_ID."
        scancel $SLURM_JOB_ID
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

log_with_timestamp "Starting training on port $CARLA_PORT..."
launch_carla

# Start training (blocking)
log_with_timestamp "Running command: $TRAINING_COMMAND"
CUDA_VISIBLE_DEVICES=$GPU_DEVICE $TRAINING_COMMAND >> $LOG_FILE 2>&1
TRAIN_EXIT_CODE=$?

# Cleanup and release job after Python finishes
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    log_with_timestamp "Training completed successfully. Cleaning up..."
else
    log_with_timestamp "Training exited with code $TRAIN_EXIT_CODE. Cleaning up..."
fi

cleanup
