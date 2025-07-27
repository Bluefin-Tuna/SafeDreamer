#!/bin/bash

# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt"

# # Run modes to evaluate
# MODES=("gaussian_random_any" "none_random_any" "glare_random_any" "jitter_random_any" "occlusion_random_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE"
# done

# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointButton1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/pointbutton_1_geigh_3-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="pointbutton_1_geigh_3"

# # Run modes to evaluate
# MODES=("gaussian_random_any" "none_random_any" "glare_random_any" "jitter_random_any" "occlusion_random_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done

# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/temp_geigh_3-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="temp_geigh_3"

# # Run modes to evaluate
# MODES=("gaussian_random_any" "none_random_any" "glare_random_any" "jitter_random_any" "occlusion_random_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done

# Base command parameters
SCRIPT="python SafeDreamer/train.py"
CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0"
JAX="--jax.logical_gpus 0"
SCRIPT_MODE="--run.script eval_only"
CHECKPOINT="--run.from_checkpoint /home/general/logdir/temp_geigh_3_augment-safedreamer/checkpoint.ckpt"
LOGDIR_ALGO="temp_geigh_3_augment"

# Run modes to evaluate
MODES=("none_any")

# Loop over each mode
for MODE in "${MODES[@]}"; do
    echo "Evaluating mode: $MODE"
    $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
done

# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/cargoal_1_augment-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="cargoal_1_augment"

# # Run modes to evaluate
# MODES=("gaussian_any" "none_any" "glare_any" "jitter_any" "occlusion_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done

# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointButton1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/pointbutton_1_augment-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="pointbutton_1_augment"

# # Run modes to evaluate
# MODES=("gaussian_any" "none_any" "glare_any" "jitter_any" "occlusion_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done

# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointButton1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/pointbutton_1_standard-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="pointbutton_1_standard"

# # Run modes to evaluate
# MODES=("gaussian_any" "none_any" "glare_any" "jitter_any" "occlusion_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done


# # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="cargoal_1_geigh_3"

# # Run modes to evaluate
# MODES=("none_random_any")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done

# # Final training command (different task, no eval_only)
# echo "Running final training on SafetyPointGoal2-v0..."
# $SCRIPT --configs osrp --method osrp --task safetygym_SafetyPointGoal2-v0 --jax.logical_gpus 0

# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/temp_geigh_3-safedreamer/checkpoint.ckpt --run.mode gaussian_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/temp_geigh_3-safedreamer/checkpoint.ckpt --run.mode none_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/temp_geigh_3-safedreamer/checkpoint.ckpt --run.mode glare_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/temp_geigh_3-safedreamer/checkpoint.ckpt --run.mode jitter_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyPointGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/temp_geigh_3-safedreamer/checkpoint.ckpt --run.mode occlusion_random_any

# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt --run.mode occlusion_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt --run.mode jitter_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt --run.mode glare_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt --run.mode gaussian_random_any
# python SafeDreamer/train.py --configs osrp --method osrp --task safetygym_SafetyCarGoal1-v0 --jax.logical_gpus 0 --run.script eval_only --run.from_checkpoint /home/general/logdir/cargoal_1_geigh_3-safedreamer/checkpoint.ckpt --run.mode none_random_any
