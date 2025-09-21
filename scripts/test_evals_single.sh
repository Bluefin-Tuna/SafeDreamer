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
# # MODES=("gaussian_random_any" "none_random_any" "glare_random_any" "jitter_random_any" "occlusion_random_any")
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
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/temp_geigh_3_augment-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="temp_geigh_3_augment"

# # Run modes to evaluate
# MODES=("none_any")

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

# # # Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointButton1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/pointbutton_1_single_dropout_less-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="full_evals/pointbutton_1_single_dropout_less"

# # Run modes to evaluate
# # MODES=("gaussian_sample_any" "none_sample_any" "glare_sample_intensity0.75_any" "jitter_sample_any" "glob_sample_any")
# MODES=("glare_sample_all" "gaussian_sample_all" "jitter_sample_all" "glob_sample_all")

# # Loop over each mode
# for MODE in "${MODES[@]}"; do
#     echo "Evaluating mode: $MODE"
#     $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT --run.mode "$MODE" --run.logdir_algo $LOGDIR_ALGO
# done

# # # # Base command parameters
SCRIPT="python SafeDreamer/train.py"
CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointButton1-v0"
JAX="--jax.logical_gpus 0"
SCRIPT_MODE="--run.script eval_only"
CHECKPOINT="--run.from_checkpoint /home/general/logdir/pointbutton_1_standard_single-safedreamer/checkpoint.ckpt"
LOGDIR_ALGO="full_evals/pointbutton_1_standard_single"

# Run modes to evaluate
# MODES=("gaussian_sample_intensity0.75_any" "none_sample_any" "glare_sample_intensity0.75_any" "jitter_sample_any" "glob_sample_any")
# MODES=("jitter_sample_intensity_all" "jitter_all" "glob_all" "gaussian_all")
MODES=("gaussian_sample_all")
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


# Base command parameters
# SCRIPT="python SafeDreamer/train.py"
# CONFIG="--configs osrp --method osrp --task safetygym_SafetyPointButton1-v0"
# JAX="--jax.logical_gpus 0"
# SCRIPT_MODE="--run.script eval_only"
# CHECKPOINT="--run.from_checkpoint /home/general/logdir/pointbutton_1_standard-safedreamer/checkpoint.ckpt"
# LOGDIR_ALGO="pointbutton_1_standard"

# # Run modes to evaluate
# # MODES=("gaussian_any" "none_any" "glare_any" "jitter_any" "occlusion_any")
# MODES=("gaussian")
# AUG_LEVELS=($(seq 0.000 0.05 1.00))
# # AUG_LEVELS=(0.1)
# TIMESTEP=10

# for MODE in "${MODES[@]}"; do
#     for AUG in "${AUG_LEVELS[@]}"; do
#         MODE_AUG="${MODE}_timestep${TIMESTEP}_intensity${AUG}_all"

#         echo "Evaluating mode: $MODE_AUG"
#         $SCRIPT $CONFIG $JAX $SCRIPT_MODE $CHECKPOINT \
#             --run.mode "$MODE_AUG" \
#             --run.logdir_algo "$LOGDIR_ALGO"
#     done
# done