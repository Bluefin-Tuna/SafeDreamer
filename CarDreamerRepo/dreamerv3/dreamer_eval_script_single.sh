#!/bin/bash
cd ..
set -e

PORT=2000
GPU=0

# # Base checkpoints for dropout
# declare -A BASE_CHECKPOINTS_DROPOUT=(
#   ["carla_four_lane"]="./logdir/carla_four_lane_sensor_6_dropout/checkpoint.ckpt"
#   ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_sensor_6_dropout/checkpoint.ckpt"
#   ["carla_stop_sign"]="./logdir/carla_stop_sign_sensor_6_dropout/checkpoint.ckpt"
# )

# # Base checkpoints for default
# declare -A BASE_CHECKPOINTS_DEFAULT=(
#   ["carla_four_lane"]="./logdir/carla_four_lane_sensor_6/checkpoint.ckpt"
#   ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_sensor_6/checkpoint.ckpt"
#   ["carla_stop_sign"]="./logdir/carla_stop_sign_sensor_6/checkpoint.ckpt"
# )

# Base checkpoints for default
declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
)

# # Base checkpoints for default
# declare -A BASE_CHECKPOINTS_PROJECT=(
#   ["carla_four_lane"]="./logdir/carla_four_lane_bev_proj/checkpoint.ckpt"
#   ["carla_right_turn_simple"]="./logdir/carla_right_turn_bev_proj/checkpoint.ckpt"
#   ["carla_stop_sign"]="./logdir/carla_stop_sign_sensor_bev_proj/checkpoint.ckpt"
# )

# Base checkpoints for default
declare -A BASE_CHECKPOINTS_PIXEL=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev_proj/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev_proj/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev_pixel/checkpoint.ckpt"
)

# # Base checkpoints for default
# declare -A BASE_CHECKPOINTS_RECON=(
#   ["carla_four_lane"]="./logdir/carla_four_lane_bev_recon/checkpoint.ckpt"
#   ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev_recon/checkpoint.ckpt"
#   ["carla_stop_sign"]="./logdir/carla_stop_sign_bev_recon/checkpoint.ckpt"
# )

# Scenarios
SCENARIOS=("carla_stop_sign")
AUG_TYPES=("gaussian")
# AUG_LEVELS=(0.1 0.010)
# AUG_LEVELS=($(seq 0.000 0.005 0.100))
PROPORTION_LEVELS=(0.75)

AUG_LEVELS=(1.0)

run_eval() {
  local checkpoint="$1"
  local variant="$2"
  local scenario="$3"
  echo "Running: $variant on $scenario"
  bash eval_dm3_sequential.sh "$PORT" "$GPU" "$checkpoint" "$variant" "$scenario"
}

# echo "=== Running Augment High Default ==="
# for scenario in "${SCENARIOS[@]}"; do
#   run_eval "${BASE_CHECKPOINTS_PROJECT[$scenario]}" "Default" "$scenario"
# done

# echo "=== Running Single Projection Default ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       run_eval "${BASE_CHECKPOINTS_PROJECT[$scenario]}" "sample_Default" "$scenario"
#     done
#   done
# done

# echo "=== Running Single Projection Augmentations ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       run_eval "${BASE_CHECKPOINTS_PIXEL[$scenario]}" "sample_${aug}_1" "$scenario"
#     done
#   done
# done
#augtype_policytype_timestep$_intensity$

# echo "=== Running Sampled Augmentations ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       for proport in "${PROPORTION_LEVELS[@]}"; do
#         run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_sample_proportion${proport}_timestep10_${level}" "$scenario"
#       done
#     done
#   done
# done

echo "=== Running Reject Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_reject_proportion${proport}_timestep10_${level}" "$scenario"
      done
    done
  done
done


# echo "=== Running Default Augmentations ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       for proport in "${PROPORTION_LEVELS[@]}"; do
#         run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_proportion${proport}_timestep10_${level}" "$scenario"
#       done
#     done
#   done
# done

# echo "=== Running Filter Augmentations ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       for proport in "${PROPORTION_LEVELS[@]}"; do
#         run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_filter_proportion${proport}_timestep10_${level}" "$scenario"
#       done
#     done
#   done
# done


# echo "=== Running Pixel Augmentations ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       run_eval "${BASE_CHECKPOINTS_PIXEL[$scenario]}" "sample_${aug}_${level}" "$scenario"
#     done
#   done
# done