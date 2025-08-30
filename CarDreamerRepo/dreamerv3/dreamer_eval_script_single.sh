#!/bin/bash
cd ..
set -e

PORT=2000
GPU=0


# Base checkpoints for default
declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
)

# Base checkpoints for default
declare -A BASE_CHECKPOINTS_PIXEL_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev_pixel/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev_pixel/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev_pixel/checkpoint.ckpt"
)

# Scenarios
SCENARIOS=("carla_right_turn_simple")
AUG_TYPES=("gaussian" "glare" "jitter" "occlusion")
# AUG_TYPES=("glare")
AUG_LEVELS=(1)

run_eval() {
  local checkpoint="$1"
  local variant="$2"
  local scenario="$3"
  echo "Running: $variant on $scenario"
  bash eval_dm3_sequential.sh "$PORT" "$GPU" "$checkpoint" "$variant" "$scenario"
}


# echo "=== Running Standard Default ==="
# for scenario in "${SCENARIOS[@]}"; do
#   run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "Default" "$scenario"
# done

# echo "=== Running Standard Augmentations ==="
# for scenario in "${SCENARIOS[@]}"; do
#   for aug in "${AUG_TYPES[@]}"; do
#     for level in "${AUG_LEVELS[@]}"; do
#       run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "${aug}_${level}" "$scenario"
#     done
#   done
# done

# echo "=== Running Pixel Default ==="
# for scenario in "${SCENARIOS[@]}"; do
#   run_eval "${BASE_CHECKPOINTS_PIXEL_BEV[$scenario]}" "sample_Default" "$scenario"
# done

echo "=== Running Pixel Augmentations ==="
for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      run_eval "${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}" "sample_${aug}_${level}" "$scenario"
    done
  done
done

