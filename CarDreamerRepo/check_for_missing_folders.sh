#!/bin/bash

# Same configuration as your submission script
SCENARIOS=("carla_stop_sign" "carla_right_turn_simple" "carla_four_lane")
AUG_TYPES=("gaussian" "jitter" "glare" "occlusion")
AUG_LEVELS=(0.625 0.75 0.875 1.0)
PROPORTION_LEVELS=(0.5 0.625 0.75 0.875)

# Base checkpoints to get DIR_NAME
declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir/carla_four_lane_bev/checkpoint.ckpt"
  ["carla_right_turn_simple"]="./logdir/carla_right_turn_simple_bev/checkpoint.ckpt"
  ["carla_stop_sign"]="./logdir/carla_stop_sign_bev/checkpoint.ckpt"
)

BASE_DIR="/coc/flash5/gzollicoffer3/SafeDreamer/CarDreamerRepo/logdir/evals_finegrained"

# Calculate total expected folders
total_expected=$((${#SCENARIOS[@]} * ${#AUG_TYPES[@]} * ${#AUG_LEVELS[@]} * ${#PROPORTION_LEVELS[@]}))
echo "Expected total folders: $total_expected"
echo ""

missing_count=0
empty_count=0
found_count=0
missing_folders=()
empty_folders=()

for scenario in "${SCENARIOS[@]}"; do
  for aug in "${AUG_TYPES[@]}"; do
    for level in "${AUG_LEVELS[@]}"; do
      for proport in "${PROPORTION_LEVELS[@]}"; do
        checkpoint="${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}"
        dir_name=$(basename "$(dirname "$checkpoint")")
        variant="${aug}_reject5_proportion${proport}_timestep10_${level}"
        
        # This matches the LOG_DIR format in your eval_dm3_sequential_jobs.sh
        expected_folder="${BASE_DIR}/${dir_name}_${scenario}_${variant}"
        
        if [ -d "$expected_folder" ]; then
          # Check if folder is empty
          if [ -z "$(ls -A "$expected_folder")" ]; then
            ((empty_count++))
            empty_folders+=("$(basename "$expected_folder")")
            echo "⚠ Empty: $(basename "$expected_folder")"
          else
            ((found_count++))
            echo "✓ Found: $(basename "$expected_folder")"
          fi
        else
          ((missing_count++))
          missing_folders+=("$(basename "$expected_folder")")
          echo "✗ Missing: $(basename "$expected_folder")"
        fi
      done
    done
  done
done

echo ""
echo "========================================"
echo "Summary:"
echo "  Total expected: $total_expected"
echo "  Found (with content): $found_count"
echo "  Empty folders: $empty_count"
echo "  Missing: $missing_count"
echo "========================================"

has_issues=0

if [ $missing_count -gt 0 ]; then
  echo ""
  echo "Missing folders:"
  for folder in "${missing_folders[@]}"; do
    echo "  - $folder"
  done
  has_issues=1
fi

if [ $empty_count -gt 0 ]; then
  echo ""
  echo "Empty folders:"
  for folder in "${empty_folders[@]}"; do
    echo "  - $folder"
  done
  has_issues=1
fi

if [ $has_issues -eq 1 ]; then
  exit 1
else
  echo ""
  echo "✓ All folders present and contain files!"
  exit 0
fi