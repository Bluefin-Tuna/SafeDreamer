# cd ..
# # Example 1: Use default settings to train an agent
# bash train_dm3.sh 2000 0 --task carla_right_turn_simple --dreamerv3.logdir ./logdir/carla_right_turn_simple_sensor_6_augment_high --dreamerv3.run.steps 200000
# bash train_dm3.sh 2000 0 --task carla_right_turn_simple --logdir ./logdir/carla_right_turn_simple --configs carla #--run.steps 200000
# bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane_bev_pixel --dreamerv3.run.steps 270000
# Example 1: Use default settings to train an agent
# python dreamer.py --logdir ./logdir_safedreamer/SafetyCarGoal1 --task safetygym_SafetyCarGoal1-v0 

# python dreamer.py --logdir ./logdir_safedreamer/SafetyPointButton1_eval --task safetygym_SafetyPointButton1-v0 --config safetygym --mode gaussian_all

declare -A BASE_CHECKPOINTS_DEFAULT_BEV=(
  ["carla_four_lane"]="./logdir_carla/carla_four_lane/latest.pt"
  ["carla_right_turn_simple"]="./logdir_carla/carla_right_turn_simple/latest.pt"
  ["carla_stop_sign"]="./logdir_carla/carla_stop_sign/latest.pt"
)

scenario='carla_four_lane'
PORT=2000
GPU=0
checkpoint=${BASE_CHECKPOINTS_DEFAULT_BEV[$scenario]}
variant='chrome_masked_proportion0.7_timestep10_1.0'

bash eval_dm3_sequential.sh  $PORT $GPU $checkpoint $variant $scenario 