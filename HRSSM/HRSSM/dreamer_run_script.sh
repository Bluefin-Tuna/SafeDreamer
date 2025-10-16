# cd ..
# # Example 1: Use default settings to train an agent
# bash train_dm3.sh 2000 0 --task carla_right_turn_simple --dreamerv3.logdir ./logdir/carla_right_turn_simple_sensor_6_augment_high --dreamerv3.run.steps 200000
# bash train_dm3.sh 2000 0 --task carla_right_turn_simple --logdir ./logdir/carla_right_turn_simple --configs carla #--run.steps 200000
# bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane_bev_pixel --dreamerv3.run.steps 270000
# Example 1: Use default settings to train an agent
 python dreamer.py --logdir ./logdir_safedreamer/SafetyCarGoal1 --task safetygym_SafetyCarGoal1-v0 

python dreamer.py --logdir ./logdir_safedreamer/SafetyPointButton1_eval --task safetygym_SafetyPointButton1-v0 --config safetygym --mode gaussian_all



