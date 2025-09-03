cd ..
# # Example 1: Use default settings to train an agent
# bash train_dm3.sh 2000 0 --task carla_right_turn_simple --dreamerv3.logdir ./logdir/carla_right_turn_simple_sensor_6_augment_high --dreamerv3.run.steps 200000
bash train_dm3.sh 2000 0 --task carla_lane_merge --dreamerv3.logdir ./logdir/carla_lane_merge --dreamerv3.run.steps 200000
# bash train_dm3.sh 2000 0 --task carla_four_lane --dreamerv3.logdir ./logdir/carla_four_lane_bev_pixel --dreamerv3.run.steps 270000
# Example 1: Use default settings to train an agent

# bash train_dm3.sh 2000 0 --task carla_lane_merge --dreamerv3.logdir ./logdir/carla_lane_merge_bev_proj

# bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane_sensor_6_dropout/checkpoint.ckpt Default carla_four_lane 
# bash eval_dm3.sh 2000 0 ./logdir/carla_four_lane_sensor_6_dropout/checkpoint.ckpt Default carla_four_lane 

# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt Default carla_four_lane 
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_right_turn_simple_sensor_6_augment_high/checkpoint.ckpt Default carla_right_turn_simple
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_stop_sign_sensor_6_augment_high/checkpoint.ckpt Default carla_stop_sign_simple

# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_1 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_2 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_3 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_4 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_5 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_6 carla_four_lane

# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_1 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_2 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_3 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_4 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_5 carla_four_lane
# bash eval_dm3_sequential.sh 2000 0 ./logdir/carla_four_lane_sensor_6_augment_high/checkpoint.ckpt jitter_6 carla_four_lane



