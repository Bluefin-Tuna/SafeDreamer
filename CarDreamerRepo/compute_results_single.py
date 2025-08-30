import os
import re
from collections import defaultdict
import json
import numpy as np
import matplotlib.pyplot as plt

def gen_score(file_path):
    scores = []

    # with open("pointbutton_1_geigh_3_occlusion_surprise_any/metrics.jsonl", "r") as f:
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            scores.append(data["episode/score"])

    scores = np.array(scores)
    mean = np.mean(scores)
    std = np.std(scores)

    return mean#, std

# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_recon_carla_stop_sign_sample_glare_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_carla_stop_sign_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_sample_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_carla_stop_sign_jitter_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_sample_jitter_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_carla_stop_sign_glare_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_sample_glare_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_carla_stop_sign_occlusion_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_stop_sign_bev_pixel_carla_stop_sign_sample_occlusion_1/metrics.jsonl'))
print('------------')
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_gaussian_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_pixel_carla_right_turn_simple_sample_gaussian_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_sample_gaussian_1/metrics.jsonl'))

print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_jitter_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_pixel_carla_right_turn_simple_sample_jitter_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_sample_jitter_1/metrics.jsonl'))

print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_glare_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_pixel_carla_right_turn_simple_sample_glare_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_sample_glare_1/metrics.jsonl'))

print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_occlusion_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_pixel_carla_right_turn_simple_sample_occlusion_1/metrics.jsonl'))
print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_right_turn_simple_bev_carla_right_turn_simple_sample_occlusion_1/metrics.jsonl'))
print('----------')
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_carla_four_lane_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_pixel_carla_four_lane_sample_gaussian_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_carla_four_lane_jitter_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_pixel_carla_four_lane_sample_jitter_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_carla_four_lane_glare_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_pixel_carla_four_lane_sample_glare_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_carla_four_lane_occlusion_1/metrics.jsonl'))
# print(gen_score('/home/general/Documents/work/Trolls/SafeRL/CarDreamer/CarDreamer/CarDreamerRepo/logdir/evals/carla_four_lane_bev_pixel_carla_four_lane_sample_occlusion_1/metrics.jsonl'))