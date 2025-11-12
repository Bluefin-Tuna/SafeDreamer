from typing import Any


from math import sqrt
import re

import embodied
import numpy as np
import os
from matplotlib import pylab # type: ignore
import cv2
import matplotlib.pyplot as plt
import imageio
import numpy as np

def eval_only(agent, env, logger, args):

  logdir = embodied.Path(args.logdir)
  logdir.mkdirs()
  should_log = embodied.when.Clock(args.log_every)
  step = logger.step
  metrics = embodied.Metrics()
  
  print('Observation space:', env.obs_space)
  print('Action space:', env.act_space)

  timer = embodied.Timer()
  timer.wrap('agent', agent, ['policy'])
  timer.wrap('env', env, ['step'])
  timer.wrap('logger', logger, ['write'])

  nonzeros = set()
  def per_episode(ep):
    length = len(ep["reward"]) - 1
    score = float(ep["reward"].astype(np.float64).sum())
    logger.add({"length": length, "score": score}, prefix="episode")
    print(f"Episode has {length} steps and return {score:.1f}.")
    stats = {}
    for key in ep:
      if 'custom' in key:
        stats[key] = ep[key]
    for key in args.log_keys_video:
      if key in ep:
        stats[f"policy_{key}"] = ep[key]
    custom_values = ['stages', 'condition_1', 'condition_2', 'condition_3',
             'gradients_exact', 'mu_gradients']
    def log(key, value):
      if key == 'log_surprise_mean':
        stats['log_surprise_mean'] = value[10] # Set value index to wanted.
      if key in custom_values:
        stats[key] = np.round(value, decimals=4)
      if re.match(args.log_keys_sum, key):
        stats[f"sum_{key}"] = value.sum()
      if re.match(args.log_keys_mean, key):
        stats[f"mean_{key}"] = value.mean()
      if re.match(args.log_keys_max, key):
        stats[f"max_{key}"] = value.max(0).mean()
    for key, value in ep.items():
      if not args.log_zeros and key not in nonzeros and (value == 0).all():
        continue
      nonzeros.add(key)
      log(key, value)
    if "stages" in ep:
      print(ep["stages"])
    logger.add(metrics.result())
    logger.add(timer.stats(), prefix="timer")
    logger.write(fps=True)
    metrics.add(stats, prefix="stats")
  driver = embodied.Driver(env)
  driver.on_episode(lambda ep, worker: per_episode(ep))
  driver.on_step(lambda tran, _: step.increment())

  checkpoint = embodied.Checkpoint()
  checkpoint.agent = agent
  print("Checkpoint", args.from_checkpoint)
  checkpoint.load(args.from_checkpoint, keys=['agent'])

  print('Start evaluation loop.')
  if 'surprise' in args.mode:
    print('Running Surprise policy')
    policy = lambda *args: agent.policy(*args, mode='surprise')
  elif 'random' in args.mode:
    print('Running Random policy')
    policy = lambda *args: agent.policy(*args, mode='random')
  elif 'sample' in args.mode:
    print('Running Sample policy')
    policy = lambda *args: agent.policy(*args, mode='sample')
  elif 'reject' in args.mode:
    print('Running Reject policy')
    policy = lambda *args: agent.policy(*args, mode='reject')
  elif 'filter' in args.mode:
    print('Running Filter policy')
    policy = lambda *args: agent.policy(*args, mode='filter')
  else:
    print('Runing Eval Policy')
    policy = lambda *args: agent.policy(*args, mode='eval')
  # policy = lambda *args: agent.policy(*args, mode='surprise')
  while step < args.steps:
    driver(policy, steps=100)

  logger.write()
  #video_list.store_video(args.logdir)
