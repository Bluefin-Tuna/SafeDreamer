import functools
import os
import random

import embodied
import numpy as np
import cv2
np.random.seed(0)
import re

MU_FACTOR = 20
STD_FACTOR = 20
BRIGHTNESS_FACTOR = 10

class SafetyGym(embodied.Env):

  def __init__(self, env, platform='gpu', repeat=1, obs_key='image', render=False, size=(64, 64), camera=-1, mode='train', camera_name='vision'):

    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if platform =='gpu' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'

    import gymnasium
    import safety_gymnasium
    
    if mode == 'train':
      env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name=camera_name, width=size[0], height=size[1])
    else:
      env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name=camera_name, width=1024, height=1024)
    # elif mode=='eval':
    #   env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name=camera_name, width=1024, height=1024)
    # elif mode=='gaussian':
    #   env = safety_gymnasium.make(env,render_mode='rgb_array',camera_name=camera_name, width=1024, height=1024)

    self._dmenv = env
    from . import from_gymnasium
    self._env = from_gymnasium.FromGymnasium(self._dmenv,obs_key=obs_key)
    self._render = render if mode=='train' else True
    self._size = size
    self._camera = camera
    self._camera_name = camera_name
    self._repeat = repeat
    self._mode = mode
    self._ts = 0

  @property
  def repeat(self):
    return self._repeat

  @property
  def mode(self):
    return self._mode

  @functools.cached_property
  def obs_space(self):
    spaces = self._env.obs_space.copy()
    if self._render:
      spaces['image'] = embodied.Space(np.uint8, self._size + (3,))
      if self._camera_name == 'vision_front_back':
        spaces['image2'] = embodied.Space(np.uint8, self._size + (3,))
      spaces['image3'] = embodied.Space(np.uint8, self._size + (3,))

    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space
  
  def set_mode(self, mode):
    print('setting mode to inner', mode)
    self._mode = mode

  def step(self, action):
    
    self._ts += 1
    
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])

    action = action.copy()
    if action['reset']:
      obs = self._reset()
    else:
        # print('Info: ', self._env.info)
        # print('Step:',self._env._env._elapsed_steps)
        reward = 0.0
        cost = 0.0
        for i in range(self._repeat):
            obs = self._env.step(action)
            reward += obs['reward']
            if 'cost' in obs.keys():
                cost += obs['cost']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        if 'cost' in obs.keys():
            obs['cost'] = np.float32(cost)

    if self._render:
      if self._mode == 'train':
        image1 = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='vision', cost={})
        obs['image'] = image1
        if self._camera_name == 'vision_front_back':
          image2 = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='vision_back', cost={})
          obs['image2'] = image2
        
        obs['image3'] = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='fixedfar', cost={'cost_sum': obs['cost']})
      # elif self._mode == 'eval':
      else:
        obs['image_orignal'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='vision', cost={})
        image = cv2.resize(
            obs['image_orignal'], self._size, interpolation=cv2.INTER_AREA)
        obs['image'] = image
        obs['image3'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='fixedfar', cost={'cost_sum': obs['cost']})
               
        obs['image3']= cv2.resize(obs['image3'], self._size, interpolation=cv2.INTER_AREA)

        if self._camera_name == 'vision_front_back':
          obs['image_orignal2'] = self._env.task.render(width=1024, height=1024, mode='rgb_array', camera_name='vision_back', cost={})
          image2 = cv2.resize(
              obs['image_orignal2'], self._size, interpolation=cv2.INTER_AREA)
          obs['image2'] = image2
    
    obs = self._simulate_failure(obs)

    return obs

  def _simulate_failure(self, obs):
    np.random.seed(0)
    # Add novel changes here:
    nov_key = self._mode.split('_')[-1]
    available_keys = ['image','image2']
    if nov_key not in available_keys:
      if nov_key == 'all':
          nov_keys = available_keys
      else:
        nov_keys = [random.choice(available_keys)]

    noise_intensity = 0

    match = re.search(r'(?:^|_)timestep(\d+)(?:_|$)', self.mode)
    if match:
        noise_timestep = int(match.group(1))
    else:
        noise_timestep = -1

    match = re.search(r'(?:^|_)proportion([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:_|$)', self.mode)
    if match:
        proportion = float(match.group(1))
    else:
        proportion = -1.0
    
    def apply_jitter(obs, nov_key, intensity):
      mu = MU_FACTOR * intensity
      std = STD_FACTOR * intensity
      img = obs[nov_key].astype(np.float32)
      brightness = np.random.uniform(mu, std)
      contrast = np.random.uniform(mu, std)
      img = np.clip(img * contrast + brightness * BRIGHTNESS_FACTOR, 0, 255).astype(np.uint8)
      obs[nov_key] = img
      obs['high_def_nov'] = np.clip(obs['high_def_nov'].astype(np.float32) * contrast + brightness * BRIGHTNESS_FACTOR, 0, 255).astype(np.uint8)
      return obs

    def apply_glare(obs, nov_key, intensity): # ISSUES: Unsure about the contrast factor. Seemingly zeros out the image.
      mu = MU_FACTOR * intensity
      std = STD_FACTOR * intensity
      img = obs[nov_key].astype(np.float32)
      brightness = np.random.uniform(mu, std)
      contrast = 0 # No contrast
      img = np.clip(img * contrast + brightness * BRIGHTNESS_FACTOR, 0, 255).astype(np.uint8)
      obs[nov_key] = img
      obs['high_def_nov'] = np.clip(obs['high_def_nov'].astype(np.float32) * contrast + brightness * BRIGHTNESS_FACTOR, 0, 255).astype(np.uint8)
      return obs

    def apply_gaussian(obs, nov_key, intensity):
      mu = MU_FACTOR * intensity
      std = STD_FACTOR * intensity
      noise = np.random.normal(mu, std, obs[nov_key].shape).astype(np.uint8)
      noise_high_def = np.random.normal(mu, std, obs['high_def_nov'].shape).astype(np.uint8)
      obs[nov_key] = np.clip(obs[nov_key] + noise, 0, 255)
      obs['high_def_nov'] = np.clip(obs['high_def_nov'] + noise_high_def, 0, 255)
      return obs

    def apply_occlusion(obs, nov_key, intensity):
      
      mu = MU_FACTOR * intensity
      std = STD_FACTOR * intensity
      scale_factor = 2 # Scale factor for the high-definition image
      
      mask_w, mask_h = np.random.randint(mu, std), np.random.randint(mu, std)
      mask_w_hd, mask_h_hd = mask_w * scale_factor, mask_h * scale_factor

      img = obs[nov_key].copy()
      h, w, _ = img.shape
      x, y = np.random.randint(0, w//2), np.random.randint(0, h//2)
      img[y:y+mask_h, x:x+mask_w] = 0
      obs[nov_key] = img
      
      img_hd = obs['high_def_nov'].copy()
      h_hd, w_hd, _ = img_hd.shape
      x_hd, y_hd = np.random.randint(0, w_hd // 2), np.random.randint(0, h_hd // 2)
      img_hd[y_hd:y_hd + mask_h_hd, x_hd:x_hd + mask_w_hd] = 0
      obs['high_def_nov'] = img_hd

      return obs
    
    def apply_channelswap(obs, nov_key, intensity):
      obs['high_def_nov'] = obs['image_orignal'].copy()
      obs[nov_key] = obs[nov_key][..., np.random.permutation(3)]
      obs['high_def_nov'] = obs['high_def_nov'][..., np.random.permutation(3)]
      return obs

    def apply_chromatic_aberration(obs, nov_key, intensity):

      def translate(channel, h, w, dx, dy):
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(channel, M, (w, h), borderMode=cv2.BORDER_REFLECT)

      img = obs[nov_key].astype(np.float32)
      h, w, _ = img.shape
      max_shift = int(2 + 10 * intensity)  # scale with intensity
      shift_r = np.random.randint(-max_shift, max_shift + 1, size=2)
      shift_g = np.random.randint(-max_shift, max_shift + 1, size=2)
      shift_b = np.random.randint(-max_shift, max_shift + 1, size=2)
      r = translate(img[:, :, 0], h, w, *shift_r)
      g = translate(img[:, :, 1], h, w, *shift_g)
      b = translate(img[:, :, 2], h, w, *shift_b)
      merged = np.stack([r, g, b], axis=2)
      obs[nov_key] = np.clip(merged, 0, 255).astype(np.uint8)

      img_hd = obs['high_def_nov'].astype(np.float32)
      h_hd, w_hd, _ = img_hd.shape
      max_shift_hd = int(2 + 10 * intensity)  # scale with intensity
      shift_r_hd = np.random.randint(-max_shift_hd, max_shift_hd + 1, size=2)
      shift_g_hd = np.random.randint(-max_shift_hd, max_shift_hd + 1, size=2)
      shift_b_hd = np.random.randint(-max_shift_hd, max_shift_hd + 1, size=2)
      r_hd = translate(img_hd[:, :, 0], h_hd, w_hd, *shift_r_hd)
      g_hd = translate(img_hd[:, :, 1], h_hd, w_hd, *shift_g_hd)
      b_hd = translate(img_hd[:, :, 2], h_hd, w_hd, *shift_b_hd)
      merged_hd = np.stack([r_hd, g_hd, b_hd], axis=2)
      obs['high_def_nov'] = np.clip(merged_hd, 0, 255).astype(np.uint8)
      
      return obs

    if random.random() > proportion:
      return obs
    if noise_timestep == -1 or self._ts > noise_timestep:
      obs['high_def_nov'] = obs['image_orignal'].copy()
      for nov_key in nov_keys:
        if 'jitter' in self._mode:
          obs = apply_jitter(obs, nov_key, noise_intensity)
        if 'glare' in self._mode:
          obs = apply_glare(obs, nov_key, noise_intensity)
        if 'gaussian' in self._mode:
          obs = apply_gaussian(obs, nov_key, noise_intensity)
        if 'occlusion' in self._mode:
          obs = apply_occlusion(obs, nov_key, noise_intensity)
        if 'channelswap' in self._mode:
          obs = apply_channelswap(obs, nov_key, noise_intensity)
        if 'chrome' in self._mode:
          obs = apply_chromatic_aberration(obs, nov_key, noise_intensity)
    return obs

  def _reset(self):
    self._ts = 0
    obs = self._env.step({'reset': True})
    return obs

  def render(self):
    return self._dmenv.render()
