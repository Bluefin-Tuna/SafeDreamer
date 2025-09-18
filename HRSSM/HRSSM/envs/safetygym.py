import functools
import os
import random

from . import base
import numpy as np
import cv2
np.random.seed(0)
import re
from . import space as spacelib

class SafetyGym(base.Env):

  def __init__(self, env, platform='gpu', repeat=5, obs_key='image', render=True, size=(64, 64), camera=-1, mode='train', camera_name='vision_front_back'):
    # TODO: This env variable is meant for headless GPU machines but may fail
    # on CPU-only machines.
    if platform =='gpu' and 'MUJOCO_GL' not in os.environ:
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['LIBGL_ALWAYS_SOFTWARE']=True

    # Uncomment and modify this section:
    # os.environ['MUJOCO_GL'] = 'egl'
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'

    # import gymnasium
    import safety_gymnasium
    if mode=='train':
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
    self._render = True
    self._size = size
    self._camera = camera
    self._camera_name = camera_name
    self._repeat = repeat
    self._mode = mode
    print(self._mode)

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
      spaces['image'] = spacelib.Space(np.uint8, self._size + (3,))

      # if self._camera_name == 'vision_front_back':
      #   spaces['image2'] = spacelib.Space(np.uint8, self._size + (3,))
      # spaces['image3'] = spacelib.Space(np.uint8, self._size + (3,))
    # print(spaces)
    return spaces

  @functools.cached_property
  def act_space(self):
    return self._env.act_space
  
  def set_mode(self, mode):
    print('setting mode to inner', mode)
    self._mode = mode
  def reset(self):
    #  print('Reset in safety gym')

     return self.step(action={'reset' : True, 'action': np.array([0., 1.])})[0]


    #  print(obs)
    #  return obs
  
  def step(self, action):
    for key, space in self.act_space.items():
      if not space.discrete:
        assert np.isfinite(action[key]).all(), (key, action[key])

    action = action.copy()
    # if action['reset']:
    if 'reset' in action.keys():
      obs = self._reset()
    else:
        # print('Info: ', self._env.info)
        # print('Step:',self._env._env._elapsed_steps)
        reward = 0.0
        cost = 0.0
        for i in range(self._repeat):
            obs = self._env.step(action)
            # print(obs)
            # print(obs['reward'])
            reward += obs['reward']
            if 'cost' in obs.keys():
                cost += obs['cost']
            if obs['is_last'] or obs['is_terminal']:
                break
        obs['reward'] = np.float32(reward)
        if 'cost' in obs.keys():
            obs['cost'] = np.float32(cost)
    #obs= obs['vision']
    if self._render:
      if self._mode == 'train':
        image1 = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='vision', cost={})
        obs['image'] = image1
        # if self._camera_name == 'vision_front_back':
        #   image2 = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='vision_back', cost={})
        #   obs['image2'] = image2
        
        # obs['image3'] = self._env.task.render(width=64, height=64, mode='rgb_array', camera_name='fixedfar', cost={'cost_sum': obs['cost']})
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
        np.random.seed(0)
        # Add novel changes here:
        nov_key = self._mode.split('_')[-1] # {MODE}_timestep${TIMESTEP}_intensity${AUG}_all
        # available_keys = ['image','image2']
        available_keys = ['image']
        if nov_key not in available_keys:
          if nov_key == 'all':
             nov_keys = available_keys
          else:
            nov_keys = [random.choice(available_keys)]

        match = re.search(r'(?:^|_)timestep(\d+)(?:_|$)', self._mode)
        # print(match)
        if match:
            noise_timestep = int(match.group(1))
            # print(f"Found timestep number: {noise_timestep}")
        else:
            noise_timestep = -1 #Do all the time

        match = re.search(r'(?:^|_)intensity(\d+(?:\.\d+)?)(?:_|$)', self._mode)
        if match:
            intensity = float(match.group(1))
        else:
            intensity = 1.0

        

          # print(nov_key)
        if noise_timestep == -1 or noise_timestep == int(self._env._env._elapsed_steps/self._repeat):
          mu = 20 * intensity
          std  = 30 * intensity
          # print(noise_timestep)
          for nov_key in nov_keys:
            if 'jitter' in self._mode:
              obs['high_def_nov'] = obs['image_orignal'].copy()
              # print('Mode switch')
              # 1. Apply color jitter to simulate lighting variation
              image_jitter = obs[nov_key].astype(np.float32)
              brightness_factor = np.random.uniform(mu, std)
              contrast_factor = np.random.uniform(mu, std)
              image_jitter = np.clip(image_jitter * contrast_factor + brightness_factor * 10, 0, 255).astype(np.uint8)
              obs[nov_key] = image_jitter
              obs['high_def_nov'] = np.clip(obs['high_def_nov'].astype(np.float32) * contrast_factor + brightness_factor * 10, 0, 255).astype(np.uint8)

            if 'glare' in self._mode:
              obs['high_def_nov'] = obs['image_orignal'].copy()
              # print('Mode switch')
              # 1. Apply color jitter to simulate lighting variation
              image_jitter = obs[nov_key].astype(np.float32)
              brightness_factor = np.random.uniform(mu, std)
              contrast_factor = 0
              image_jitter = np.clip(image_jitter * contrast_factor + brightness_factor * 10, 0, 255).astype(np.uint8)
              obs[nov_key] = image_jitter
              obs['high_def_nov'] = np.clip(obs['high_def_nov'].astype(np.float32) * contrast_factor + brightness_factor * 10, 0, 255).astype(np.uint8)

            if 'gaussian' in self._mode:
              obs['high_def_nov'] = obs['image_orignal'].copy()
              # 2. Add Gaussian noise to simulate sensor noise
              noise = np.random.normal(mu, std, obs[nov_key].shape).astype(np.uint8)
              noise_high_def = np.random.normal(mu, std, obs['high_def_nov'].shape).astype(np.uint8)
              obs[nov_key] = np.clip(obs[nov_key] + noise, 0, 255)
              # obs['image'] = np.clip(obs['image'] + noise, 0, 255)
              obs['high_def_nov'] = np.clip(obs['high_def_nov'] + noise_high_def, 0, 255)
              
            if 'occlusion' in self._mode:
              obs['high_def_nov'] = obs['image_orignal'].copy()
              # print('Mode switch')
              # 3. Mask part of the image to simulate occlusion
              occlusion_mask = obs[nov_key].copy()
              h, w, _ = occlusion_mask.shape
              x, y = np.random.randint(0, w//2), np.random.randint(0, h//2)
              mask_w, mask_h = np.random.randint(35, 40), np.random.randint(35, 40)
              occlusion_mask[y:y+mask_h, x:x+mask_w] = 0
              # obs['image_occlusion'] = occlusion_mask
              obs[nov_key] = occlusion_mask

              occlusion_mask_hd = obs['high_def_nov'].copy()
              h_hd, w_hd, _ = occlusion_mask_hd.shape
              x_hd, y_hd = np.random.randint(0, w_hd // 2), np.random.randint(0, h_hd // 2)
              # Scale mask size for HD (e.g., 2x larger than low-res)
              scale_factor = 2
              mask_w_hd, mask_h_hd = mask_w * scale_factor, mask_h * scale_factor
              occlusion_mask_hd[y_hd:y_hd + mask_h_hd, x_hd:x_hd + mask_w_hd] = 0
              obs['high_def_nov'] = occlusion_mask_hd
            if 'blob' in self._mode:
              obs['high_def_nov'] = obs['image_orignal'].copy()
              
              # Copy image for corruption
              blob_img = obs[nov_key].copy()
              h, w, _ = blob_img.shape
              
              # Random blob center + radius
              center_x, center_y = np.random.randint(w), np.random.randint(h)
              radius = np.random.randint(10, 70)
              
              # Random intensity (bright or dark)
              color = np.random.choice([0, 255])
              
              # Apply blob (circle)
              Y, X = np.ogrid[:h, :w]
              mask = (X - center_x)**2 + (Y - center_y)**2 <= radius**2
              blob_img[mask] = color
              
              obs[nov_key] = blob_img
              
              # Do the same for high-def (scaled radius)
              hd = obs['high_def_nov'].copy()
              h_hd, w_hd, _ = hd.shape
              scale = 2
              center_x_hd, center_y_hd = center_x*scale, center_y*scale
              radius_hd = radius*scale
              Y_hd, X_hd = np.ogrid[:h_hd, :w_hd]
              mask_hd = (X_hd - center_x_hd)**2 + (Y_hd - center_y_hd)**2 <= radius_hd**2
              hd[mask_hd] = color
              obs['high_def_nov'] = hd

    # print(obs['image'].shape)
    # obs['image'] = np.expand_dims(obs['image'], axis=0)
    return obs, obs['reward'], (obs['is_last'] or obs['is_terminal']), {} #obs

  def _reset(self):
    obs = self._env.step({'reset': True})
    # obs['image'] = np.expand_dims(obs['image'], axis=0)
    return obs

  def render(self):
    return self._dmenv.render()
