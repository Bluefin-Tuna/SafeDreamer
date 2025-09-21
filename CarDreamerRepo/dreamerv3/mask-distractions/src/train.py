import torch
import os
import numpy as np
import gym
from utils import ReplayBuffer
import time
import sys
import torch.nn.functional as F
import cv2
import imageio
from collections import deque
# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
# import wandb
from arguments import parse_args
from algorithms.factory import make_agent
from logger import Logger
# from video import VideoRecorder, MaskRecorder, AugmentationRecorder
from video import AugmentationRecorder
import datetime
import warnings

import dreamerv3.embodied as embodied
import ruamel.yaml as yaml

import car_dreamer
import dreamerv3

warnings.filterwarnings("ignore", ".*truncated to dtype int32.*")



class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False
    
def wrap_env(env, config):
    args = config.wrapper
    env = embodied.wrappers.InfoWrapper(env)
    for name, space in env.act_space.items():
        if name == "reset":
            continue
        elif space.discrete:
            env = embodied.wrappers.OneHotAction(env, name)
        elif args.discretize:
            env = embodied.wrappers.DiscretizeAction(env, name, args.discretize)
        else:
            env = embodied.wrappers.NormalizeAction(env, name)
    env = embodied.wrappers.ExpandScalars(env)
    if args.length:
        env = embodied.wrappers.TimeLimit(env, args.length, args.reset)
    if args.checks:
        env = embodied.wrappers.CheckSpaces(env)
    for name, space in env.act_space.items():
        if not space.discrete:
            env = embodied.wrappers.ClipAction(env, name)
    return env

def _preprocess_frame_np(frame, size=96):
    """
    Convert an (H, W, 3) frame to (3, size, size) uint8 in [0, 255].
    """
    if frame is None:
        raise ValueError('Received None frame in preprocessing')
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    # Ensure float32 for scaling
    frame_np = frame.astype(np.float32, copy=False)
    # If appears to be 0..1, scale to 0..255
    if frame_np.max() <= 1.0:
        frame_np = np.clip(frame_np * 255.0, 0, 255)
    # Resize to (size, size)
    frame_np = cv2.resize(frame_np, (size, size), interpolation=cv2.INTER_AREA)
    # Convert to uint8 and CHW
    frame_np = frame_np.astype(np.uint8)
    chw = np.transpose(frame_np, (2, 0, 1))
    return chw

def init_frame_buffer(first_frame, frame_stack=3, size=96):
    """
    Initialize deque with the first processed frame repeated frame_stack times.
    Returns (deque, stacked_obs CHW with C=3*frame_stack).
    """
    proc = _preprocess_frame_np(first_frame, size)
    buf = deque(maxlen=frame_stack)
    for _ in range(frame_stack):
        buf.append(proc)
    stacked = np.concatenate(list(buf), axis=0)
    return buf, stacked

def push_frame_and_stack(buf, frame, size=96):
    """
    Append new processed frame and return stacked CHW (C=3*frame_stack).
    """
    proc = _preprocess_frame_np(frame, size)
    buf.append(proc)
    return np.concatenate(list(buf), axis=0)

def evaluate(env, agent, mask_rec, args, L, step, test_env=False, test_mode=None):
    episode_rewards = []
    for i in range(args.eval_episodes):
        obs = env.reset()
        # mask_rec.init(enabled=(i == 0))
        done = False
        episode_reward, episode_step = 0, 0
        # torch_obs, torch_action = [], []
        with torch.no_grad():
            with eval_mode(agent):
                while not done:
                    action = agent.select_action(obs, test_env)
                    obs, reward, done, _ = env.step(action)
                    # mask_rec.record(obs, agent, episode_step, step, test_env, test_mode, L)
                    # if args.algorithm == 'sgqn':
                    #     utils.logging_sgqn(agent, i, episode_step, step, obs, torch_obs, torch_action, action, test_mode)
                    episode_reward += reward
                    episode_step += 1

        if L is not None:
            _test_env = f'_test_env_{test_mode}' if test_env else ''
            L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards)

def prepare_action_for_step(raw_action, n_actions):
    """
    Converts a raw continuous action vector into a one-hot vector for step().
    """
    # Pick the largest value as the action index
    action_index = int(np.argmax(raw_action))
    
    # Create one-hot vector
    one_hot_action = np.zeros(n_actions, dtype=np.float32)
    one_hot_action[action_index] = 1.0
    
    return one_hot_action

def evaluate_discrete(env, agent, simple_args, L, step, video_dir=None, fps=25):
    """
    Evaluate policy on the same env API (embodied) using one-hot discrete actions
    and stacked frame observations.
    """
    episode_rewards = []
    if video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
    for i in range(simple_args.eval_episodes):
        # Reset env
        check, info = env.step({'action': None, 'reset': True})
        frames = []
        try:
            if isinstance(check.get('camera', None), np.ndarray):
                frames.append(check['camera'])
            elif isinstance(check.get('birdeye_wpt', None), np.ndarray):
                frames.append(check['birdeye_wpt'])
        except Exception:
            pass
        frame_buffer, obs = init_frame_buffer(
            check['birdeye_wpt'],
            frame_stack=simple_args.frame_stack,
            size=simple_args.image_crop_size,
        )
        done = False
        ep_reward = 0.0
        with torch.no_grad():
            while not done:
                action = agent.select_action(obs, test_env=True)
                action_one_hot = prepare_action_for_step(action, n_actions=15)
                check, info = env.step({'action': action_one_hot, 'reset': False})
                try:
                    if isinstance(check.get('camera', None), np.ndarray):
                        frames.append(check['camera'])
                    elif isinstance(check.get('birdeye_wpt', None), np.ndarray):
                        frames.append(check['birdeye_wpt'])
                except Exception:
                    pass
                obs = push_frame_and_stack(
                    frame_buffer,
                    check['birdeye_wpt'],
                    size=simple_args.image_crop_size,
                )
                reward = check['reward']
                done = True if (check['is_terminal'] or check['is_last']) else False
                ep_reward += reward
        # Save video
        if video_dir is not None and len(frames) > 0:
            try:
                # Ensure uint8 HxWx3
                frames_uint8 = []
                for fr in frames:
                    fr_np = fr
                    if fr_np.dtype != np.uint8:
                        fr_np = (np.clip(fr_np, 0.0, 1.0) * 255.0).astype(np.uint8) if fr_np.max() <= 1.0 else fr_np.astype(np.uint8)
                    frames_uint8.append(fr_np)
                imageio.mimsave(os.path.join(video_dir, f'eval_step{step}_ep{i}.mp4'), frames_uint8, fps=fps)
            except Exception:
                pass
        L.log('eval/episode_reward', ep_reward, step)
        episode_rewards.append(ep_reward)
    if episode_rewards:
        L.log('eval/mean_reward', float(np.mean(episode_rewards)), step)

def main(argv=None):
    model_configs = yaml.YAML(typ="safe").load((embodied.Path(__file__).parent / "dreamerv3.yaml").read())
    config = embodied.Config({"dreamerv3": model_configs["defaults"]})
    config = config.update({"dreamerv3": model_configs["small"]})

    parsed, other = embodied.Flags(task=["carla_navigation"]).parse_known(argv)
    for name in parsed.task:
        print("Using task: ", name)
        env, env_config = car_dreamer.create_task(name, argv)
        config = config.update(env_config)
    config = embodied.Flags(config).parse(other)
    # print(config)

    logdir = embodied.Path(config.dreamerv3.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(
        step,
        [
            embodied.logger.TerminalOutput(),
            embodied.logger.JSONLOutput(logdir, "metrics.jsonl"),
            embodied.logger.TensorBoardOutput(logdir),
            embodied.logger.WandBOutput(logdir.name, config)
        ],
    )

    from embodied.envs import from_gym

    dreamerv3_config = config.dreamerv3
    env = from_gym.FromGym(env)
    env = wrap_env(env, dreamerv3_config)
    # env = embodied.BatchEnv([env], parallel=False)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    config_filename = f"config_{timestamp}.yaml"
    config.save(str(logdir / config_filename))
    print(f"[Train] Config saved to {logdir / config_filename}")

    # agent = dreamerv3.Agent(env.obs_space, env.act_space, step, dreamerv3_config)
    # replay = embodied.replay.Uniform(dreamerv3_config.batch_length, dreamerv3_config.replay_size, logdir / "replay")
    args = embodied.Config(
        **dreamerv3_config.run,
        logdir=dreamerv3_config.logdir,
        batch_steps=dreamerv3_config.batch_size * dreamerv3_config.batch_length,
        actor_dist_disc=dreamerv3_config.actor_dist_disc,
    )
    # embodied.run.train(agent, env, replay, logger, args)

    # Init wandb
    # import wandb
    # wandb.init(project="madi", config=vars(args), name=utils.set_experiment_name(args),
                # mode=args.wandb_mode)
    print(args)
    print(env.act_space)
    start_training_time = time.time()
    # utils.set_seed_everywhere(args.seed)
    work_dir = './dreamerv3/mask-distractions/mask_logdir'
    model_dir = './dreamerv3/mask-distractions/mask_logdir/models'
    aug_dir = './dreamerv3/mask-distractions/mask_logdir/augs'

    # Initialize environments
    gym.logger.set_level(40)
    # Build default mask-distractions args without reading CLI.
    import types, sys as _sys
    _saved_argv = list(_sys.argv)
    try:
        _sys.argv = [_sys.argv[0]]
        simple_args = parse_args()
    finally:
        _sys.argv = _saved_argv

    # Force desired mask defaults here for faster iteration.
    simple_args.mask_type = 'hard'
    simple_args.mask_threshold_type = 'fix'
    simple_args.mask_threshold = 0.0
    simple_args.train_steps = 100000
    simple_args.init_steps = 5000
    simple_args.eval_episodes = 1
    ###
    # Stabilizers: lower LRs, shorter target update, lower discount
    simple_args.actor_lr = 1e-4
    simple_args.critic_lr = 1e-4
    simple_args.critic_tau = 0.005
    simple_args.discount = 0.98
    ###
    # env, train_env_eval, test_envs, test_modes = utils.make_envs(args)
    print(simple_args)

    # Prepare agent
    assert torch.cuda.is_available(), 'must have cuda enabled'
    replay_buffer = ReplayBuffer(
        obs_shape=(9,96,96),#(128, 128, 3), # Duplicate 3 of Carla, Carla is 128 but MADI needs 96
        action_shape=(15,),
        capacity=simple_args.train_steps,
        batch_size=28
    )

    aug_rec = AugmentationRecorder(aug_dir, simple_args.save_aug, simple_args)
    agent_args = {'aug_recorder': aug_rec} 

    cropped_obs_shape = (3 * simple_args.frame_stack, simple_args.image_crop_size, simple_args.image_crop_size)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=(15,),
        args=simple_args,
        agent_args=agent_args
    )
    
    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    mask_rec = None
    start_time = time.time()
    frame_buffer = None
    last_eval_interval = -1  # ensure first eligible interval runs once when episode ends
    for step in range(start_step, simple_args.train_steps + 1):
        if done:
            if step > start_step:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # Evaluate once per interval when episode ends; avoids missing intervals when mid-episode
            cur_interval = step // simple_args.eval_freq if simple_args.eval_freq > 0 else -1
            if step >= simple_args.eval_freq and cur_interval > last_eval_interval:
                print('Evaluating:', work_dir)
                L.log('eval/episode', episode, step)
                evaluate_discrete(env, agent, simple_args, L, step, video_dir=os.path.join(work_dir, 'videos'), fps=25)
                L.dump(step)
                last_eval_interval = cur_interval

            # Periodically save the agent
            if (step > start_step and step % simple_args.save_freq == 0) or step == 1e5:
                torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

            # # Periodically record a video
            # if step > start_step and step % args.save_vid_every == 0:
            #     # record_video(train_env_eval, agent, video, step)
            #     if test_envs is not None:
            #         for test_env, test_mode in zip(test_envs, test_modes):
            #             # record_video(test_env, agent, video, step, test_env=True, test_mode=test_mode)

            L.log('train/episode_reward', episode_reward, step)
            L.log('train/episode', episode + 1, step)

            check, info = env.step({'action': None, 'reset':True})
            
            # obs = np.transpose(check['birdeye_wpt'], (2, 0, 1)) #check['birdeye_wpt']
            # Resize to (100, 100)
            # obs = cv2.resize(check['birdeye_wpt'], (100, 100))   # still (H, W, C)
            # # Reorder to (C, H, W)
            # obs = np.transpose(obs, (2, 0, 1))
            if frame_buffer is None:
                frame_buffer, obs = init_frame_buffer(
                    check['birdeye_wpt'],
                    frame_stack=simple_args.frame_stack,
                    size=simple_args.image_crop_size,
                )
            else:
                # reset: reinitialize buffer with current frame
                frame_buffer, obs = init_frame_buffer(
                    check['birdeye_wpt'],
                    frame_stack=simple_args.frame_stack,
                    size=simple_args.image_crop_size,
                )
            # Debug: log and occasionally print input ranges
            try:
                raw_max = float(np.max(check['birdeye_wpt']))
                obs_max = float(np.max(obs))
                L.log('debug/raw_obs_max', raw_max, step)
                L.log('debug/obs_max', obs_max, step)
                if step % 200 == 0:
                    print(f"[InputMax][reset] step {step} raw_max={raw_max:.3f} obs_max={obs_max}")
            except Exception as _e:
                pass
            # obs = env._env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1

        # Sample action for data collection
        if step < simple_args.init_steps: # FLAG
            n_actions = 15

            # Randomly choose an index
            index = np.random.randint(0, n_actions)

            # Create one-hot vector
            one_hot = np.zeros(n_actions, dtype=np.float32)
            one_hot[index] = 1.0
            action = one_hot#np.random.uniform(0, 1, size=(15,)) #env.act_space.sample()
        else:
            with eval_mode(agent):
                action = agent.sample_action(obs)

        # Run training updates for initial random steps
        if step == simple_args.init_steps:
            for i in range(1, simple_args.init_steps + 1):
                agent.update(replay_buffer, L, i)

        # Run training update
        if step > simple_args.init_steps:
            agent.update(replay_buffer, L, step)

        # Take step
        # print(action)
        # Suppose your env has n_actions = n_acc * n_steer = 3 * 5 = 15
        n_actions = 15
        action_one_hot = prepare_action_for_step(action, n_actions)

        # next_obs, reward, done, _ = env.step({'action':action_one_hot, 'reset':False })
        check, info = env.step({'action':action_one_hot, 'reset':False })
        # print(check)


        # next_obs = np.transpose(, (2, 0, 1)) #check['birdeye_wpt']e
        # Resize to (100, 100)
        # next_obs = cv2.resize(check['birdeye_wpt'], (100, 100))   # still (H, W, C)

        # # Reorder to (C, H, W)
        # next_obs = np.transpose(next_obs, (2, 0, 1))
        next_obs = push_frame_and_stack(
            frame_buffer,
            check['birdeye_wpt'],
            size=simple_args.image_crop_size,
        )
        # Debug: log and occasionally print input ranges
        try:
            raw_next_max = float(np.max(check['birdeye_wpt']))
            next_obs_max = float(np.max(next_obs))
            L.log('debug/raw_next_obs_max', raw_next_max, step)
            L.log('debug/next_obs_max', next_obs_max, step)
            if step % 200 == 0:
                print(f"[InputMax][step]  step {step} raw_max={raw_next_max:.3f} obs_max={next_obs_max}")
        except Exception as _e:
            pass
        # print(next_obs.shape)  # (3, 100, 100)

        reward = check['reward']
        done = True if (check['is_terminal'] or check['is_last']) else False 

        # Treat time_exceeded as terminal (count as done)
        done_for_buffer = float(done)
        # store the executed one-hot action for consistency with discrete agent
        replay_buffer.add(obs, action_one_hot, reward, next_obs, done_for_buffer)
        episode_reward += reward
        episode_step += 1
        obs = next_obs

    L.log('train/total_training_hours', (time.time() - start_training_time) / 3600., step=simple_args.train_steps)
    print('Completed training for', work_dir)


if __name__ == "__main__":
    main()
