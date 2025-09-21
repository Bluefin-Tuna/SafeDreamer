from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from algorithms.vit.svea_vit import SVEA_ViT
import algorithms.modules as m
from video import AugmentationRecorder
from augmentations import strong_augment



class DiscreteActor(nn.Module):
	def __init__(self, encoder, num_actions, hidden_dim=256):
		super().__init__()
		self.encoder = encoder
		self.num_actions = num_actions
		self.mlp = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, num_actions)
		)

	def forward(self, obs, detach=False):
		features = self.encoder(obs, detach=detach)
		logits = self.mlp(features)
		return logits


class DiscreteCritic(nn.Module):
	def __init__(self, encoder, num_actions, hidden_dim=256):
		super().__init__()
		self.encoder = encoder
		self.num_actions = num_actions
		# Simple linear heads over encoder features to all action values
		self.Q1 = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, num_actions)
		)
		self.Q2 = nn.Sequential(
			nn.Linear(self.encoder.out_dim, hidden_dim), nn.ReLU(),
			nn.Linear(hidden_dim, num_actions)
		)

	def forward(self, obs, action: torch.Tensor = None, detach: bool = False):
		features = self.encoder(obs, detach=detach)
		q1_all = self.Q1(features)
		q2_all = self.Q2(features)
		if action is None:
			return q1_all, q2_all
		# action expected one-hot (B, A)
		q1 = (q1_all * action).sum(dim=-1, keepdim=True)
		q2 = (q2_all * action).sum(dim=-1, keepdim=True)
		return q1, q2


class MaDi_ViT(SVEA_ViT):
	"""MaDi: Masking Distractions for Generalization in Reinforcement Learning
	Vision Transformer backbone adapted to discrete (one-hot) action spaces"""
	def __init__(self, obs_shape, action_shape, args, aug_recorder: AugmentationRecorder):
		super().__init__(obs_shape, action_shape, args)
		self.masker = m.MaskerNet(obs_shape, args).cuda()
		self.masker_optimizer = torch.optim.Adam(
			self.masker.parameters(), lr=args.masker_lr, betas=(args.masker_beta, 0.999)
		)
		self.num_masks = args.frame_stack
		self.aug_recorder = aug_recorder
		self.num_actions = action_shape[0]
		# Replace continuous actor with categorical discrete actor, reuse existing encoder
		self.actor = DiscreteActor(self.actor.encoder, self.num_actions, args.hidden_dim).cuda()
		self.actor_optimizer = torch.optim.Adam(
			self.actor.parameters(), lr=args.actor_lr, betas=(args.actor_beta, 0.999)
		)
		# Replace continuous critic with discrete critic over all actions
		old_critic_encoder = self.critic.encoder
		self.critic = DiscreteCritic(old_critic_encoder, self.num_actions, args.hidden_dim).cuda()
		self.critic_target = deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(
			self.critic.parameters(), lr=args.critic_lr, betas=(args.critic_beta, 0.999)
		)
		# Use discrete target entropy
		self.target_entropy = np.log(self.num_actions)

	def apply_mask(self, obs, test_env=False):
		# obs: tensor shaped as (B, 9, H, W)
		frames = obs.chunk(self.num_masks, dim=1)  # frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
		frames_cat = torch.cat(frames, dim=0)  # concat in batch dim. frames_cat: tensor shaped (B*3, 3, H, W)
		masks_cat = self.masker(frames_cat, test_env=test_env)  # apply MaskerNet just once. masks_cat: (B*3, 1, H, W)
		masks = masks_cat.chunk(self.num_masks, dim=0)  # split the batch dim back into channel dim. masks: list of tensors [ (B,1,H,W) , (B,1,H,W) , (B,1,H,W) ]
		masked_frames = [m * f for m, f in zip(masks, frames)]  # element-wise multiplication. uses broadcasting. masked_frames: list of tensors [ (B,3,H,W) , (B,3,H,W) , (B,3,H,W) ]
		return torch.cat(masked_frames, dim=1)  # concat in channel dim. returns: tensor shaped (B, 9, H, W)

	def select_action(self, obs, test_env=False):
		_obs = self._obs_to_input(obs)
		_obs = self.apply_mask(_obs, test_env)
		with torch.no_grad():
			logits = self.actor(_obs)
			probs = F.softmax(logits, dim=-1)
			action_idx = torch.argmax(probs, dim=-1)
			action_onehot = F.one_hot(action_idx, num_classes=self.num_actions).float()
		return action_onehot.cpu().data.numpy().flatten()

	def sample_action(self, obs):
		_obs = self._obs_to_input(obs)
		_obs = self.apply_mask(_obs)
		with torch.no_grad():
			logits = self.actor(_obs)
			probs = F.softmax(logits, dim=-1)
			dist = torch.distributions.Categorical(probs=probs)
			action_idx = dist.sample()
			action_onehot = F.one_hot(action_idx, num_classes=self.num_actions).float()
		return action_onehot.cpu().data.numpy().flatten()

	def _q_values_all_actions(self, critic, obs):
		# Returns (q1_all, q2_all) each of shape (B, num_actions)
		return critic(obs, action=None)

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			next_obs_masked = self.apply_mask(next_obs)
			action_logits = self.actor(next_obs_masked)
			action_probs = F.softmax(action_logits, dim=-1)
			log_probs = F.log_softmax(action_logits, dim=-1)
			target_Q1_all, target_Q2_all = self._q_values_all_actions(self.critic_target, next_obs_masked)
			min_target_Q_all = torch.min(target_Q1_all, target_Q2_all)
			expected_Q = torch.sum(action_probs * min_target_Q_all, dim=-1, keepdim=True)
			entropy = -torch.sum(action_probs * log_probs, dim=-1, keepdim=True)
			target_V = expected_Q + self.alpha.detach() * entropy
			target_Q = reward + (not_done * self.discount * target_V)

		obs_aug = strong_augment(obs, self.augment, self.overlay_alpha)
		self.aug_recorder.record(obs_aug, step)

		if self.svea_alpha == self.svea_beta:
			obs_combined = utils.cat(obs, obs_aug)
			obs_combined = self.apply_mask(obs_combined)
			self.aug_recorder.record(obs_combined[obs_combined.shape[0] // 2:], step, masked=True)
			action_combined = utils.cat(action, action)
			target_Q_combined = utils.cat(target_Q, target_Q)
			current_Q1, current_Q2 = self.critic(obs_combined, action_combined)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q_combined) + F.mse_loss(current_Q2, target_Q_combined))
		else:
			obs_masked = self.apply_mask(obs)
			current_Q1, current_Q2 = self.critic(obs_masked, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
			obs_aug_masked = self.apply_mask(obs_aug)
			self.aug_recorder.record(obs_aug_masked, step, masked=True)
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug_masked, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)

		self.masker_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		###
		# Stabilizer: gradient clipping on critic and masker
		critic_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
		torch.nn.utils.clip_grad_norm_(self.masker.parameters(), max_norm=10.0)
		self.critic_optimizer.step()
		self.masker_optimizer.step()
		###

	def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
		obs_masked = self.apply_mask(obs)
		logits = self.actor(obs_masked, detach=True)
		action_probs = F.softmax(logits, dim=-1)
		log_probs = F.log_softmax(logits, dim=-1)
		# Detach critic when optimizing the actor
		actor_Q1_all, actor_Q2_all = self.critic(obs_masked, action=None, detach=True)
		actor_Q_all = torch.min(actor_Q1_all, actor_Q2_all)
		expected_Q = torch.sum(action_probs * actor_Q_all, dim=-1, keepdim=True)
		entropy = -torch.sum(action_probs * log_probs, dim=-1, keepdim=True)
		actor_loss = -(expected_Q + self.alpha.detach() * entropy).mean()

		if L is not None:
			L.log('train_actor/loss', actor_loss, step)
			L.log('train_actor/entropy', entropy.mean(), step)

		self.actor_optimizer.zero_grad()
		###
		# Stabilizer: gradient clipping on actor
		actor_loss.backward()
		torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
		self.actor_optimizer.step()
		###

		if update_alpha:
			self.log_alpha_optimizer.zero_grad()
			target_entropy = self.target_entropy if hasattr(self, 'target_entropy') else np.log(self.num_actions)
			alpha_loss = (self.alpha * (entropy.detach() - target_entropy)).mean()

			if L is not None:
				L.log('train_alpha/loss', alpha_loss, step)
				L.log('train_alpha/value', self.alpha, step)

			alpha_loss.backward()
			self.log_alpha_optimizer.step()

# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import utils
# from algorithms.vit.svea_vit import SVEA_ViT
# import algorithms.modules as m
# from video import AugmentationRecorder
# from augmentations import strong_augment


# class MaDi_ViT(SVEA_ViT):
#     """MaDi: Masking Distractions for Generalization in Reinforcement Learning
#     Vision Transformer backbone adapted for discrete action spaces (one-hot actions)"""
    
#     def __init__(self, obs_shape, action_shape, args, aug_recorder: AugmentationRecorder):
#         # For discrete actions, action_shape should be (15,) for 15 possible actions
#         super().__init__(obs_shape, action_shape, args)
        
#         self.masker = m.MaskerNet(obs_shape, args).cuda()
#         self.masker_optimizer = torch.optim.Adam(
#             self.masker.parameters(), lr=args.masker_lr, betas=(args.masker_beta, 0.999)
#         )
#         self.num_masks = args.frame_stack
#         self.aug_recorder = aug_recorder
#         self.actor = DiscreteActor()
#         self.critic=  DiscreteCritic()
        
#         # For discrete actions
#         self.num_actions = action_shape[0]  # Should be 15
        
#         # You'll need to modify the actor network to output discrete action probabilities
#         # This assumes your base SVEA_ViT actor is modified for discrete actions
#         # The actor should output logits of shape (batch_size, num_actions)

#     def apply_mask(self, obs, test_env=False):
#         # Same as original - no changes needed for masking logic
#         frames = obs.chunk(self.num_masks, dim=1)
#         frames_cat = torch.cat(frames, dim=0)
#         masks_cat = self.masker(frames_cat, test_env=test_env)
#         masks = masks_cat.chunk(self.num_masks, dim=0)
#         masked_frames = [m * f for m, f in zip(masks, frames)]
#         return torch.cat(masked_frames, dim=1)

#     def select_action(self, obs, test_env=False):
#         """Select action greedily (for evaluation)"""
#         _obs = self._obs_to_input(obs)
#         _obs = self.apply_mask(_obs, test_env)
#         with torch.no_grad():
#             # Actor should return logits for discrete actions
#             action_logits = self.actor(_obs)
#             action_probs = F.softmax(action_logits, dim=-1)
#             # Select action with highest probability
#             action_idx = torch.argmax(action_probs, dim=-1)
            
#             # Convert to one-hot encoding
#             action_onehot = F.one_hot(action_idx, num_classes=self.num_actions).float()
            
#         return action_onehot.cpu().data.numpy().flatten()

#     def sample_action(self, obs):
#         """Sample action from policy distribution (for training)"""
#         _obs = self._obs_to_input(obs)
#         _obs = self.apply_mask(_obs)
#         with torch.no_grad():
#             # Actor should return logits for discrete actions
#             action_logits = self.actor(_obs)
#             action_probs = F.softmax(action_logits, dim=-1)
            
#             # Sample from categorical distribution
#             action_dist = torch.distributions.Categorical(probs=action_probs)
#             action_idx = action_dist.sample()
            
#             # Convert to one-hot encoding
#             action_onehot = F.one_hot(action_idx, num_classes=self.num_actions).float()
            
#         return action_onehot.cpu().data.numpy().flatten()

#     def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
#         """Update critic - modified for discrete actions"""
#         with torch.no_grad():
#             next_obs_masked = self.apply_mask(next_obs)
            
#             # Get next action probabilities from actor
#             next_action_logits = self.actor(next_obs_masked)
#             next_action_probs = F.softmax(next_action_logits, dim=-1)
            
#             # For discrete actions, we can compute expected Q-value over all actions
#             target_Q1_all, target_Q2_all = self.critic_target(next_obs_masked)  # Shape: (batch, num_actions)
#             target_Q1 = torch.sum(next_action_probs * target_Q1_all, dim=-1, keepdim=True)
#             target_Q2 = torch.sum(next_action_probs * target_Q2_all, dim=-1, keepdim=True)
            
#             # Entropy term for discrete actions
#             log_probs = F.log_softmax(next_action_logits, dim=-1)
#             entropy = -torch.sum(next_action_probs * log_probs, dim=-1, keepdim=True)
            
#             target_V = torch.min(target_Q1, target_Q2) + self.alpha.detach() * entropy
#             target_Q = reward + (not_done * self.discount * target_V)

#         # Data augmentation
#         obs_aug = strong_augment(obs, self.augment, self.overlay_alpha)
#         self.aug_recorder.record(obs_aug, step)

#         if self.svea_alpha == self.svea_beta:
#             obs_combined = utils.cat(obs, obs_aug)
#             obs_combined = self.apply_mask(obs_combined)
#             self.aug_recorder.record(obs_combined[obs_combined.shape[0] // 2:], step, masked=True)
            
#             action_combined = utils.cat(action, action)
#             target_Q_combined = utils.cat(target_Q, target_Q)
            
#             # For discrete actions, critic takes one-hot encoded actions
#             current_Q1, current_Q2 = self.critic(obs_combined, action_combined)
#             critic_loss = (self.svea_alpha + self.svea_beta) * \
#                 (F.mse_loss(current_Q1, target_Q_combined) + F.mse_loss(current_Q2, target_Q_combined))
#         else:
#             obs_masked = self.apply_mask(obs)
#             current_Q1, current_Q2 = self.critic(obs_masked, action)
#             critic_loss = self.svea_alpha * \
#                 (F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
                
#             obs_aug_masked = self.apply_mask(obs_aug)
#             self.aug_recorder.record(obs_aug_masked, step, masked=True)
#             current_Q1_aug, current_Q2_aug = self.critic(obs_aug_masked, action)
#             critic_loss += self.svea_beta * \
#                 (F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

#         if L is not None:
#             L.log('train_critic/loss', critic_loss, step)

#         # Update both critic and masker
#         self.masker_optimizer.zero_grad()
#         self.critic_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_optimizer.step()
#         self.masker_optimizer.step()

#     def update_actor_and_alpha(self, obs, L=None, step=None, update_alpha=True):
#         """Update actor and alpha - modified for discrete actions"""
#         obs_masked = self.apply_mask(obs)
        
#         # Get action logits from actor
#         action_logits = self.actor(obs_masked, detach=True)
#         action_probs = F.softmax(action_logits, dim=-1)
#         log_probs = F.log_softmax(action_logits, dim=-1)
        
#         # Get Q-values for all actions
#         actor_Q1_all, actor_Q2_all = self.critic(obs_masked, detach=True)  # Shape: (batch, num_actions)
#         actor_Q_all = torch.min(actor_Q1_all, actor_Q2_all)
        
#         # Compute policy loss using expected Q-value
#         expected_Q = torch.sum(action_probs * actor_Q_all, dim=-1, keepdim=True)
#         entropy = -torch.sum(action_probs * log_probs, dim=-1, keepdim=True)
#         actor_loss = -(expected_Q + self.alpha.detach() * entropy).mean()

#         if L is not None:
#             L.log('train_actor/loss', actor_loss, step)
#             L.log('train_actor/entropy', entropy.mean(), step)

#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         if update_alpha:
#             # Alpha loss for discrete actions
#             self.log_alpha_optimizer.zero_grad()
#             # Target entropy for discrete actions is typically log(num_actions)
#             target_entropy = np.log(self.num_actions) if not hasattr(self, 'target_entropy') else self.target_entropy
#             alpha_loss = (self.alpha * (entropy.detach() - target_entropy)).mean()

#             if L is not None:
#                 L.log('train_alpha/loss', alpha_loss, step)
#                 L.log('train_alpha/value', self.alpha, step)

#             alpha_loss.backward()
#             self.log_alpha_optimizer.step()


# # Additional modifications you'll need to make to your base SVEA_ViT:

# class DiscreteActor(nn.Module):
#     """Example discrete actor network for MaDi_ViT_Discrete"""
    
#     def __init__(self, obs_encoder, num_actions, hidden_dim=256):
#         super().__init__()
#         self.obs_encoder = obs_encoder  # Your ViT encoder
#         self.num_actions = num_actions
        
#         # Get the output dimension of your ViT encoder
#         encoder_dim = obs_encoder.output_dim  # You'll need to check this
        
#         self.policy_head = nn.Sequential(
#             nn.Linear(encoder_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, num_actions)
#         )
    
#     def forward(self, obs, detach=False):
#         features = self.obs_encoder(obs)
#         if detach:
#             features = features.detach()
        
#         action_logits = self.policy_head(features)
#         return action_logits


# class DiscreteCritic(nn.Module):
#     """Example discrete critic network for MaDi_ViT_Discrete"""
    
#     def __init__(self, obs_encoder, num_actions, hidden_dim=256):
#         super().__init__()
#         self.obs_encoder = obs_encoder
#         self.num_actions = num_actions
        
#         encoder_dim = obs_encoder.output_dim
        
#         # Q1 and Q2 networks for double Q-learning
#         self.Q1 = nn.Sequential(
#             nn.Linear(encoder_dim + num_actions, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
        
#         self.Q2 = nn.Sequential(
#             nn.Linear(encoder_dim + num_actions, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 1)
#         )
    
#     def forward(self, obs, action=None, detach=False):
#         features = self.obs_encoder(obs)
#         if detach:
#             features = features.detach()
            
#         if action is not None:
#             # Standard Q(s,a) evaluation with one-hot actions
#             features_action = torch.cat([features, action], dim=-1)
#             q1 = self.Q1(features_action)
#             q2 = self.Q2(features_action)
#             return q1, q2
#         else:
#             # Return Q-values for all actions Q(s,a) for a in all_actions
#             batch_size = features.shape[0]
#             q1_all = []
#             q2_all = []
            
#             for a in range(self.num_actions):
#                 action_onehot = F.one_hot(torch.tensor([a]).cuda(), num_classes=self.num_actions).float()
#                 action_onehot = action_onehot.repeat(batch_size, 1)
#                 features_action = torch.cat([features, action_onehot], dim=-1)
                
#                 q1_all.append(self.Q1(features_action))
#                 q2_all.append(self.Q2(features_action))
            
#             return torch.cat(q1_all, dim=-1), torch.cat(q2_all, dim=-1)