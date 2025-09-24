import jax
import numpy as np
import jax.numpy as jnp
import jax.nn as jnn
import random as py_rand
from jax import random
tree_map = jax.tree_util.tree_map
sg = lambda x: tree_map(jax.lax.stop_gradient, x)
import itertools
import logging

logger = logging.getLogger()


class CheckTypesFilter(logging.Filter):
    def filter(self, record):
        return "check_types" not in record.getMessage()


logger.addFilter(CheckTypesFilter())

from . import behaviors, jaxagent, jaxutils, nets
from . import ninjax as nj


@jaxagent.Wrapper
class Agent(nj.Module):
    def __init__(self, obs_space, act_space, step, config):
        self.config = config
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.step = step
        self.wm = WorldModel(obs_space, act_space, config, name="wm")
        self.task_behavior = getattr(behaviors, config.task_behavior)(self.wm, self.act_space, self.config, name="task_behavior")
        if config.expl_behavior == "None":
            self.expl_behavior = self.task_behavior
        else:
            self.expl_behavior = getattr(behaviors, config.expl_behavior)(self.wm, self.act_space, self.config, name="expl_behavior")

    def policy_initial(self, batch_size):
        return (
            self.wm.initial(batch_size),
            self.task_behavior.initial(batch_size),
            self.expl_behavior.initial(batch_size),
        )

    def train_initial(self, batch_size):
        return self.wm.initial(batch_size)
    
    def get_recon_imgs(self, obs, posterior, prior, key='birdeye_wpt'):
        reconstruct_prior = self.wm.heads['decoder'](prior)[key].mode()
        reconstruct_post = self.wm.heads['decoder'](posterior)[key].mode()
        truth = obs[key]
        
        return truth, reconstruct_prior, reconstruct_post

    def policy(self, obs, state, mode="train"):

        self.config.jax.jit and print("Tracing policy function.")
        obs = self.preprocess(obs)
        # (prev_latent, prev_action), task_state, expl_state, stage = state
        if len(state) == 4:
            (prev_latent, prev_action), task_state, expl_state, stage = state
        else:
            (prev_latent, prev_action), task_state, expl_state = state
            stage = jnp.array(0) # or some default value
        embed = self.wm.encoder(obs)
        latent, prior_orig = self.wm.rssm.obs_step(prev_latent, prev_action, embed, obs["is_first"])
        if mode not in ['train', 'eval', 'explore']:
            available_keys = ["birdeye_gt","birdeye_raw","birdeye_with_traffic_lights","birdeye_wpt","camera","lidar"]
            if 'surprise' in mode:
                used_keys = ['None'] + available_keys
                used_keys = np.array(used_keys)
                total_newlat_seperate = []
                surprises = []
                # Compute baseline surprise
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                if 'full' in mode:
                    # Iterate over all combinations of keys (masking 1, 2, ..., N keys)
                    for r in range(1, len(available_keys) + 1):
                        for combo in itertools.combinations(available_keys, r):
                            obs_masked = jax.tree_map(lambda x: x, obs)
                            
                            # Mask all keys in the current combination
                            for key in combo:
                                obs_masked[key] = jnp.zeros(obs[key].shape, dtype=float)

                            # Encode and compute surprise
                            embed = self.wm.encoder(obs_masked)
                            temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                                prev_latent, prev_action, embed, obs['is_first']
                            )
                            surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                                self.wm.rssm.get_dist(temp_prior_orig)
                            )

                            surprises.append(surprise)
                            total_newlat_seperate.append(temp_latent)
                else: #Run N version
                    # Mask each key individually and compute surprise
                    # for ite, image_key in enumerate(available_keys): #Masks each one at a time:
                    #     obs_masked = jax.tree_map(lambda x: x, obs)
                    #     obs_masked[image_key] = jnp.zeros(obs[image_key].shape, dtype=float)
                    #     embed = self.wm.encoder(obs_masked)

                    #     temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                    #     prev_latent, prev_action, embed, obs['is_first'])

                    #     surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                    #         self.wm.rssm.get_dist(temp_prior_orig))
                    
                        
                    #     surprises.append(surprise)
                    #     total_newlat_seperate.append(temp_latent)
                    for ite, image_key in enumerate(available_keys): #Keeps only one sensor at a time:
                        # Start with all observations as zeros
                        obs_masked = jax.tree_map(lambda x: jnp.zeros_like(x), obs)
                        
                        # Keep only the current image_key unmasked (use original data)
                        obs_masked[image_key] = obs[image_key]
                        
                        embed = self.wm.encoder(obs_masked)

                        temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                            prev_latent, prev_action, embed, obs['is_first'])

                        surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                            self.wm.rssm.get_dist(temp_prior_orig))

                        surprises.append(surprise)
                        total_newlat_seperate.append(temp_latent)

                    #Sort sensors by surprise (lowest surprise first)
                    surprises_init = jnp.array(surprises)
                    surprises_init = surprises_init.flatten()
                    sorted_indices = jnp.argsort(surprises_init, descending=True)

                    #Create N more latents by iteratively masking according to the order of sorted indices:
                    obs_iter = jax.tree_map(lambda x: x, obs)
                    for i in range(len(sorted_indices)):
                        idx = sorted_indices[i] # 
                        # Use conditional masking for each key based on sorted order
                        for j, key_name in enumerate(available_keys):
                            # Check if this key should be masked at this step (idx matches the key's position)
                            should_mask = (idx == j)
                            obs_iter[key_name] = jnp.where(
                                should_mask,
                                jnp.zeros_like(obs_iter[key_name]),
                                obs_iter[key_name]
                            )
                        
                        embed = self.wm.encoder(obs_iter)
                        temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                            prev_latent, prev_action, embed, obs['is_first'])
                        surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                            self.wm.rssm.get_dist(temp_prior_orig))
                        
                        surprises.append(surprise)
                        total_newlat_seperate.append(temp_latent)
                
                total_newlat_seperate.append(latent)
                surprises.append(base_surprise)
                # Find index of minimum surprise
                surprises = jnp.array(surprises)
                min_idx = jnp.argmin(surprises)
                
                # Stack candidates and select using advanced indexing
                stacked_latents = jax.tree_map(
                    lambda *candidates: jnp.stack(candidates, axis=0),
                    *total_newlat_seperate
                )
            
                # Select the latent with lowest surprise
                latent = jax.tree_map(
                    lambda stacked: stacked[min_idx],
                    stacked_latents
                )
                latent = {k: v[min_idx] for k, v in stacked_latents.items() if k != "image_key"} # I dont think this does anything.

            elif 'random' in mode: # mode == 'random':
                k = py_rand.randint(0, len(available_keys))  # actual number of keys chosen
                chosen_keys = py_rand.sample(available_keys, k)
                obs_masked = jax.tree_map(lambda x: x, obs)
                for ite, image_key in enumerate(chosen_keys):
                    obs_masked[image_key] = jnp.zeros(obs[image_key].shape, dtype=float)
                embed = self.wm.encoder(obs_masked)
                latent, temp_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, obs['is_first'])
                
            elif 'sample' in mode:
                # Smart gradient-based dropout instead of brute force search
                surprises = []
                total_newlat_seperate = []
                # First, get baseline surprise with original observation
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                
                
                # Smart dropout function integrated into your existing structure
                def compute_surprise_for_obs(obs_input):
                    """Compute surprise for given observation - used for gradient computation"""
                    embed = self.wm.encoder(obs_input)
                    temp_latent, temp_prior = self.wm.rssm.obs_step(
                        prev_latent, prev_action, embed, obs_input['is_first']
                    )
                    surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                        self.wm.rssm.get_dist(temp_prior)
                    )
                    return surprise.mean()  # Return scalar for gradient computation
                
                # Compute gradients to find harmful pixels
                surprise_grad_fn = jax.grad(compute_surprise_for_obs)
                #Check
                
                # Get image key
                image_key = self.wm.config.encoder.cnn_keys#[0]
                
                # 0. Sample from the prior to get some image of the prior | Done
                # 1. Compute surprise gradients | Done
                # 2. Apply the modified tanh function to the surprise gradients
                # 3. (1 - surpise_gradients) * obs_input + surprise_gradients * prior_orig

                # Try different dropout ratios, but use smart selection
                # target_dropout_ratios = np.linspace(0.65, 1, 50)  # Fewer samples, smarter selection

                truth, prior_img, reconstruct_post = self.get_recon_imgs(obs, latent, prior_orig, image_key)

                gradients = surprise_grad_fn(obs)

                def tanh(gradients):
                    normalized = (gradients - gradients.min()) / (gradients.max() - gradients.min() + 1e-8)
                    # normalized = jnn.sigmoid(gradients - 0.5)
                    normalized = 2.0 * jnp.power(normalized - 0.5, 3.0) + 0.5
                    mean = jnp.max(gradients)
                    return normalized, mean
                
                def activation_experimental(gradients):
                    
                    normalized = jnn.sigmoid(gradients)
                    # normalized = 2.0 * jnp.power(nor - 0.5, 3.0) + 0.5
                    mean = jnp.mean(gradients)
                    return normalized, mean

                normalized_gradients, mean_gradients = tanh(jnp.abs(gradients[image_key]))


                sampled_obs = jax.tree_map(lambda x: x, obs)
                interpolated_img = (1 - normalized_gradients) * obs[image_key] + normalized_gradients * prior_img
                sampled_obs[image_key] = interpolated_img
                interpolated_img = jnp.array(interpolated_img)
                video = jnp.concatenate([truth, prior_img, reconstruct_post, interpolated_img], axis=2)

                # Compute latent and surprise for this dropout configuration
                embed = self.wm.encoder(sampled_obs)
                temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, obs['is_first']
                )
                surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                    self.wm.rssm.get_dist(temp_prior_orig)
                )
                    
                surprises.append(surprise)
                total_newlat_seperate.append(temp_latent)
 
                # Add original observation (no dropout) as final candidate
                total_newlat_seperate.append(latent)
                surprises.append(base_surprise)
                
                # Select best configuration
                surprises = jnp.array(surprises)
                min_idx = jnp.argmin(surprises)
                
                # Stack candidates and select using advanced indexing
                stacked_latents = jax.tree_map(
                    lambda *candidates: jnp.stack(candidates, axis=0),
                    *total_newlat_seperate
                )
                
                latent = temp_latent
                # self.wm.surprise_mean = jax.lax.cond(
                #     obs['is_first'],
                #     lambda: 0,
                #     lambda: base_surprise 
                # )
                # self.wm.surprise_mean = base_surprise
                # # Select the latent with lowest surprise
                # latent = jax.tree_map(
                #     lambda stacked: stacked[min_idx],
                #     stacked_latents
                # )
            elif 'reject' in mode:
                # Smart gradient-based dropout instead of brute force search
                surprises = []
                total_newlat_seperate = []
                # First, get baseline surprise with original observation
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                
                
                # # Smart dropout function integrated into your existing structure
                # def compute_surprise_for_obs(obs_input):
                #     """Compute surprise for given observation - used for gradient computation"""
                #     embed = self.wm.encoder(obs_input)
                #     temp_latent, temp_prior = self.wm.rssm.obs_step(
                #         prev_latent, prev_action, embed, obs_input['is_first']
                #     )
                #     surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                #         self.wm.rssm.get_dist(temp_prior)
                #     )
                #     return surprise.mean()  # Return scalar for gradient computation
                
                # Compute gradients to find harmful pixels
                # surprise_grad_fn = jax.grad(compute_surprise_for_obs)
                #Check
                
                # Get image key
                image_key = self.wm.config.encoder.cnn_keys#[0]
                
                # 0. Sample from the prior to get some image of the prior | Done
                # 1. Compute surprise gradients | Done
                # 2. Apply the modified tanh function to the surprise gradients
                # 3. (1 - surpise_gradients) * obs_input + surprise_gradients * prior_orig

                # Try different dropout ratios, but use smart selection
                # target_dropout_ratios = np.linspace(0.65, 1, 50)  # Fewer samples, smarter selection
                batch_size = 1  # or however you determine batch size
                is_first = jnp.ones((batch_size,), dtype=jnp.float32)

                dropped_residual_latent, dropped_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, is_first
                )
                truth, reconstruct_post, reconstruct_post_dropped = self.get_recon_imgs(obs, dropped_residual_latent, latent, image_key)

                # normalized_gradients, mean_gradients = tanh(jnp.abs(gradients[image_key]))
                #Do we take the latent or the dropped latent.
                # if stage == 0:
                #     latent, stage = jax.lax.cond(
                #         jnp.mean(jnp.abs(obs[image_key] - reconstruct_post_dropped)) < 1,
                #         lambda: (latent, 0),
                #         lambda: (prior_orig, 1) 
                #     )
                # elif stage == 1:
                #     latent, stage = jax.lax.cond(
                #         jnp.mean(jnp.abs(obs[image_key] - reconstruct_post_dropped)) < 1,
                #         lambda: (dropped_residual_latent, 0),
                #         lambda: (prior_orig, 1)
                #     )

                condition = jnp.mean(jnp.abs(obs[image_key] - reconstruct_post_dropped)) < .1

                latent, stage = jax.lax.cond(
                    stage == jnp.array(0),
                    # When stage == 0
                    lambda: jax.lax.cond(
                        condition,
                        lambda: (latent, jnp.array(0)),
                        lambda: (prior_orig, jnp.array(1))
                    ),
                    # When stage != 0 (assuming stage == 1)
                    lambda: jax.lax.cond(
                        condition,
                        lambda: (dropped_residual_latent, jnp.array(0)),
                        lambda: (prior_orig, jnp.array(1))
                    )
                )

                _, prior_img, interpolated_img = self.get_recon_imgs(obs, latent, prior_orig, image_key)

                # interpolated_img = (1 - normalized_gradients) * obs[image_key] + normalized_gradients * prior_img

                # sampled_obs[image_key] = interpolated_img
                # interpolated_img = jnp.array(interpolated_img)

                video = jnp.concatenate([truth, prior_img, reconstruct_post, reconstruct_post_dropped, interpolated_img], axis=2)
                

                # # Compute latent and surprise for this dropout configuration
                # embed = self.wm.encoder(sampled_obs)
                # temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                #     prev_latent, prev_action, embed, obs['is_first']
                # )
                # surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                #     self.wm.rssm.get_dist(temp_prior_orig)
                # )
                    
                # surprises.append(surprise)
                # total_newlat_seperate.append(temp_latent)
 
                # # Add original observation (no dropout) as final candidate
                # total_newlat_seperate.append(latent)
                # surprises.append(base_surprise)
                
                # # Select best configuration
                # surprises = jnp.array(surprises)
                # min_idx = jnp.argmin(surprises)
                
                # # Stack candidates and select using advanced indexing
                # stacked_latents = jax.tree_map(
                #     lambda *candidates: jnp.stack(candidates, axis=0),
                #     *total_newlat_seperate
                # )
                
                # latent = temp_latent
                # self.wm.surprise_mean = jax.lax.cond(
                #     obs['is_first'],
                #     lambda: 0,
                #     lambda: base_surprise 
                # )
                # self.wm.surprise_mean = base_surprise
                # # Select the latent with lowest surprise
                # latent = jax.tree_map(
                #     lambda stacked: stacked[min_idx],
                #     stacked_latents
                # )
            
            elif 'filter' in mode:
                # Smart gradient-based dropout instead of brute force search
                surprises = []
                total_newlat_seperate = []
                # First, get baseline surprise with original observation
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                
                # Smart dropout function integrated into your existing structure
                def compute_surprise_for_obs(obs_input):
                    """Compute surprise for given observation - used for gradient computation"""
                    embed = self.wm.encoder(obs_input)
                    temp_latent, temp_prior = self.wm.rssm.obs_step(
                        prev_latent, prev_action, embed, obs_input['is_first']
                    )
                    surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                        self.wm.rssm.get_dist(temp_prior)
                    )
                    return surprise.mean()  # Return scalar for gradient computation
                
                def median_blur(image: jnp.ndarray, ksize: int) -> jnp.ndarray:
                    """
                    Apply median blur to a batch of images (NHWC format).

                    Args:
                        image: jnp.ndarray of shape (N, H, W, C)
                        ksize: odd integer kernel size (e.g. 3, 5, 7)

                    Returns:
                        Blurred image of same shape as input.
                    """
                    assert ksize % 2 == 1, "ksize must be odd"
                    pad = ksize // 2
                    N, H, W, C = image.shape

                    # Pad with reflect to mimic OpenCV behavior
                    padded = jnp.pad(
                        image, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="reflect"
                    )

                    # Extract patches using dynamic slicing
                    def get_patch(n, i, j):
                        return jax.lax.dynamic_slice(
                            padded, (n, i, j, 0), (1, ksize, ksize, C)
                        ).reshape(-1, C)

                    # Meshgrid over image coords
                    ii, jj = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")

                    # Vectorize over batch, height, width
                    patches = jax.vmap(
                        lambda n: jax.vmap(
                            lambda i_row, j_row: jax.vmap(
                                get_patch, in_axes=(None, 0, 0)
                            )(n, i_row, j_row),
                            in_axes=(0, 0),
                        )(ii, jj),
                        in_axes=(0,),
                    )(jnp.arange(N))  # (N, H, W, ksize*ksize, C)

                    # Take median across neighborhood axis
                    result = jnp.median(patches, axis=3)  # (N, H, W, C)
                    return result
                # Compute gradients to find harmful pixels
                surprise_grad_fn = jax.grad(compute_surprise_for_obs)
                #Check
                
                # Get image key
                image_key = self.wm.config.encoder.cnn_keys#[0]
                

                sampled_obs = jax.tree_map(lambda x: x, obs)
                sampled_obs[image_key] = median_blur(sampled_obs[image_key],ksize=3)
                embed = self.wm.encoder(sampled_obs)
                temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                    prev_latent, prev_action, embed, obs['is_first']
                )
                latent = temp_latent
                
                # # Select the latent with lowest surprise
                # latent = jax.tree_map(
                #     lambda stacked: stacked[min_idx],
                #     stacked_latents
                # )

            elif 'sample_v0' in mode:
                # Smart gradient-based dropout instead of brute force search
                
                # First, get baseline surprise with original observation
                base_surprise = self.wm.rssm.get_dist(latent).kl_divergence(self.wm.rssm.get_dist(prior_orig))
                
                # Smart dropout function integrated into your existing structure
                def compute_surprise_for_obs(obs_input):
                    """Compute surprise for given observation - used for gradient computation"""
                    embed = self.wm.encoder(obs_input)
                    temp_latent, temp_prior = self.wm.rssm.obs_step(
                        prev_latent, prev_action, embed, obs_input['is_first']
                    )
                    surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                        self.wm.rssm.get_dist(temp_prior)
                    )
                    return surprise.mean()  # Return scalar for gradient computation
                
                # Compute gradients to find harmful pixels
                surprise_grad_fn = jax.grad(compute_surprise_for_obs)
                #Check
                
                # Get image key
                image_key = self.wm.config.encoder.cnn_keys#[0]
                
                # Try different dropout ratios, but use smart selection
                # target_dropout_ratios = np.linspace(0.65, 1, 50)  # Fewer samples, smarter selection
                n_samples = 70
                target_dropout_ratios = np.linspace(0.01, 1, n_samples)
                
                total_newlat_seperate = []
                surprises = []
                key = random.PRNGKey(42)
                done = False  # Python bool at start
                surprise_threshold = 3.0  # Example threshold, tune based on your domain
                gradient_magnitude_delta = 10.0
                old_gradient_magnitude = jnp.array(0.0)

                for target_ratio in target_dropout_ratios:
                    try:
                        # print('Trying: ',target_ratio)
                        # Compute pixel gradients
                        pixel_gradients = surprise_grad_fn(obs)
                        
                        # Get importance scores for the image
                        pixel_importance = jnp.abs(pixel_gradients[image_key])
                        batch_size, H, W, C = obs[image_key].shape
                        
                        # Flatten importance scores
                        flat_importance = pixel_importance.reshape(batch_size, -1)
                        
                        # Number of pixels to drop based on target ratio
                        n_drop = int(target_ratio * H * W * C)
                        
                        if n_drop > 0:
                            # Get indices of most harmful pixels (highest gradient magnitude)
                            harmful_indices = jnp.argpartition(flat_importance, -n_drop, axis=1)[:, -n_drop:]
                            
                            # Create dropout mask
                            dropout_mask = jnp.zeros_like(flat_importance, dtype=bool)
                            batch_indices = jnp.arange(batch_size)[:, None]
                            dropout_mask = dropout_mask.at[batch_indices, harmful_indices].set(True)
                            dropout_mask = dropout_mask.reshape(batch_size, H, W, C)
                            
                            # Apply smart dropout
                            sampled_obs = jax.tree_map(lambda x: x, obs)
                            sampled_obs[image_key] = jnp.where(dropout_mask, 0.0, obs[image_key])  # Set to 0 instead of mask_value
                        else:
                            # No dropout for very small ratios
                            sampled_obs = jax.tree_map(lambda x: x, obs)
                        
                    except Exception as e:
                        # Fallback to random dropout if gradient computation fails
                        print(f"Gradient computation failed, using random dropout: {e}")
                        sampled_obs = jax.tree_map(lambda x: x, obs)
                        sampled_obs = self.wm.randomly_dropout_pixels_diffuse(sampled_obs, key, mask_value=target_ratio)
                    
                    # Compute latent and surprise for this dropout configuration
                    embed = self.wm.encoder(sampled_obs)
                    temp_latent, temp_prior_orig = self.wm.rssm.obs_step(
                        prev_latent, prev_action, embed, obs['is_first']
                    )
                    surprise = self.wm.rssm.get_dist(temp_latent).kl_divergence(
                        self.wm.rssm.get_dist(temp_prior_orig)
                    )
                    # Masking trick: only keep if not "done" yet
                    keep = jnp.logical_not(done)

                    # surprises.append(jnp.where(keep, surprise, jnp.inf))
                    # total_newlat_seperate.append(
                    #     jax.tree_map(lambda x: jnp.where(keep, x, jnp.nan), temp_latent)
                    # )
                    
                    # Update done flag (once True, stays True)
                    # done = jnp.logical_or(done, surprise < surprise_threshold)
                    gradient_magnitude = jnp.mean(jnp.abs(pixel_gradients[image_key]))
                    done = jnp.logical_or(done, gradient_magnitude-old_gradient_magnitude < gradient_magnitude_delta)
                    old_gradient_magnitude = gradient_magnitude
                    
                    
                    
                    
                    surprises.append(surprise)
                    total_newlat_seperate.append(temp_latent)
                # Assuming `keep` is a boolean array or pytree matching temp_latent
                n_levels = jnp.sum(keep)  # counts all True
                # Add original observation (no dropout) as final candidate
                total_newlat_seperate.append(latent)
                surprises.append(base_surprise)
                
                # Select best configuration
                surprises = jnp.array(surprises)
                min_idx = jnp.argmin(surprises)
                
                # Stack candidates and select using advanced indexing
                stacked_latents = jax.tree_map(
                    lambda *candidates: jnp.stack(candidates, axis=0),
                    *total_newlat_seperate
                )

                # Select the latent with lowest surprise
                best_latent = jax.tree_map(
                    lambda stacked: stacked[min_idx],
                    stacked_latents
                )
                
                # Check if minimum surprise is still too high
                min_surprise = surprises[min_idx]
                
                # Ensure min_surprise is a scalar
                min_surprise_scalar = jnp.squeeze(min_surprise)  # Remove dimensions of size 1
                # or alternatively:
                # min_surprise_scalar = min_surprise.item()  # Convert to Python scalar

                # # Use prior_orig if surprise exceeds threshold, otherwise use best latent
                latent = jax.lax.cond(
                    min_surprise_scalar > surprise_threshold,
                    lambda: prior_orig,  # If surprise too high, use prior
                    lambda: best_latent  # Otherwise use best dropout result
                )
                # predicate = jnp.logical_or(
                #     min_surprise_scalar > surprise_threshold,
                #     n_levels > jnp.squeeze(n_samples*.9)
                # )

                # latent = jax.lax.cond(
                #     predicate,
                #     lambda _: prior_orig,
                #     lambda _: best_latent,
                #     operand=None  # pass None because lambdas expect one argument
                # )
                    

        self.expl_behavior.policy(latent, expl_state)
        task_outs, task_state = self.task_behavior.policy(latent, task_state)
        expl_outs, expl_state = self.expl_behavior.policy(latent, expl_state)
        
        def reshape_outputs(surprises_dict):
            reshaped = {}
            for k, v in surprises_dict.items():
                if k.startswith('log_surprise'):
                    # reshape scalar [] or any shape to [1]
                    reshaped[k] = jnp.reshape(v, (1,))
                if k.startswith('mu_gradients'):
                    reshaped[k] = jnp.reshape(v, (1,))
                else:
                    reshaped[k] = v
            return reshaped

        if mode == "eval":
            outs = task_outs
            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])
            surprise = lambda post, prior: self.wm.rssm.get_dist(post).kl_divergence(
                self.wm.rssm.get_dist(prior)
            )
            image_key = 'birdeye_wpt'
            truth, prior_img, reconstruct_post = self.get_recon_imgs(obs, latent, prior_orig, image_key)
            video = jnp.concatenate([truth, prior_img, reconstruct_post], axis=2)
            video = jnp.expand_dims(video, 0)
            # metrics = {}
            # outs.update({f"openl_{key}" : jaxutils.video_grid(video)})
            # outs.update(jaxutils.tensorstats(surprise(latent, prior_orig), "log_surprise"))
            # print(outs)
            outs = reshape_outputs(outs)
            print({k: v.shape for k, v in outs.items()})

            outs.update({f"openl_custom_{image_key}" : jaxutils.video_grid(video)})

        elif mode == "explore":
            outs = expl_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        elif mode == "train":
            outs = task_outs
            outs["log_entropy"] = outs["action"].entropy()
            outs["action"] = outs["action"].sample(seed=nj.rng())
        else: #Run eval by default
            outs = task_outs
            if mode in ['sample','reject']:
                video = jnp.expand_dims(video, 0)
                # interpolated_img = jnp.expand_dims(interpolated_img, 0)
                # mean_gradients = reshape_outputs({"mu_gradients": mean_gradients})["mu_gradients"]
                # outs.update({f"openl_custom_interpolated_{image_key}": jaxutils.video_grid(interpolated_img)})
                outs.update({f"openl_custom_{image_key}" : jaxutils.video_grid(video)})
                # outs.update({"mu_gradients": mean_gradients})
                # outs.update({"gradients_exact":gradients[image_key]})

            outs["action"] = outs["action"].sample(seed=nj.rng())
            outs["log_entropy"] = jnp.zeros(outs["action"].shape[:1])

        state = ((latent, outs["action"]), task_state, expl_state, stage)
        return outs, state

    def train(self, data, state):
        self.config.jax.jit and print("Tracing train function.")
        metrics = {}
        data = self.preprocess(data)
        state, wm_outs, mets = self.wm.train(data, state)
        metrics.update(mets)
        context = {**data, **wm_outs["post"]}
        start = tree_map(lambda x: x.reshape([-1] + list(x.shape[2:])), context)
        _, mets = self.task_behavior.train(self.wm.imagine, start, context)
        metrics.update(mets)
        if self.config.expl_behavior != "None":
            _, mets = self.expl_behavior.train(self.wm.imagine, start, context)
            metrics.update({"expl_" + key: value for key, value in mets.items()})

        if "keyA" in data.keys():
            outs = {
                "key": data["key"],
                "env_step": data["env_step"],
                "model_loss": metrics["model_loss_raw"].copy(),
                "td_error": metrics["td_error"].copy(),
            }

        else:
            outs = {}

        # Don't need the full model_loss_raw or td_error after the priority calculation, summarize it.
        metrics.update({"model_loss_raw": metrics["model_loss_raw"].mean()})
        metrics.update({"td_error": metrics["td_error"].mean()})

        return outs, state, metrics

    def report(self, data):
        self.config.jax.jit and print("Tracing report function.")
        data = self.preprocess(data)
        report = {}
        report.update(self.wm.report(data))
        mets = self.task_behavior.report(data)
        report.update({f"task_{k}": v for k, v in mets.items()})
        if self.expl_behavior is not self.task_behavior:
            mets = self.expl_behavior.report(data)
            report.update({f"expl_{k}": v for k, v in mets.items()})
        return report

    def preprocess(self, obs):
        obs = obs.copy()
        for key, value in obs.items():
            if key.startswith("log_") or key in ("key", "env_step"):
                continue
            if len(value.shape) > 3 and value.dtype == jnp.uint8:
                value = jaxutils.cast_to_compute(value) / 255.0
            else:
                value = value.astype(jnp.float32)
            obs[key] = value
        obs["cont"] = 1.0 - obs["is_terminal"].astype(jnp.float32)
        return obs


class WorldModel(nj.Module):
    def __init__(self, obs_space, act_space, config):
        self.obs_space = obs_space
        self.act_space = act_space["action"]
        self.config = config
        shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
        shapes = {k: v for k, v in shapes.items() if not k.startswith("log_")}
        self.encoder = nets.MultiEncoder(shapes, **config.encoder, name="enc")
        self.rssm = nets.RSSM(**config.rssm, name="rssm")
        self.heads = {
            "decoder": nets.MultiDecoder(shapes, **config.decoder, name="dec"),
            "reward": nets.MLP((), **config.reward_head, name="rew"),
            "cont": nets.MLP((), **config.cont_head, name="cont"),
        }
        self.opt = jaxutils.Optimizer(name="model_opt", **config.model_opt)
        scales = self.config.loss_scales.copy()
        image, vector = scales.pop("image"), scales.pop("vector")
        scales.update({k: image for k in self.heads["decoder"].cnn_shapes})
        scales.update({k: vector for k in self.heads["decoder"].mlp_shapes})
        self.scales = scales
        self.stage = 0

    def initial(self, batch_size):
        prev_latent = self.rssm.initial(batch_size)
        prev_action = jnp.zeros((batch_size, *self.act_space.shape))
        return prev_latent, prev_action

    def train(self, data, state):
        modules = [self.encoder, self.rssm, *self.heads.values()]
        mets, (state, outs, metrics) = self.opt(modules, self.loss, data, state, has_aux=True)
        metrics.update(mets)
        return state, outs, metrics
    
    def randomly_mask_images_per_timestep(self, data, key, mask_value=0.0):
        """
        Randomly mask 0 to n-1 images for each timestep, ensuring at least one image remains unmasked.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            mask_value: Value to use for masked pixels (default: 0.0)
        
        Returns:
            Modified data dictionary with randomly masked images
        """
        # Create a copy of the data to avoid modifying the original
        masked_data = data.copy()
        
        # Image keys to process
                # Image keys to process
        image_keys = [
            "birdeye_gt",
            "birdeye_raw",
            "birdeye_with_traffic_lights",
            "birdeye_wpt",
            "camera",
            "lidar"
        ]
        available_keys = [k for k in image_keys if k in data]
        # print('Loss on Keys: ',available_keys)
        if len(available_keys) == 0:
            return masked_data
        
        n_images = len(available_keys)
        batch_size, seq_len = data[available_keys[0]].shape[0], data[available_keys[0]].shape[1]  # 16, 64
        
        # Split keys
        key1, key2 = random.split(key)
        
        # For each (batch, timestep), randomly choose how many images to mask (0 to n-1)
        num_to_mask = random.randint(key1, shape=(batch_size, seq_len), minval=0, maxval=n_images)
        
        # Generate random values for each image at each (batch, timestep)
        # Shape: [batch_size, seq_len, n_images]
        random_vals = random.uniform(key2, shape=(batch_size, seq_len, n_images))
        
        # Sort the random values to get rankings (0 = smallest, n_images-1 = largest)
        rankings = jnp.argsort(random_vals, axis=-1)
        
        # Create a mask where we mask images with ranking < num_to_mask
        # This gives us a random selection of num_to_mask images
        mask_matrix = jnp.zeros((batch_size, seq_len, n_images), dtype=bool)
        
        for i in range(n_images):
            # For each image position, check if its ranking is less than num_to_mask
            image_rank = jnp.where(rankings == i, 
                                jnp.arange(n_images)[None, None, :], 
                                n_images)  # Set to n_images if not this image
            min_rank = jnp.min(image_rank, axis=-1)  # Get the ranking for image i
            should_mask = min_rank < num_to_mask
            mask_matrix = mask_matrix.at[:, :, i].set(should_mask)
        
        # Apply masks to each image
        for i, img_key in enumerate(available_keys):
            images = data[img_key]
            
            # Expand mask to match image dimensions
            mask = mask_matrix[:, :, i].reshape(batch_size, seq_len, 1, 1, 1)
            
            # Apply mask: where mask is True, replace with mask_value
            masked_images = jnp.where(mask, mask_value, images)
            
            # Update the data dictionary
            masked_data[img_key] = masked_images
            # print('Masking...')
        
        return masked_data
  
    def randomly_noise_images_per_timestep(self, data, key, mask_value=0.0):
        """
        Randomly mask 0 to n-1 images for each timestep, ensuring at least one image remains unmasked.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            mask_value: Value to use for masked pixels (default: 0.0)
        
        Returns:
            Modified data dictionary with randomly masked images
        """
        # Create a copy of the data to avoid modifying the original
        masked_data = data.copy()
        
        # Image keys to process
        image_keys = [
            "birdeye_gt",
            "birdeye_raw",
            "birdeye_with_traffic_lights",
            "birdeye_wpt",
            "camera",
            "lidar"
        ]
        
        available_keys = [k for k in image_keys if k in data]
        
        if len(available_keys) == 0:
            return masked_data
        
        n_images = len(available_keys)
        batch_size, seq_len = data[available_keys[0]].shape[0], data[available_keys[0]].shape[1]  # 16, 64
        
        # Split keys
        key1, key2 = random.split(key)
        
        # For each (batch, timestep), randomly choose how many images to mask (0 to n-1)
        num_to_mask = random.randint(key1, shape=(batch_size, seq_len), minval=0, maxval=n_images)
        
        # Generate random values for each image at each (batch, timestep)
        # Shape: [batch_size, seq_len, n_images]
        random_vals = random.uniform(key2, shape=(batch_size, seq_len, n_images))
        
        # Sort the random values to get rankings (0 = smallest, n_images-1 = largest)
        rankings = jnp.argsort(random_vals, axis=-1)
        
        # Create a mask where we mask images with ranking < num_to_mask
        # This gives us a random selection of num_to_mask images
        mean = 20.0
        std = 30.0
        mask_matrix = mean + std * random.normal(key2, (batch_size, seq_len, n_images))  # mean=0, std=1 #jnp.zeros((batch_size, seq_len, n_images), dtype=bool)
        
        for i in range(n_images):
            # For each image position, check if its ranking is less than num_to_mask
            image_rank = jnp.where(rankings == i, 
                                jnp.arange(n_images)[None, None, :], 
                                n_images)  # Set to n_images if not this image
            min_rank = jnp.min(image_rank, axis=-1)  # Get the ranking for image i
            should_mask = min_rank < num_to_mask
            mask_matrix = mask_matrix.at[:, :, i].set(should_mask)
        
        # Apply masks to each image
        for i, img_key in enumerate(available_keys):
            images = data[img_key]
            
            # Expand mask to match image dimensions
            mask = mask_matrix[:, :, i].reshape(batch_size, seq_len, 1, 1, 1)
            
            # Apply mask: where mask is True, replace with mask_value
            masked_images = jnp.where(mask, mask_value, images)
            
            # Update the data dictionary
            masked_data[img_key] = masked_images
            
        
        return masked_data

    def randomly_project_images_per_timestep(self, data, key, min_projection_ratio=0.3, max_projection_ratio=0.7):
        """
        Randomly project 0 to n-1 images for each timestep using random linear projections,
        ensuring at least one image remains unprojected.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            min_projection_ratio: Minimum ratio of original dimensions to project to (default: 0.3)
            max_projection_ratio: Maximum ratio of original dimensions to project to (default: 0.7)
        
        Returns:
            Modified data dictionary with randomly projected images
        """
        # Create a copy of the data to avoid modifying the original
        projected_data = data.copy()
        
        # Image keys to process
        image_keys = [
            "birdeye_wpt"
        ]
        
        available_keys = [k for k in image_keys if k in data]
        
        if len(available_keys) == 0:
            return projected_data
        
        n_images = len(available_keys)
        batch_size, seq_len = data[available_keys[0]].shape[0], data[available_keys[0]].shape[1]
        
        # Split keys
        key1, key2, key3, key4 = random.split(key, 4)
        
        # For each (batch, timestep), randomly choose how many images to project (0 to n-1)
        num_to_project = random.randint(key1, shape=(batch_size, seq_len), minval=0, maxval=n_images)
        
        # Generate random values for each image at each (batch, timestep)
        random_vals = random.uniform(key2, shape=(batch_size, seq_len, n_images))
        
        # Sort the random values to get rankings
        rankings = jnp.argsort(random_vals, axis=-1)
        
        # Create selection matrix for which images to project
        selection_matrix = jnp.zeros((batch_size, seq_len, n_images), dtype=bool)
        
        for i in range(n_images):
            # For each image position, check if its ranking is less than num_to_project
            image_rank = jnp.where(rankings == i, 
                                jnp.arange(n_images)[None, None, :], 
                                n_images)
            min_rank = jnp.min(image_rank, axis=-1)
            should_project = min_rank < num_to_project
            selection_matrix = selection_matrix.at[:, :, i].set(should_project)
        
        # Apply random projections to selected images
        projection_keys = random.split(key3, n_images)
        
        # Randomly sample projection ratios for each image type
        projection_ratios = random.uniform(key4, shape=(n_images,), 
                                        minval=min_projection_ratio, 
                                        maxval=max_projection_ratio)
        
        for i, img_key in enumerate(available_keys):
            images = data[img_key]  # Shape: [batch_size, seq_len, height, width, channels]
            original_shape = images.shape
            height, width, channels = original_shape[2], original_shape[3], original_shape[4]
            
            # Calculate projection dimensions using randomly sampled ratio
            original_dim = height * width * channels
            projected_dim_float = original_dim * projection_ratios[i]
            projected_dim = jnp.maximum(1, jnp.round(projected_dim_float)).astype(jnp.int32)
            
            # Use a fixed maximum size for the projection matrix, then slice
            max_projected_dim = int(original_dim * max_projection_ratio) + 1
            
            # Create random projection matrix with fixed maximum size
            proj_matrix_full = random.normal(projection_keys[i], (original_dim, max_projected_dim))
            
            # Orthogonalize using QR decomposition for stable projections
            Q, R = jnp.linalg.qr(proj_matrix_full)
            
            # Slice to actual projected dimension - we'll handle this with masking
            # Create a mask for the columns we want to use
            col_indices = jnp.arange(max_projected_dim)
            col_mask = col_indices < projected_dim
            
            # Apply column mask to effectively "slice" the matrix
            proj_matrix = Q * col_mask[None, :]  # Broadcasting the mask
            
            # Flatten images for projection
            flat_images = images.reshape(batch_size, seq_len, -1)
            
            # Apply projection and reconstruction using masked matrix
            projected = jnp.dot(flat_images, proj_matrix)  # Project to lower dim
            reconstructed = jnp.dot(projected, proj_matrix.T)  # Reconstruct back
            
            # Reshape back to original image dimensions
            reconstructed_images = reconstructed.reshape(original_shape)
            
            # Apply selection: use projected version where selected, original otherwise
            selection_mask = selection_matrix[:, :, i].reshape(batch_size, seq_len, 1, 1, 1)
            final_images = jnp.where(selection_mask, reconstructed_images, images)
            
            # Update the data dictionary
            projected_data[img_key] = final_images
        
        return projected_data
    

    def randomly_sample_projection(self, data, key, max_projection_ratio):
        """
        Apply random projection to images with 64x64 processing.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key  
            max_projection_ratio: Single projection ratio (equivalent to diffusion_coef)
        
        Returns:
            Single modified data dictionary with randomly projected images
        """
        import jax.numpy as jnp
        import jax.random as random
        from jax.scipy.ndimage import map_coordinates
        
        # Image keys to process
        image_keys = [
            "birdeye_wpt"
        ]
        
        available_keys = [k for k in image_keys if k in data]
        
        if len(available_keys) == 0:
            return data.copy()
        
        n_images = len(available_keys)
        
        # Get shape information - handle 4D case (batch, h, w, c)
        sample_image = data[available_keys[0]]
        if sample_image.ndim == 4:
            batch_size = sample_image.shape[0]
            height, width, channels = sample_image.shape[1], sample_image.shape[2], sample_image.shape[3]
        else:
            raise ValueError(f"Expected 4D array with shape (batch, h, w, c), got shape: {sample_image.shape}")
        
        print(f"Processing images with shape: {sample_image.shape}")
        print(f"Projection ratio: {max_projection_ratio}")
        
        # Split keys for different operations
        key1, key2 = random.split(key, 2)
        
        # For each batch, randomly choose how many images to project (0 to n-1)
        num_to_project = 1
        
        # Generate random values for each image at each batch
        random_vals = random.uniform(key1, shape=(batch_size, n_images))
        
        # Sort the random values to get rankings
        rankings = jnp.argsort(random_vals, axis=-1)
        
        # Create selection matrix for which images to project
        selection_matrix = jnp.zeros((batch_size, n_images), dtype=bool)
        
        for i in range(n_images):
            # For each image position, check if its ranking is less than num_to_project
            image_rank = jnp.where(rankings == i, 
                                jnp.arange(n_images)[None, :], 
                                n_images)
            min_rank = jnp.min(image_rank, axis=-1)
            should_project = min_rank < num_to_project
            selection_matrix = selection_matrix.at[:, i].set(should_project)
        
        # Create the output data dictionary
        projected_data = {}
        
        # Copy non-image data
        for k, v in data.items():
            if k not in available_keys:
                projected_data[k] = v
        
        # Apply random projections to selected images
        projection_keys = random.split(key2, n_images)
        
        for i, img_key in enumerate(available_keys):
            images = data[img_key]
            
            # Apply 64x64 random projection
            reconstructed_images = self._apply_64x64_random_projection(
                images, projection_keys[i], max_projection_ratio
            )
            
            # Apply selection: use projected version where selected, original otherwise
            selection_mask = selection_matrix[:, i].reshape(batch_size, 1, 1, 1)
            final_images = jnp.where(selection_mask, reconstructed_images, images)
            
            # Update the data dictionary
            projected_data[img_key] = final_images
        
        return projected_data


    def _apply_64x64_random_projection(self, images, projection_key, projection_ratio):
        """
        Helper method to apply 64x64 random projection to images
        
        Args:
            images: Input images (batch, h, w, c)
            projection_key: JAX random key
            projection_ratio: Ratio of dimensions to keep (0.3 = 30% compression)
        
        Returns:
            Randomly projected images with same shape as input
        """
        import jax.numpy as jnp
        import jax.random as random
        from jax.scipy.ndimage import map_coordinates
        
        batch_size, orig_h, orig_w, channels = images.shape
        
        # Calculate dimensions based on 64x64 target
        original_dim = 64 * 64 * channels
        projected_dim_float = original_dim * projection_ratio
        projected_dim = jnp.maximum(1, jnp.round(projected_dim_float)).astype(jnp.int32)
        
        # For consistency with your original code, use a max dimension
        max_projected_dim = int(original_dim * .7) + 1  # Fixed max
        
        # Step 1: Resize to 64x64
        resized_images = self._resize_to_64x64(images)
        
        # Step 2: Create random projection matrix
        proj_matrix = self._create_projection_matrix(
            projection_key, original_dim, max_projected_dim
        )
        
        # Step 3: Flatten and project
        flat_images = resized_images.reshape(batch_size, -1)
        
        # Create mask for projection dimensions
        col_mask = jnp.arange(max_projected_dim) < projected_dim
        masked_proj_matrix = proj_matrix * col_mask[None, :]
        
        # Apply projection and reconstruction
        projected = jnp.dot(flat_images, masked_proj_matrix)
        reconstructed_flat = jnp.dot(projected, masked_proj_matrix.T)
        
        # Step 4: Reshape back to 64x64
        reconstructed_64x64 = reconstructed_flat.reshape(batch_size, 64, 64, channels)
        
        # Step 5: Resize back to original size if needed
        if (orig_h, orig_w) != (64, 64):
            final_images = self._resize_from_64x64(reconstructed_64x64, (orig_h, orig_w))
        else:
            final_images = reconstructed_64x64
        
        return final_images


    def _resize_to_64x64(self, images):
        """Resize images to 64x64"""
        import jax.numpy as jnp
        from jax.scipy.ndimage import map_coordinates
        
        batch_size, orig_h, orig_w, channels = images.shape
        
        if orig_h == 64 and orig_w == 64:
            return images
        
        h_coords = jnp.linspace(0, orig_h - 1, 64)
        w_coords = jnp.linspace(0, orig_w - 1, 64)
        h_grid, w_grid = jnp.meshgrid(h_coords, w_coords, indexing='ij')
        
        resized_images = []
        for b in range(batch_size):
            batch_resized = []
            for c in range(channels):
                resized_channel = map_coordinates(
                    images[b, :, :, c],
                    [h_grid, w_grid],
                    order=1,
                    mode='nearest'
                )
                batch_resized.append(resized_channel)
            resized_images.append(jnp.stack(batch_resized, axis=-1))
        
        return jnp.stack(resized_images, axis=0)


    def _resize_from_64x64(self, images_64x64, target_size):
        """Resize from 64x64 back to target size"""
        import jax.numpy as jnp
        from jax.scipy.ndimage import map_coordinates
        
        batch_size, _, _, channels = images_64x64.shape
        target_h, target_w = target_size
        
        h_coords = jnp.linspace(0, 63, target_h)
        w_coords = jnp.linspace(0, 63, target_w)
        h_grid, w_grid = jnp.meshgrid(h_coords, w_coords, indexing='ij')
        
        resized_images = []
        for b in range(batch_size):
            batch_resized = []
            for c in range(channels):
                resized_channel = map_coordinates(
                    images_64x64[b, :, :, c],
                    [h_grid, w_grid],
                    order=1,
                    mode='nearest'
                )
                batch_resized.append(resized_channel)
            resized_images.append(jnp.stack(batch_resized, axis=-1))
        
        return jnp.stack(resized_images, axis=0)


    def _create_projection_matrix(self, key, input_dim, max_projected_dim):
        """Create orthogonal random projection matrix"""
        import jax.numpy as jnp
        import jax.random as random
        
        # Create random Gaussian matrix
        random_matrix = random.normal(key, (input_dim, max_projected_dim))
        
        # Orthogonalize using QR decomposition
        Q, R = jnp.linalg.qr(random_matrix)
        
        return Q
    
    def randomly_dropout_pixels_diffuse(self, data, key, mask_value=0.20):
        """
        Randomly dropout pixels with a random ratio between 0% to 100% for each timestep.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            mask_value: Value to use for masked pixels (default: 0.0)
        
        Returns:
            Modified data dictionary with randomly dropped out pixels
        """
        # Create a copy of the data to avoid modifying the original
        masked_data = data.copy()
        
        # Image keys to process
        image_keys = [
            "birdeye_wpt"
        ]
        available_keys = [k for k in image_keys if k in data]
        # print('Loss on Keys: ',available_keys)
        if len(available_keys) == 0:
            return masked_data
        
        batch_size = data[available_keys[0]].shape[0]
        
        # Split keys for each image
        keys = random.split(key, len(available_keys))
        
        # Apply pixel dropout to each image
        for i, img_key in enumerate(available_keys):
            images = data[img_key]  # Shape: [batch_size, H, W, C]
            
            # Get image dimensions
            H, W, C = images.shape[1], images.shape[2], images.shape[3]
            
            # Split key for dropout ratio and pixel selection
            key_ratio, key_pixels = random.split(keys[i])
            
            # Generate random dropout ratios for each (batch, timestep)
            # Shape: [batch_size, seq_len, 1, 1, 1] for broadcasting
            # dropout_ratios = random.uniform(
            #     key_ratio, 
            #     shape=(batch_size, 1, 1, 1), 
            #     minval=0.0, 
            #     maxval=mask_value
            # )
            dropout_ratios = jnp.full(
                shape=(batch_size, 1, 1, 1), 
                fill_value=mask_value
            )
            
            # Generate random values for each pixel
            # Shape: [batch_size, seq_len, H, W, C]
            pixel_random_vals = random.uniform(
                key_pixels, 
                shape=(batch_size, H, W, C)
            )
            
            # Create dropout mask: True where pixel should be dropped
            # Pixel is dropped if its random value < dropout_ratio
            dropout_mask = pixel_random_vals < dropout_ratios
            
            # Apply dropout: where mask is True, replace with mask_value
            masked_images = jnp.where(dropout_mask, mask_value, images)
            
            # Update the data dictionary
            masked_data[img_key] = masked_images
            # print(f'Pixel dropout applied to {img_key}...')
        
        return masked_data
    
    def randomly_dropout_pixels_per_timestep(self, data, key, mask_value=0.0):
        """
        Randomly dropout pixels with a random ratio between 0% to 100% for each timestep.
        
        Args:
            data: Dictionary containing the dataset with image keys
            key: JAX random key
            mask_value: Value to use for masked pixels (default: 0.0)
        
        Returns:
            Modified data dictionary with randomly dropped out pixels
        """
        # Create a copy of the data to avoid modifying the original
        masked_data = data.copy()
        
        # Image keys to process
        image_keys = [
            "birdeye_wpt"
        ]
        available_keys = [k for k in image_keys if k in data]
        # print('Loss on Keys: ',available_keys)
        if len(available_keys) == 0:
            return masked_data
        
        batch_size, seq_len = data[available_keys[0]].shape[0], data[available_keys[0]].shape[1]  # 16, 64
        
        # Split keys for each image
        keys = random.split(key, len(available_keys))
        
        # Apply pixel dropout to each image
        for i, img_key in enumerate(available_keys):
            images = data[img_key]  # Shape: [batch_size, seq_len, H, W, C]
            
            # Get image dimensions
            H, W, C = images.shape[2], images.shape[3], images.shape[4]
            
            # Split key for dropout ratio and pixel selection
            key_ratio, key_pixels = random.split(keys[i])
            
            # Generate random dropout ratios for each (batch, timestep)
            # Shape: [batch_size, seq_len, 1, 1, 1] for broadcasting
            dropout_ratios = random.uniform(
                key_ratio, 
                shape=(batch_size, seq_len, 1, 1, 1), 
                minval=0.0, 
                maxval=1.0
            )
            
            # Generate random values for each pixel
            # Shape: [batch_size, seq_len, H, W, C]
            pixel_random_vals = random.uniform(
                key_pixels, 
                shape=(batch_size, seq_len, H, W, C)
            )
            
            # Create dropout mask: True where pixel should be dropped
            # Pixel is dropped if its random value < dropout_ratio
            dropout_mask = pixel_random_vals < dropout_ratios
            
            # Apply dropout: where mask is True, replace with mask_value
            masked_images = jnp.where(dropout_mask, mask_value, images)
            
            # Update the data dictionary
            masked_data[img_key] = masked_images
            # print(f'Pixel dropout applied to {img_key}...')
        
        return masked_data
    


    
    # def randomly_sample_projections(self, data, key, N_samples, min_projection_ratio=0.3, max_projection_ratio=0.7):
    #     """
    #     Randomly project 0 to n-1 images for each timestep using random linear projections,
    #     ensuring at least one image remains unprojected.
        
    #     Args:
    #         data: Dictionary containing the dataset with image keys
    #         key: JAX random key
    #         N_samples: Number of different projection ratios to sample
    #         min_projection_ratio: Minimum ratio of original dimensions to project to (default: 0.3)
    #         max_projection_ratio: Maximum ratio of original dimensions to project to (default: 0.7)
        
    #     Returns:
    #         List of modified data dictionaries with randomly projected images
    #     """
    #     import jax.numpy as jnp
    #     import jax.random as random
    #     import numpy as np
        
    #     # Image keys to process
    #     image_keys = [
    #         "birdeye_wpt"
    #     ]
        
    #     available_keys = [k for k in image_keys if k in data]
        
    #     if len(available_keys) == 0:
    #         return [data.copy()]
        
    #     n_images = len(available_keys)
        
    #     # Get shape information - handle 4D case (batch, h, w, c)
    #     sample_image = data[available_keys[0]]
    #     if sample_image.ndim == 4:
    #         # Format: (batch, h, w, c) - treat as single timestep
    #         batch_size, seq_len = sample_image.shape[0], 1
    #         height, width, channels = sample_image.shape[1], sample_image.shape[2], sample_image.shape[3]
    #     else:
    #         raise ValueError(f"Expected 4D array with shape (batch, h, w, c), got shape: {sample_image.shape}")
        
    #     print(f"Processing images with shape: {sample_image.shape}")
        
    #     # Split keys
    #     key1, key2, key3 = random.split(key, 3)
        
    #     # For each (batch, timestep), randomly choose how many images to project (0 to n-1)
    #     num_to_project = 1
        
    #     # Generate random values for each image at each (batch, timestep)
    #     random_vals = random.uniform(key2, shape=(batch_size, n_images))
        
    #     # Sort the random values to get rankings
    #     rankings = jnp.argsort(random_vals, axis=-1)
        
    #     projections = []
        
    #     for diffusion_coef in np.linspace(min_projection_ratio, max_projection_ratio, N_samples):
    #         projected_data = {}
            
    #         # Copy non-image data
    #         for k, v in data.items():
    #             if k not in available_keys:
    #                 projected_data[k] = v
            
    #         # Create selection matrix for which images to project
    #         selection_matrix = jnp.zeros((batch_size, n_images), dtype=bool)
            
    #         for i in range(n_images):
    #             # For each image position, check if its ranking is less than num_to_project
    #             image_rank = jnp.where(rankings == i, 
    #                                 jnp.arange(n_images)[None, :], 
    #                                 n_images)
    #             min_rank = jnp.min(image_rank, axis=-1)
    #             should_project = min_rank < num_to_project
    #             selection_matrix = selection_matrix.at[:, i].set(should_project)
            
    #         # Apply random projections to selected images
    #         projection_keys = random.split(key3, n_images)
            
    #         for i, img_key in enumerate(available_keys):
    #             images = data[img_key]
    #             original_shape = images.shape
                
    #             # Handle 4D input shape - reshape to (batch, seq, h, w, c) for consistent processing
    #             if images.ndim == 4:
    #                 # Reshape (batch, h, w, c) to (batch, 1, h, w, c)
    #                 images = images[:, None, ...]
    #                 process_shape = images.shape
    #             else:
    #                 process_shape = original_shape
                
    #             # Calculate projection dimensions using randomly sampled ratio
    #             original_dim = height * width * channels
    #             projected_dim_float = original_dim * diffusion_coef
    #             projected_dim = jnp.maximum(1, jnp.round(projected_dim_float)).astype(jnp.int32)
                
    #             # Use a fixed maximum size for the projection matrix, then slice
    #             max_projected_dim = int(original_dim * max_projection_ratio) + 1
                
    #             # Create random projection matrix with fixed maximum size
    #             proj_matrix_full = random.normal(projection_keys[i], (original_dim, max_projected_dim))
                
    #             # Orthogonalize using QR decomposition for stable projections
    #             Q, R = jnp.linalg.qr(proj_matrix_full)
                
    #             # Create a mask for the columns we want to use
    #             col_indices = jnp.arange(max_projected_dim)
    #             col_mask = col_indices < projected_dim
                
    #             # Apply column mask to effectively "slice" the matrix
    #             proj_matrix = Q * col_mask[None, :]
                
    #             # Flatten images for projection
    #             flat_images = images.reshape(batch_size, seq_len, -1)
                
    #             # Apply projection and reconstruction using masked matrix
    #             projected = jnp.dot(flat_images, proj_matrix)  # Project to lower dim
    #             reconstructed = jnp.dot(projected, proj_matrix.T)  # Reconstruct back
                
    #             # Reshape back to processing dimensions
    #             reconstructed_images = reconstructed.reshape(process_shape)
                
    #             # Apply selection: use projected version where selected, original otherwise
    #             # For multiple batch/timesteps, use broadcasting
    #             selection_mask = selection_matrix[:, i].reshape(batch_size, 1, 1, 1, 1)
    #             final_images = jnp.where(selection_mask, reconstructed_images, images)
                
    #             # Reshape back to original shape if needed
    #             if original_shape != final_images.shape:
    #                 final_images = final_images.reshape(original_shape)
                
    #             # Update the data dictionary
    #             projected_data[img_key] = final_images
            
    #         projections.append(projected_data)
        
    #     return projections

    def loss(self, data, state):
        key = random.PRNGKey(42)
        # data = self.randomly_mask_images_per_timestep(data, key, mask_value=0.0)
        enc_data = self.randomly_dropout_pixels_per_timestep(data, key)
        embed = self.encoder(enc_data)
        prev_latent, prev_action = state
        prev_actions = jnp.concatenate([prev_action[:, None], data["action"][:, :-1]], 1)
        post, prior = self.rssm.observe(embed, prev_actions, data["is_first"], prev_latent)
        dists = {}
        feats = {**post, "embed": embed}
        for name, head in self.heads.items():
            out = head(feats if name in self.config.grad_heads else sg(feats))
            out = out if isinstance(out, dict) else {name: out}
            dists.update(out)
        losses = {}
        losses["dyn"] = self.rssm.dyn_loss(post, prior, **self.config.dyn_loss)
        losses["rep"] = self.rssm.rep_loss(post, prior, **self.config.rep_loss)
        for key, dist in dists.items():
            loss = -dist.log_prob(data[key].astype(jnp.float32))
            assert loss.shape == embed.shape[:2], (key, loss.shape)
            losses[key] = loss
        scaled = {k: v * self.scales[k] for k, v in losses.items()}
        model_loss = sum(scaled.values())
        out = {"embed": embed, "post": post, "prior": prior}
        out.update({f"{k}_loss": v for k, v in losses.items()})
        last_latent = {k: v[:, -1] for k, v in post.items()}
        last_action = data["action"][:, -1]
        state = last_latent, last_action
        metrics = self._metrics(data, dists, post, prior, losses, model_loss)
        metrics["model_loss_raw"] = model_loss  # Store model loss for Curious Replay prioritization
        return model_loss.mean(), (state, out, metrics)

    def imagine(self, policy, start, horizon):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        start["action"] = policy(start)

        def step(prev, _):
            prev = prev.copy()
            state = self.rssm.img_step(prev, prev.pop("action"))
            return {**state, "action": policy(state)}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items()}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def imagine_carry(self, policy, start, horizon, carry):
        first_cont = (1.0 - start["is_terminal"]).astype(jnp.float32)
        keys = list(self.rssm.initial(1).keys())
        start = {k: v for k, v in start.items() if k in keys}
        outs, carry = policy(start, carry)
        start["action"] = outs
        start["carry"] = carry

        def step(prev, _):
            prev = prev.copy()
            carry = prev.pop("carry")
            state = self.rssm.img_step(prev, prev.pop("action"))
            outs, carry = policy(state, carry)
            return {**state, "action": outs, "carry": carry}

        traj = jaxutils.scan(step, jnp.arange(horizon), start, self.config.imag_unroll)
        traj = {k: jnp.concatenate([start[k][None], v], 0) for k, v in traj.items() if k != "carry"}
        cont = self.heads["cont"](traj).mode()
        traj["cont"] = jnp.concatenate([first_cont[None], cont[1:]], 0)
        discount = 1 - 1 / self.config.horizon
        traj["weight"] = jnp.cumprod(discount * traj["cont"], 0) / discount
        return traj

    def report(self, data):
        state = self.initial(len(data["is_first"]))
        report = {}
        report.update(self.loss(data, state)[-1][-1])
        context, _ = self.rssm.observe(self.encoder(data)[:6, :5], data["action"][:6, :5], data["is_first"][:6, :5])
        start = {k: v[:, -1] for k, v in context.items()}
        recon = self.heads["decoder"](context)
        openl = self.heads["decoder"](self.rssm.iagine(data["action"][:6, 5:], start))
        for key in self.heads["decoder"].cnn_shapes.keys():
            truth = data[key][:6].astype(jnp.float32)
            model = jnp.concatenate([recon[key].mode()[:, :5], openl[key].mode()], 1)
            error = (model - truth + 1) / 2
            video = jnp.concatenate([truth, model, error], 2)
            report[f"openl_{key}"] = jaxutils.video_grid(video)
        return report

    def _metrics(self, data, dists, post, prior, losses, model_loss):
        entropy = lambda feat: self.rssm.get_dist(feat).entropy()
        metrics = {}
        metrics.update(jaxutils.tensorstats(entropy(prior), "prior_ent"))
        metrics.update(jaxutils.tensorstats(entropy(post), "post_ent"))
        metrics.update({f"{k}_loss_mean": v.mean() for k, v in losses.items()})
        metrics.update({f"{k}_loss_std": v.std() for k, v in losses.items()})
        metrics["model_loss_mean"] = model_loss.mean()
        metrics["model_loss_std"] = model_loss.std()
        metrics["reward_max_data"] = jnp.abs(data["reward"]).max()
        metrics["reward_max_pred"] = jnp.abs(dists["reward"].mean()).max()
        if "reward" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["reward"], data["reward"], 0.1)
            metrics.update({f"reward_{k}": v for k, v in stats.items()})
        if "cont" in dists and not self.config.jax.debug_nans:
            stats = jaxutils.balance_stats(dists["cont"], data["cont"], 0.5)
            metrics.update({f"cont_{k}": v for k, v in stats.items()})
        return metrics


class ImagActorCritic(nj.Module):
    def __init__(self, critics, scales, act_space, config):
        critics = {k: v for k, v in critics.items() if scales[k]}
        for key, scale in scales.items():
            assert not scale or key in critics, key
        self.critics = {k: v for k, v in critics.items() if scales[k]}
        self.scales = scales
        self.act_space = act_space
        self.config = config
        disc = act_space.discrete
        self.grad = config.actor_grad_disc if disc else config.actor_grad_cont
        self.actor = nets.MLP(
            name="actor",
            dims="deter",
            shape=act_space.shape,
            **config.actor,
            dist=config.actor_dist_disc if disc else config.actor_dist_cont,
        )
        self.retnorms = {k: jaxutils.Moments(**config.retnorm, name=f"retnorm_{k}") for k in critics}
        self.opt = jaxutils.Optimizer(name="actor_opt", **config.actor_opt)

    def initial(self, batch_size):
        return {}

    def policy(self, state, carry):
        return {"action": self.actor(state)}, carry

    def train(self, imagine, start, context):
        def loss(start):
            policy = lambda s: self.actor(sg(s)).sample(seed=nj.rng())
            traj = imagine(policy, start, self.config.imag_horizon)
            loss, metrics = self.loss(traj)
            return loss, (traj, metrics)

        mets, (traj, metrics) = self.opt(self.actor, loss, start, has_aux=True)
        metrics.update(mets)
        for key, critic in self.critics.items():
            mets = critic.train(traj, self.actor)
            metrics.update({f"{key}_critic_{k}": v for k, v in mets.items()})
        return traj, metrics

    def loss(self, traj):
        metrics = {}
        advs = []
        total = sum(self.scales[k] for k in self.critics)
        for key, critic in self.critics.items():
            rew, ret, base = critic.score(traj, self.actor)
            offset, invscale = self.retnorms[key](ret)
            normed_ret = (ret - offset) / invscale
            normed_base = (base - offset) / invscale
            advs.append((normed_ret - normed_base) * self.scales[key] / total)
            metrics.update(jaxutils.tensorstats(rew, f"{key}_reward"))
            metrics.update(jaxutils.tensorstats(ret, f"{key}_return_raw"))
            metrics.update(jaxutils.tensorstats(normed_ret, f"{key}_return_normed"))
            metrics[f"{key}_return_rate"] = (jnp.abs(ret) >= 0.5).mean()

        # if len(self.critics) != 1:
        #  raise NotImplementedError('Must have exactly one critic for TD error calculation.')

        r = jnp.reshape(rew[0], (self.config.batch_size, self.config.batch_length))
        v = jnp.reshape(base[0], (self.config.batch_size, self.config.batch_length))
        disc = jnp.reshape(traj["cont"][0], (self.config.batch_size, self.config.batch_length)) * (1 - 1 / self.config.horizon)
        td_error = r[:, :-1] + disc[:, 1:] * v[:, 1:] - v[:, :-1]
        metrics["td_error"] = td_error  # Store TD error for PER prioritization

        adv = jnp.stack(advs).sum(0)
        policy = self.actor(sg(traj))
        logpi = policy.log_prob(sg(traj["action"]))[:-1]
        loss = {"backprop": -adv, "reinforce": -logpi * sg(adv)}[self.grad]
        ent = policy.entropy()[:-1]
        loss -= self.config.actent * ent
        loss *= sg(traj["weight"])[:-1]
        loss *= self.config.loss_scales.actor
        metrics.update(self._metrics(traj, policy, logpi, ent, adv))
        return loss.mean(), metrics

    def _metrics(self, traj, policy, logpi, ent, adv):
        metrics = {}
        ent = policy.entropy()[:-1]
        rand = (ent - policy.minent) / (policy.maxent - policy.minent)
        rand = rand.mean(range(2, len(rand.shape)))
        act = traj["action"]
        act = jnp.argmax(act, -1) if self.act_space.discrete else act
        metrics.update(jaxutils.tensorstats(act, "action"))
        metrics.update(jaxutils.tensorstats(rand, "policy_randomness"))
        metrics.update(jaxutils.tensorstats(ent, "policy_entropy"))
        metrics.update(jaxutils.tensorstats(logpi, "policy_logprob"))
        metrics.update(jaxutils.tensorstats(adv, "adv"))
        metrics["imag_weight_dist"] = jaxutils.subsample(traj["weight"])
        return metrics


class VFunction(nj.Module):
    def __init__(self, rewfn, config):
        self.rewfn = rewfn
        self.config = config
        self.net = nets.MLP((), name="net", dims="deter", **self.config.critic)
        self.slow = nets.MLP((), name="slow", dims="deter", **self.config.critic)
        self.updater = jaxutils.SlowUpdater(
            self.net,
            self.slow,
            self.config.slow_critic_fraction,
            self.config.slow_critic_update,
        )
        self.opt = jaxutils.Optimizer(name="critic_opt", **self.config.critic_opt)

    def train(self, traj, actor):
        target = sg(self.score(traj)[1])
        mets, metrics = self.opt(self.net, self.loss, traj, target, has_aux=True)
        metrics.update(mets)
        self.updater()
        return metrics

    def loss(self, traj, target):
        metrics = {}
        traj = {k: v[:-1] for k, v in traj.items()}
        dist = self.net(traj)
        loss = -dist.log_prob(sg(target))
        if self.config.critic_slowreg == "logprob":
            reg = -dist.log_prob(sg(self.slow(traj).mean()))
        elif self.config.critic_slowreg == "xent":
            reg = -jnp.einsum("...i,...i->...", sg(self.slow(traj).probs), jnp.log(dist.probs))
        else:
            raise NotImplementedError(self.config.critic_slowreg)
        loss += self.config.loss_scales.slowreg * reg
        loss = (loss * sg(traj["weight"])).mean()
        loss *= self.config.loss_scales.critic
        metrics = jaxutils.tensorstats(dist.mean())
        return loss, metrics

    def score(self, traj, actor=None):
        rew = self.rewfn(traj)
        assert len(rew) == len(traj["action"]) - 1, "should provide rewards for all but last action"
        discount = 1 - 1 / self.config.horizon
        disc = traj["cont"][1:] * discount
        value = self.net(traj).mean()
        vals = [value[-1]]
        interm = rew + disc * value[1:] * (1 - self.config.return_lambda)
        for t in reversed(range(len(disc))):
            vals.append(interm[t] + disc[t] * self.config.return_lambda * vals[-1])
        ret = jnp.stack(list(reversed(vals))[:-1])
        return rew, ret, value[:-1]
