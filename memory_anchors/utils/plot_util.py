import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.animation as animation
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cv2
import os

default_action_dim_names_rot6d = ['rel pos x', 'rel pos y', 'rel pos z', 'rel rot6d 1', 'rel rot6d 2', 'rel rot6d 3', 'rel rot6d 4', 'rel rot6d 5', 'rel rot6d 6', 'gripper action']

def vis_training_batch_action_preds(prompt, action_preds, output_path, action_dim_names=default_action_dim_names_rot6d):
    """
    Visualize the action predictions for a given prompt. Assumes that the action preds are for the entire sequence.

    Args:
        prompt (dict): The prompt to visualize. (tensor dict)
        action_preds (numpy.ndarray): The action predictions to visualize. (T, num_pred_steps, action_dim) (numpy dict)
    """

    prompt_actions = prompt['action'].numpy() # (T, chunk_dim, num_pred_steps, action_dim)
    T, chunk_dim, num_pred_steps, action_dim = prompt_actions.shape
    action_preds = action_preds.numpy() # (T, num_pred_steps, action_dim)

    prompt_actions = prompt_actions[:, :, 0, :] # (T, chunk_dim, action_dim)
    prompt_actions = prompt_actions.reshape(T*chunk_dim, action_dim)
    action_preds = action_preds[:, :chunk_dim, :] # (T, num_pred_steps, action_dim)-> (T, chunk_dim, action_dim)
    action_preds = action_preds.reshape(T*chunk_dim, action_dim)

    mask_location = np.where(prompt['metadata']['prompt_mask'].numpy())[0][0] * chunk_dim # this is a time index where the prompt ends

    eos = prompt['metadata']['eos'].numpy()
    end_segment_locations = list((np.where(eos)[0] + 1) * chunk_dim)
        
    # Create figure with subplots for each action dimension
    action_dim = prompt_actions.shape[1]
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 3*action_dim))
    time_steps = np.arange(len(prompt_actions))

    # Create handles and labels for the legend
    legend_handles = []
    legend_labels = []

    # Plot each action dimension
    for i in range(action_dim):
        if action_dim == 1:
            ax = axes
        else:
            ax = axes[i]
            
        # Plot lines and store handles for legend
        gt_line = ax.plot(time_steps, prompt_actions[:, i], label='Ground Truth', alpha=0.7)[0]
        pred_line = ax.plot(time_steps, action_preds[:, i], label='Predicted', alpha=0.7)[0]
        
        # Only add handles and labels once
        if i == 0:
            legend_handles.extend([gt_line, pred_line])
            legend_labels.extend(['Ground Truth', 'Predicted'])
            
        ax.set_xlabel('Time Step')
        ax.set_ylabel(f'{action_dim_names[i]}')
        ax.grid(True)

        # Add background shading to indicate prompt region
        # Shade prompt region (0 to mask_location) in light blue
        prompt_patch = ax.axvspan(0, mask_location-1, color='lightblue', alpha=0.3, label='Prompt Region')
        # Shade prediction region (after mask_location) in light gray 
        pred_patch = ax.axvspan(mask_location, len(prompt_actions), color='lightgray', alpha=0.3, label='Rollout Region')

        # Add vertical lines to indicate end of segments
        for end_loc in end_segment_locations:
            ax.axvline(x=end_loc, color='black', linestyle='--', alpha=0.5)
        
        # Only add patch handles once
        if i == 0:
            legend_handles.extend([prompt_patch, pred_patch])
            legend_labels.extend(['Prompt Region', 'Prediction Region'])

    # Add a single legend at the bottom of the figure
    fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    # Adjust layout to make room for the legend
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(output_path)
    plt.close()

    print(f'Saved action predictions figure to {output_path}')

def vis_actions_rollout(output_path, actions, prompt=None, action_dim_names=default_action_dim_names_rot6d):
    """
    Visualize the actions for a given rollout (potentially with a prompt before the rollout).

    Args:
        actions (numpy.ndarray): The actions to visualize. (T, action_dim) (numpy dict)
        prompt (dict) [optional]: The prompt to visualize. (tensor dict). Prompt should not have a batch dimension.
    """

    if prompt is not None:
        prompt_actions = prompt['action'].cpu().numpy() # (T_prompt, chunk dim, num_pred_steps, action_dim)

        if len(prompt_actions.shape) == 4: # remove the num_pred_steps dimension if present
            prompt_actions = prompt_actions[:, :, 0] # (T_prompt, chunk dim, num_pred_steps, action_dim) -> (T_prompt, chunk_dim, action_dim)

        T_prompt, chunk_dim, action_dim = prompt_actions.shape
        prompt_actions = prompt_actions.reshape(T_prompt*chunk_dim, action_dim)
    else:
        action_dim = actions.shape[1]

    # Create figure with subplots for each action dimension
    fig, axes = plt.subplots(action_dim, 1, figsize=(10, 3*action_dim))

    # Create handles and labels for the legend
    legend_handles = []
    legend_labels = []

    # Plot each action dimension
    for i in range(action_dim):
        ax = axes[i]
        
        if prompt is not None:
            # Plot prompt actions starting at 0
            prompt_line = ax.plot(np.arange(len(prompt_actions)), prompt_actions[:, i], label='Prompt Actions')[0]
            
            # Plot rollout actions continuing after prompt
            rollout_line = ax.plot(np.arange(len(prompt_actions), len(prompt_actions) + len(actions)), 
                                 actions[:, i], label='Rollout Actions')[0]
            
            # Add background shading to indicate prompt region
            prompt_patch = ax.axvspan(0, len(prompt_actions), color='lightblue', alpha=0.3, label='Prompt Region')
            rollout_patch = ax.axvspan(len(prompt_actions), len(prompt_actions) + len(actions), 
                                     color='lightgray', alpha=0.3, label='Rollout Region')
            
            ax.axvline(x=len(prompt_actions), color='black', linestyle='--', alpha=0.5)
        else:
            # If no prompt, just plot actions starting at 0
            rollout_line = ax.plot(np.arange(len(actions)), actions[:, i], label='Actions')[0]
        
        ax.set_ylabel(f'{action_dim_names[i]}')
        ax.set_xlabel('Time Step')
        ax.grid(True)

        # Only add handles once
        if i == 0:
            if prompt is not None:
                legend_handles.extend([prompt_line, rollout_line, prompt_patch, rollout_patch])
                legend_labels.extend(['Prompt Actions', 'Rollout Actions', 'Prompt Region', 'Rollout Region'])
            else:
                legend_handles.append(rollout_line)
                legend_labels.append('Actions')

    # Add a single legend at the bottom of the figure
    fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(output_path)
    plt.close()
    
    print(f'Saved action rollout figure to {output_path}')

def vis_proprio_rollout(output_path, proprio_list, proprio_key_names, proprio_key_sizes, prompt=None, proprio_legend_names=None):
    """
    Visualize the proprio for a given rollout (potentially with a prompt before the rollout).

    Args:
        proprio_list (list of numpy.ndarray): List of proprio arrays to visualize. Each array should be (T, proprio_dim)
        proprio_key_names (list): The names of the proprio keys.
        proprio_key_sizes (list): The sizes of the proprio keys.
        prompt (dict) [optional]: The prompt to visualize. (tensor dict). Prompt should not have a batch dimension.
        proprio_legend_names (list) [optional]: List of names to use in the legend for each proprio array. If None, will use default names "Proprio 1", "Proprio 2", etc.
    """

    # Process prompt data if available
    prompt_proprio = None
    if prompt is not None:
        remove_pred_steps = lambda x: x[:, 0] if len(x.shape) == 3 else x
        prompt_proprio = np.concatenate([remove_pred_steps(prompt['obs'][key].cpu().numpy()) for key in proprio_key_names], axis=-1)
        downsampled_prompt_len = prompt_proprio.shape[0]
        chunk_n_actions = prompt['action'].shape[1]
        prompt_proprio = prompt_proprio.repeat(chunk_n_actions, axis=0)  # repeat the prompt actions because they are downsampled

        if 'eos' in prompt['metadata']:
            eos = prompt['metadata']['eos'].cpu().numpy()
            end_segment_locations = list(np.where(eos)[0])
        else:
            end_segment_locations = [downsampled_prompt_len - 1]

    # Create figure with subplots for each proprio dimension
    proprio_dim = proprio_list[0].shape[1]
    fig, axes = plt.subplots(proprio_dim, 1, figsize=(10, 3*proprio_dim))

    # Create handles and labels for the legend
    legend_handles = []
    legend_labels = []

    proprio_dim_names = []
    for key, size in zip(proprio_key_names, proprio_key_sizes):
        for i in range(size):
            proprio_dim_names.append(f'{key} {i}')

    # Plot each proprio dimension
    for i in range(proprio_dim):
        ax = axes[i]
        
        if prompt_proprio is not None:
            # Plot prompt data
            prompt_line = ax.plot(np.arange(len(prompt_proprio)), 
                                prompt_proprio[:, i], 
                                label='Prompt', alpha=0.7)[0]
            
            # Plot rollout data for each trajectory
            for j, proprio in enumerate(proprio_list):
                rollout_line = ax.plot(np.arange(len(prompt_proprio), len(prompt_proprio) + len(proprio)), 
                                     proprio[:, i],
                                     label=proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}',
                                     alpha=0.7)[0]
                
                if i == 0:  # Only add to legend once
                    legend_handles.append(rollout_line)
                    legend_labels.append(proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}')
            
            # Add background shading
            prompt_patch = ax.axvspan(0, len(prompt_proprio), color='lightblue', alpha=0.3, label='Prompt Region')
            rollout_patch = ax.axvspan(len(prompt_proprio), len(prompt_proprio) + len(proprio_list[0]), 
                                     color='lightgray', alpha=0.3, label='Rollout Region')
            
            # Add vertical lines for segment ends
            for end_loc in end_segment_locations:
                ax.axvline(x=(end_loc+1)*chunk_n_actions, color='black', linestyle='--', alpha=0.5)
                
            if i == 0:  # Only add to legend once
                legend_handles.extend([prompt_line, prompt_patch, rollout_patch])
                legend_labels.extend(['Prompt', 'Prompt Region', 'Rollout Region'])
        else:
            # If no prompt, just plot rollout data
            for j, proprio in enumerate(proprio_list):
                rollout_line = ax.plot(np.arange(len(proprio)), proprio[:, i],
                                     label=proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}')[0]
                if i == 0:  # Only add to legend once
                    legend_handles.append(rollout_line)
                    legend_labels.append(proprio_legend_names[j] if proprio_legend_names is not None else f'Rollout {j+1}')

        ax.set_ylabel(f'{proprio_dim_names[i]}')
        ax.set_xlabel('Time Step')
        ax.grid(True)

    # Add a single legend at the bottom of the figure
    if len(legend_handles) > 0:
        fig.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05)
    plt.savefig(output_path)
    plt.close()
    
    print(f'Saved proprio rollout figure to {output_path}')

def vis_attention_map(output_path, attention_map, exec_action_horizon, cross_attn_dim_names, fps=20, max_frames=None):
    """
    Visualize the attention map as a video.

    Args:
        output_path (str): The path to save the attention map video. Should end with .mp4 or .gif.
        attention_map (numpy.ndarray): The attention map to visualize. (num_inference_steps, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim)
        exec_action_horizon (int): Number of action steps per inference step.
        cross_attn_dim_names (list): The names of the cross-attention dimensions.
        fps (int): Frames per second for the output video.
        max_frames (int, optional): Maximum number of frames to render. If None, use all frames.
    """
    import matplotlib.animation as animation
    from tqdm import tqdm

    attention_map = attention_map[..., :len(cross_attn_dim_names)]

    num_inference_steps, num_diffusion_steps, num_layers, action_pred_steps, cross_attn_dim = attention_map.shape

    # Select the attention map at the middle of the diffusion steps
    mid_diff_step = num_diffusion_steps // 2
    attention_map = attention_map[:, mid_diff_step]  # (num_inference_steps, num_layers, action_pred_steps, cross_attn_dim)
    attention_map = attention_map[:, :, :exec_action_horizon] # (num_inference_steps, num_layers, action_pred_steps, cross_attn_dim)
    attention_map = np.transpose(attention_map, (1, 0, 2, 3)) # (num_layers, num_inference_steps, exec_action_horizon, cross_attn_dim)
    attention_map = attention_map.reshape(num_layers, num_inference_steps*exec_action_horizon, cross_attn_dim) # (num_layers, num_inference_steps*exec_action_horizon, cross_attn_dim)
    attention_map = np.transpose(attention_map, (1, 0, 2)) # (num_inference_steps*exec_action_horizon, num_layers, cross_attn_dim)

    num_exec_steps = num_inference_steps * exec_action_horizon
    if max_frames is not None:
        num_frames = min(num_exec_steps, max_frames)
    else:
        num_frames = num_exec_steps
    
    # Reshape to have layers on y-axis and cross_attn_dim on x-axis
    attention_map = attention_map.reshape(num_exec_steps, num_layers, cross_attn_dim)

    fig, ax = plt.subplots(figsize=(len(cross_attn_dim_names)//3, 10))
    vmin, vmax = attention_map.min(), attention_map.max()
    # Draw the first frame and colorbar
    heatmap = sns.heatmap(attention_map[0], ax=ax, cmap='viridis', cbar=False, vmin=vmin, vmax=vmax, xticklabels=cross_attn_dim_names)
    ax.set_aspect(1.0)
    # Use make_axes_locatable to match colorbar height to axes
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = heatmap.get_children()[0]
    plt.colorbar(im, cax=cax)
    ax.set_xlabel('Encoder Tokens', fontsize=22)
    ax.set_ylabel('Decoder Layers', fontsize=22)
    ax.set_title(f'Attention Map - Step 1/{num_frames}', pad=10, fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=18)
    fig.subplots_adjust(top=0.90)
    plt.tight_layout()

    def update(step):
        ax.set_title(f'Attention Map - Step {step+1}/{num_frames}', fontsize=28)
        # Remove the old heatmap but keep the colorbar
        for artist in ax.get_children():
            if isinstance(artist, (plt.matplotlib.collections.QuadMesh,)):
                artist.remove()
        sns.heatmap(attention_map[step], ax=ax, cmap='viridis', cbar=False, vmin=vmin, vmax=vmax, xticklabels=cross_attn_dim_names)
        ax.set_aspect(1.0)
        ax.set_xlabel('Encoder Tokens', fontsize=22)
        ax.set_ylabel('Decoder Layers', fontsize=22)
        ax.tick_params(axis='both', which='major', labelsize=18)
        plt.tight_layout()

    # Use tqdm for progress bar during animation saving
    def frame_generator():
        for i in tqdm(range(num_frames), desc='Rendering frames'):
            yield i

    ani = animation.FuncAnimation(fig, update, frames=frame_generator, repeat=False)

    # Save as video (mp4)
    if output_path.endswith('.mp4'):
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(output_path, writer=writer)
    elif output_path.endswith('.gif'):
        writer = animation.PillowWriter(fps=fps)
        ani.save(output_path, writer=writer)
    else:
        raise ValueError('Output path must end with .mp4 or .gif')
    plt.close()
    print(f'Saved attention map video to {output_path}')

def combine_attention_and_rollout_video(attention_map_path, rollout_video_path, output_path, prompt, prompt_img_keys, steps_per_render=2):
    """
    Combine the attention map video and the rollout video into a single video, with the prompt observation image below each frame.
    """
    # Read videos
    attention_cap = cv2.VideoCapture(attention_map_path)
    rollout_cap = cv2.VideoCapture(rollout_video_path)

    # Get video properties
    attn_width = int(attention_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    attn_height = int(attention_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    attn_fps = attention_cap.get(cv2.CAP_PROP_FPS)
    attn_frames = int(attention_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    roll_width = int(rollout_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    roll_height = int(rollout_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roll_frames = int(rollout_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate resized rollout width to match attention frame height
    scale = attn_height / roll_height
    resized_roll_width = int(roll_width * scale)

    # --- Generate prompt observation image ---
    prompt_obs_path = 'prompt_obs_temp.png'
    plot_prompt_obs(prompt, prompt_obs_path, prompt_img_keys)
    prompt_obs_img = cv2.imread(prompt_obs_path)
    if prompt_obs_img is None:
        raise RuntimeError(f"Failed to load prompt observation image from {prompt_obs_path}")
    # Resize prompt_obs_img to match the combined width of the video frames, scaling height proportionally
    combined_width = attn_width + resized_roll_width
    orig_height, orig_width = prompt_obs_img.shape[:2]
    scale_factor = combined_width / orig_width
    new_height = int(orig_height * scale_factor)
    prompt_obs_img = cv2.resize(prompt_obs_img, (combined_width, new_height))
    prompt_obs_height = prompt_obs_img.shape[0]

    # Create video writer with correct dimensions (width, height)
    out_height = attn_height + prompt_obs_height
    fourcc = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
    out = cv2.VideoWriter(output_path, fourcc, attn_fps, (combined_width, out_height))

    # Read and combine frames
    roll_frame_idx = 0
    for attn_frame_idx in tqdm(range(attn_frames), desc='Combining attention, rollout, and prompt obs'):
        ret_attn, attn_frame = attention_cap.read()
        if not ret_attn:
            break

        # Only read new rollout frame every steps_per_render frames
        if attn_frame_idx % steps_per_render == 0:
            ret_roll, roll_frame = rollout_cap.read()
            if not ret_roll:
                break
            roll_frame_idx += 1
            current_roll_frame = roll_frame
        # Resize rollout frame to match attention frame height and precomputed width
        resized_roll = cv2.resize(current_roll_frame, (resized_roll_width, attn_height))
        # Stack frames horizontally
        combined_top = np.hstack((attn_frame, resized_roll))
        # Stack prompt observation image below
        combined_full = np.vstack((combined_top, prompt_obs_img))
        out.write(combined_full)

    # Release everything
    attention_cap.release()
    rollout_cap.release()
    out.release()
    # Remove the temporary prompt observation image
    if os.path.exists(prompt_obs_path):
        os.remove(prompt_obs_path)
    print(f'Saved combined video to {output_path}')

def plot_prompt_obs(prompt, output_path, prompt_img_keys):
    """
    Plot the prompt observations by vertically stacking images from different keys for each timestep, then concatenate all timesteps horizontally with no whitespace. Adds a title above each image.
    Args:
        prompt (dict): Dictionary containing image tensors for each key in prompt_img_keys. Each tensor is (T, 3, H, W) and dtype float32.
        output_path (str): Path to save the resulting plot.
        prompt_img_keys (list): List of keys to extract images from the prompt.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Get the number of timesteps (T) from the first key
    T = prompt['obs'][prompt_img_keys[0]].shape[0]

    # For each timestep, vertically stack the images from all keys
    stacked_imgs = []
    widths = []
    for t in range(T):
        imgs_at_t = [np.transpose(prompt['obs'][key][t], (1, 2, 0)) for key in prompt_img_keys]  # (3, H, W) -> (H, W, 3)
        stacked_img = np.vstack(imgs_at_t)  # (sum_H, W, 3)
        # Handle float32 images: scale to [0, 255] and convert to uint8
        if stacked_img.dtype == np.float32 or stacked_img.dtype == np.float64:
            if stacked_img.max() <= 1.0:
                stacked_img = np.clip(stacked_img, 0, 1)
                stacked_img = (stacked_img * 255).round().astype(np.uint8)
            else:
                stacked_img = np.clip(stacked_img, 0, 255).astype(np.uint8)
        stacked_imgs.append(stacked_img)
        widths.append(stacked_img.shape[1])

    # Concatenate all stacked images horizontally (side by side)
    concat_img = np.hstack(stacked_imgs)  # (sum_H, T*W, 3)
    total_width = concat_img.shape[1]
    height = concat_img.shape[0]

    # Plot the single concatenated image
    fig, ax = plt.subplots(figsize=(total_width / 100, height / 100 + 0.5), dpi=100)
    ax.imshow(concat_img)
    ax.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.margins(0, 0)
    plt.tight_layout(pad=0)

    # Add a title above each image
    x = 0
    for t, w in enumerate(widths):
        ax.text(x + w / 2, -10, f'Prompt Obs {t}', va='bottom', ha='center', fontsize=12, color='black', fontweight='bold', transform=ax.transData, clip_on=False)
        x += w

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f'Saved prompt observation plot to {output_path}')
    