import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.ndimage import uniform_filter
from sklearn.metrics import mean_absolute_error 
# No longer importing properscoring
# import properscoring as ps 

# --- Data Transformation ---
def denormalize_data(normalized_data, clip_value=100):
    """
    Denormalizes log-transformed data back to the original physical scale.
    """
    log_of_clip_plus_one = np.log(clip_value + 1)
    scaled_data = normalized_data * log_of_clip_plus_one
    denormalized_data = np.expm1(scaled_data) # Equivalent to np.exp(scaled_data) - 1
    return np.clip(denormalized_data, 0, clip_value)

# --- Metric Functions ---
def csi(pred, target, threshold=1.0):
    """Calculates the Critical Success Index (CSI)."""
    pred_bin = (pred >= threshold)
    target_bin = (target >= threshold)
    tp = np.sum(pred_bin & target_bin)
    fp = np.sum(pred_bin & ~target_bin)
    fn = np.sum(~pred_bin & target_bin)
    return tp / (tp + fp + fn + 1e-8)

def fss(pred, target, window_size=5, threshold=2.5):
    """Calculates the Fractions Skill Score (FSS)."""
    pred_bin = (pred >= threshold).astype(float)
    target_bin = (target >= threshold).astype(float)
    smooth_pred = uniform_filter(pred_bin, size=window_size)
    smooth_target = uniform_filter(target_bin, size=window_size)
    numerator = np.mean((smooth_pred - smooth_target) ** 2)
    denominator = np.mean(smooth_pred ** 2 + smooth_target ** 2)
    return 1 - (numerator / (denominator + 1e-8))

def crps_ensemble_numpy(ensemble_pred, target):
    """
    Calculates the Continuous Ranked Probability Score (CRPS) for an ensemble forecast
    using a direct numpy implementation.

    Args:
        ensemble_pred (np.ndarray): Ensemble forecast with shape (num_samples, H, W).
        target (np.ndarray): Ground truth observation with shape (H, W).
    
    Returns:
        float: The mean CRPS value over the spatial domain (H, W).
    """
    # Ensure target can be broadcast to the ensemble shape
    # Add a new axis to target: (H, W) -> (1, H, W)
    target_broadcast = np.expand_dims(target, axis=0)
    
    # Calculate the first term: Mean absolute error between ensemble and target
    term1 = np.mean(np.abs(ensemble_pred - target_broadcast))

    # Calculate the second term: Mean absolute error between all pairs of ensemble members
    # Get the number of ensemble members
    num_samples = ensemble_pred.shape[0]
    
    # This is a more efficient way to calculate the mean difference between all pairs
    # without a double loop.
    # Sort the ensemble predictions along the sample dimension for each pixel
    ensemble_sorted = np.sort(ensemble_pred, axis=0)
    
    # Calculate the difference between consecutive sorted members
    diffs = ensemble_sorted[1:, :, :] - ensemble_sorted[:-1, :, :]
    
    # Create weights for the calculation
    # The weight for the difference between the i-th and (i+1)-th sorted member is 2 * i * (M - i)
    # where M is the number of members.
    weights = (2 * np.arange(1, num_samples) * (num_samples - np.arange(1, num_samples))).reshape(-1, 1, 1)
    
    # Calculate the second term using the weighted sum of differences
    term2 = np.mean(weights * diffs) / (num_samples * (num_samples - 1))
    
    # CRPS is term1 - term2
    return term1 - term2


def crps(ensemble_pred, target):
    """
    Wrapper for the custom numpy CRPS implementation. This is the function
    that will be called by EvaluationCallback.
    """
    # Use the new numpy-based function
    return crps_ensemble_numpy(ensemble_pred, target)

# --- Plotting Functions ---
# (The plotting functions remain unchanged as they are correct)
def get_imd_colormap():
    """Returns a standardized colormap and norm for precipitation plotting."""
    colors = [
        (1.0, 1.0, 1.0),    # White
        (0.7, 0.85, 0.95),  # Light Blue
        (0.2, 0.6, 0.85),   # Medium Blue
        (0.5, 0.85, 0.5),   # Light Green
        (1.0, 0.8, 0.2),    # Yellow/Light Orange
        (0.9, 0.0, 0.0),    # Red
    ]
    boundaries = [0.0, 0.1, 2.5, 7.6, 16.0, 50.0, 100.0]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm

def visualize_step_grid(x, y, y_hat, epoch, step, save_dir):
    """Saves a grid image comparing input, target, and the mean prediction."""
    x, y, y_hat = x[0].cpu().squeeze(1), y[0].cpu().squeeze(1), y_hat[0].cpu().squeeze(1)
    phys_x = denormalize_data(x.numpy())
    phys_y = denormalize_data(y.numpy())
    phys_y_hat = denormalize_data(y_hat.numpy())
    cmap, norm = get_imd_colormap()
    num_input_frames = phys_x.shape[0]
    num_forecast_steps = phys_y.shape[0]
    fig, axes = plt.subplots(3, num_forecast_steps, figsize=(num_forecast_steps * 3, 9.5), gridspec_kw={'hspace': 0.3, 'wspace': 0.1})
    fig.suptitle(f"Mean Prediction: Epoch {epoch}, Step {step}", fontsize=20)
    for i in range(num_forecast_steps):
        ax_in = axes[0, i]
        if i < num_input_frames: 
             ax_in.imshow(phys_x[i, :, :], cmap=cmap, norm=norm)
        else: 
            ax_in.axis('off')
        ax_tgt = axes[1, i]
        im = ax_tgt.imshow(phys_y[i, :, :], cmap=cmap, norm=norm)
        ax_pred = axes[2, i]
        ax_pred.imshow(phys_y_hat[i, :, :], cmap=cmap, norm=norm)
        lead_time_min = (i + 1) * 30
        ax_pred.set_xlabel(f"+{lead_time_min} min")
    axes[0, 0].set_ylabel("Input Context", fontsize=16, labelpad=10)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=16, labelpad=10)
    axes[2, 0].set_ylabel("Mean Prediction", fontsize=16, labelpad=10)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Precipitation Rate (mm/hr)", rotation=270, labelpad=20)
    save_path = Path(save_dir) / "visualizations_mean" / f"epoch_{epoch}_step_{step}.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path)
    plt.close(fig)

def visualize_ensemble_grid(x, y, ensemble, epoch, step, save_dir, num_members_to_show=4):
    """
    Saves a grid image comparing input, ground truth, and several members of the prediction ensemble.
    """
    x_sample, y_sample = x[0].cpu().squeeze(1), y[0].cpu().squeeze(1)
    ensemble_sample = ensemble[:, 0, ...].cpu().squeeze(2)

    phys_y = denormalize_data(y_sample.numpy())
    phys_ensemble = denormalize_data(ensemble_sample.numpy())

    cmap, norm = get_imd_colormap()
    num_forecast_steps = phys_y.shape[0]
    num_members_to_show = min(num_members_to_show, phys_ensemble.shape[0])

    fig, axes = plt.subplots(
        1 + num_members_to_show, num_forecast_steps, 
        figsize=(num_forecast_steps * 3, 3 * (1 + num_members_to_show)),
        gridspec_kw={'hspace': 0.15, 'wspace': 0.1}
    )
    fig.suptitle(f"Ensemble Validation: Epoch {epoch}, Step {step}", fontsize=20)

    for i in range(num_forecast_steps):
        ax_tgt = axes[0, i]
        im = ax_tgt.imshow(phys_y[i, :, :], cmap=cmap, norm=norm)
        if i == 0:
            ax_tgt.set_ylabel("Ground Truth", fontsize=16, labelpad=15)

    for member_idx in range(num_members_to_show):
        for i in range(num_forecast_steps):
            ax_pred = axes[member_idx + 1, i]
            ax_pred.imshow(phys_ensemble[member_idx, i, :, :], cmap=cmap, norm=norm)
            if i == 0:
                ax_pred.set_ylabel(f"Member {member_idx+1}", fontsize=16, labelpad=15)
            if member_idx == num_members_to_show - 1:
                lead_time_min = (i + 1) * 30
                ax_pred.set_xlabel(f"+{lead_time_min} min")

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Precipitation Rate (mm/hr)", rotation=270, labelpad=20)
    
    save_path = Path(save_dir) / "visualizations_ensemble" / f"epoch_{epoch}_step_{step}.png"
    save_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(save_path)
    plt.close(fig)