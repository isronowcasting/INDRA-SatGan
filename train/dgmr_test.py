# dgmr_test.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import h5py
import pandas as pd
from datetime import datetime, timedelta
import sys

# --- DGMR & PyTorch dependencies ---
import torch

# Ensure the parent directory is in the Python path to find the 'dgmr' module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dgmr import DGMR


# --- NEW dependency for manual FSS calculation ---
try:
    from scipy.ndimage import uniform_filter
except ImportError:
    print("This script now requires SciPy for manual FSS calculation.")
    print("Please install it: pip install scipy")
    exit()

# --- Geospatial plotting libraries ---
try:
    from mpl_toolkits.basemap import Basemap
except ImportError:
    print("This script requires Basemap for map plotting.")
    print("It is highly recommended to install it with conda:")
    print("conda install -c conda-forge basemap")
    exit()


# --- PLOTTING CONFIGURATION (for Publication Quality) ---
plt.rcParams.update({
    'font.size': 14, 'font.family': 'sans-serif', 'axes.titlesize': 18,
    'axes.labelsize': 16, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'legend.fontsize': 12,
})


# --- DATA & HELPER FUNCTIONS ---

def get_region_config(region_name):
    base_region = region_name.split('_')[0]
    is_test_set = '_test' in region_name
    if base_region == 'south':
        config = {
            'name': 'Southern India', 'pretty_name': 'Southern India',
            'projection': {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'},
            'x1': 71.0, 'x2': 83.8, 'y1': 5.0, 'y2': 17.8,
            'xpixelsize': 0.1, 'ypixelsize': 0.1, 'cartesian': False, 'zerovalue': 0.0
        }
    elif base_region == 'ne':
        config = {
            'name': 'Northeastern India', 'pretty_name': 'Northeastern India',
            'projection': {'proj': 'longlat', 'ellps': 'WGS84', 'datum': 'WGS84'},
            'x1': 86.1, 'x2': 98.9, 'y1': 19.1, 'y2': 31.9,
            'xpixelsize': 0.1, 'ypixelsize': 0.1, 'cartesian': False, 'zerovalue': 0.0
        }
    else:
        raise ValueError(f"Unknown base region '{base_region}'.")
    if is_test_set:
        config['pretty_name'] += ' (Test Set)'
    return config

def extract_timestamp(fpath):
    with h5py.File(fpath, 'r') as f:
        seconds_since_1980 = f['time'][()]
    DATA_EPOCH = datetime(1980, 1, 1)
    correct_datetime = DATA_EPOCH + timedelta(seconds=int(seconds_since_1980))
    return correct_datetime.timestamp()

def unix_to_yyyymmddhhmm(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp).strftime('%Y%m%d%H%M')

def is_sequence_valid(filepaths, interval_minutes=30, tolerance_seconds=300):
    expected_delta = interval_minutes * 60
    for i in range(len(filepaths) - 1):
        try:
            time_diff = int(extract_timestamp(filepaths[i+1])) - int(extract_timestamp(filepaths[i]))
            if abs(time_diff - expected_delta) > tolerance_seconds: return False
        except Exception: return False
    return True

def load_sequence(filepaths):
    # Load the entire dataset (including channel dim) to match training.
    frames = [h5py.File(fpath, 'r')['precipitationCal'][:] for fpath in filepaths]
    return np.stack(frames)

def transform_to_physical(norm_data, clip_value=100):
    # Squeeze out the channel dimension if it exists, as physical data is 2D/3D
    norm_data = np.squeeze(norm_data, axis=-3) if norm_data.ndim == 4 else norm_data
    scale_factor = np.log(clip_value + 1)
    original_data = np.exp(norm_data * scale_factor) - 1
    original_data[original_data < 0] = 0.0
    return original_data

def get_contingency_arrays(pred, obs, thr):
    predb, obsb = pred > thr, obs > thr
    hits = np.logical_and(predb, obsb)
    misses = np.logical_and(np.logical_not(predb), obsb)
    false_alarms = np.logical_and(predb, np.logical_not(obsb))
    return {'hits': hits, 'misses': misses, 'false_alarms': false_alarms}


# --- MANUAL VERIFICATION FUNCTIONS ---

def manual_fss(forecast, observation, thr, scale):
    forecast_bin = (forecast >= thr).astype(float)
    observation_bin = (observation >= thr).astype(float)
    forecast_fractions = uniform_filter(forecast_bin, size=scale)
    observation_fractions = uniform_filter(observation_bin, size=scale)
    mse = np.mean((forecast_fractions - observation_fractions) ** 2)
    mse_ref = np.mean(forecast_fractions ** 2) + np.mean(observation_fractions ** 2)
    if mse_ref == 0:
        return 1.0
    return 1.0 - (mse / mse_ref)

def manual_det_cat_fct(forecast, observation, thr):
    epsilon = 1e-6
    pred_b = forecast >= thr
    obs_b = observation >= thr
    hits = np.sum(np.logical_and(pred_b, obs_b))
    misses = np.sum(np.logical_and(np.logical_not(pred_b), obs_b))
    false_alarms = np.sum(np.logical_and(pred_b, np.logical_not(obs_b)))
    pod = hits / (hits + misses + epsilon)
    far = false_alarms / (hits + false_alarms + epsilon)
    csi = hits / (hits + misses + false_alarms + epsilon)
    return {'POD': pod, 'FAR': far, 'CSI': csi}


# --- DGMR MODEL FUNCTION ---

def generate_dgmr_forecast(model, train_data_norm, device):
    """Generates a forecast using the loaded DGMR model."""
    # --- THIS IS THE CORRECTED LINE ---
    # Convert numpy array (float64) to a tensor and cast to float32 to match model weights.
    input_tensor = torch.from_numpy(train_data_norm).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        forecast_norm_tensor = model(input_tensor)
    
    forecast_norm_np = forecast_norm_tensor.squeeze(0).cpu().numpy()
    forecast_physical = transform_to_physical(forecast_norm_np)
    return forecast_physical


# --- PUBLICATION-QUALITY PLOTTING FUNCTIONS ---

def _draw_map_features(basemap, shapefile_path):
    parallels = np.arange(0., 90., 5.)
    basemap.drawparallels(parallels, labels=[1, 0, 0, 0], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    meridians = np.arange(0., 180., 5.)
    basemap.drawmeridians(meridians, labels=[0, 0, 0, 1], fontsize=10, color='lightgray', linewidth=0.5, zorder=10)
    shapefile_base = os.path.splitext(shapefile_path)[0]
    basemap.readshapefile(shapefile_base, 'boundary', drawbounds=True, color='dimgray', linewidth=0.75, zorder=11)
    basemap.drawcoastlines(linewidth=0.75, color='black', zorder=12)
    basemap.drawcountries(linewidth=0.75, color='black', zorder=12)

def plot_spatial_maps(plots_dir, total_hits, total_misses, total_fas, threshold, metadata, shapefile_path):
    print("Generating Plot: Spatial Score Maps on Geography")
    epsilon = 1e-6
    spatial_pod = total_hits / (total_hits + total_misses + epsilon)
    spatial_far = total_fas / (total_hits + total_fas + epsilon)
    spatial_csi = total_hits / (total_hits + total_misses + total_fas + epsilon)
    consistent_cmap = 'magma'
    spatial_metrics = {
        'POD': {'data': spatial_pod, 'cmap': consistent_cmap, 'label': 'Probability of Detection'},
        'FAR': {'data': spatial_far, 'cmap': consistent_cmap, 'label': 'False Alarm Ratio'},
        'CSI': {'data': spatial_csi, 'cmap': consistent_cmap, 'label': 'Critical Success Index'}
    }
    ny, nx = spatial_pod.shape
    lons, lats = np.linspace(metadata['x1'], metadata['x2'], nx), np.linspace(metadata['y1'], metadata['y2'], ny)
    for name, metric in spatial_metrics.items():
        fig, ax = plt.subplots(figsize=(9, 9))
        m = Basemap(projection='cyl', llcrnrlon=metadata['x1'], urcrnrlon=metadata['x2'],
                    llcrnrlat=metadata['y1'], urcrnrlat=metadata['y2'], resolution='i', ax=ax)
        im = m.pcolormesh(lons, lats, metric['data'], latlon=True, cmap=metric['cmap'], vmin=0, vmax=1, zorder=5, alpha=0.75)
        _draw_map_features(m, shapefile_path)
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', pad=0.08, shrink=0.7)
        cbar.set_label(f'{name} Score')
        ax.set_title(f'Spatially Aggregated {metric["label"]}\nOver {metadata["pretty_name"]} (Threshold: {threshold} mm/hr)')
        plt.savefig(os.path.join(plots_dir, f"plot_spatial_{name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

def plot_fss(plots_dir, lead_times, model_scores, persistence_scores, fss_scales, p_size_km, thr, region_name):
    print("Generating Plot: FSS (Multi-panel)")
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    colors = {'DGMR': '#1f77b4', 'Persistence': '#ff7f0e'}
    for i, scale_pix in enumerate(fss_scales):
        ax = axes[i]
        scale_km = scale_pix * p_size_km
        ax.plot(lead_times, model_scores['FSS'][i, :], marker='o', color=colors['DGMR'], label='DGMR')
        ax.plot(lead_times, persistence_scores['FSS'][i, :], marker='s', linestyle='--', color=colors['Persistence'], label='Persistence')
        ax.set_title(f'Scale: {scale_pix} pixels (~{scale_km:.0f} km)')
        ax.grid(True, which='both', linestyle=':', linewidth=0.7); ax.set_ylim(0, 1.05); ax.set_xticks(lead_times); ax.legend()
    fig.suptitle(f'Fractions Skill Score (FSS) for {region_name}\n(Threshold: {thr} mm/hr)', y=0.98)
    fig.supxlabel('Lead Time (minutes)'); fig.supylabel('FSS (Higher is Better)')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(plots_dir, "plot_fss_multipanel.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_fss_csi_comparison(plots_dir, lead_times, model_scores, persistence_scores, fss_scales_pixels, thr, region_name):
    print("Generating Plot: FSS and CSI Comparison")
    fss_scale_to_plot = 4
    scale_idx = fss_scales_pixels.index(fss_scale_to_plot) if fss_scale_to_plot in fss_scales_pixels else 0
    fss_scale_to_plot = fss_scales_pixels[scale_idx]
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'DGMR': '#1f77b4', 'Persistence': '#ff7f0e'}
    styles = {'FSS': '-', 'CSI': '--'}
    markers = {'DGMR': 'o', 'Persistence': 's'}
    ax.plot(lead_times, model_scores['FSS'][scale_idx, :], color=colors['DGMR'], linestyle=styles['FSS'], marker=markers['DGMR'], label=f"DGMR - FSS ({fss_scale_to_plot}px)")
    ax.plot(lead_times, model_scores['CSI'], color=colors['DGMR'], linestyle=styles['CSI'], marker=markers['DGMR'], label='DGMR - CSI')
    ax.plot(lead_times, persistence_scores['FSS'][scale_idx, :], color=colors['Persistence'], linestyle=styles['FSS'], marker=markers['Persistence'], label=f"Persistence - FSS ({fss_scale_to_plot}px)")
    ax.plot(lead_times, persistence_scores['CSI'], color=colors['Persistence'], linestyle=styles['CSI'], marker=markers['Persistence'], label='Persistence - CSI')
    ax.set_title(f'Skill Score Comparison for {region_name}\n(Threshold: {thr} mm/hr)')
    ax.set_xlabel('Lead Time (minutes)'); ax.set_ylabel('Skill Score')
    ax.set_ylim(0, 1.05); ax.set_xticks(lead_times)
    ax.grid(True, which='both', linestyle=':', linewidth=0.7); ax.legend(title='Method - Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "plot_fss_csi_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def print_and_save_metrics(plots_dir, lead_times, model_scores, persistence_scores, fss_scales, p_size_km, region_name):
    data = []
    for metric in ['POD', 'FAR', 'CSI']:
        for i, lead_time in enumerate(lead_times):
            data.append(['Categorical', metric, f'{lead_time} min', model_scores[metric][i], persistence_scores[metric][i]])
    for s_idx, scale_pix in enumerate(fss_scales):
        metric_name = f'FSS ({(scale_pix * p_size_km):.0f} km)'
        for i, lead_time in enumerate(lead_times):
            data.append(['FSS', metric_name, f'{lead_time} min', model_scores['FSS'][s_idx, i], persistence_scores['FSS'][s_idx, i]])
    df = pd.DataFrame(data, columns=['Score Type', 'Metric', 'Lead Time', 'DGMR', 'Persistence']).round(4)
    print(f"\n\n--- FINAL AVERAGED SCORES ({region_name}) ---"); print(df.to_string())
    output_path = os.path.join(plots_dir, f"final_scores_table_{region_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt")
    with open(output_path, 'w') as f:
        f.write(f"--- FINAL AVERAGED SCORES ({region_name}) ---\n\n" + df.to_string())
    print(f"\nFinal scores table saved to '{output_path}'")


def main(region):
    # --- 1. HARDCODED CONFIGURATION ---
    checkpoint_path = "/home/sac/data_67/Nowcasting/vinayak/ModelSouth/train/checkpoints_full/epochepoch=137.ckpt"
    shapefile_path = "/home/sac/data_67/Nowcasting/vishwajit/PreDiff_ISRO/Shapefile/states_IND.shp"

    metadata = get_region_config(region)
    base_dir = f'/home/sac/data_67/Nowcasting/data/required_{region}'
    plots_dir = f'plots_dgmr_{region}'
    forecast_dir = f'forecast_dgmr_{region}'
    os.makedirs(plots_dir, exist_ok=True); os.makedirs(forecast_dir, exist_ok=True)
    print(f"--- Running analysis for region: {metadata['pretty_name']} ---")
    print(f"--- Data source: {base_dir} ---")
    print(f"--- Plots will be saved to: {plots_dir} ---")
    print(f"--- Forecasts will be cached in: {forecast_dir} ---")

    # --- 2. LOAD DGMR MODEL ---
    print("\n--- Loading DGMR Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    try:
        model = DGMR.load_from_checkpoint(checkpoint_path, map_location=device)
        model.to(device)
        model.eval()
        print(f"Successfully loaded DGMR model from {checkpoint_path}")
    except Exception as e:
        print(f"Error loading DGMR model: {e}")
        print("Please ensure the DGMR class is defined and the checkpoint is valid.")
        exit()

    # --- 3. INITIALIZATION ---
    n_train, n_obs = 6, 6
    seq_len = n_train + n_obs
    interval_minutes = 30
    verification_threshold = 1.0
    fss_scales_pixels = [1, 2, 4, 8]
    all_files = sorted([os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.HDF5')])
    n_seq = len(all_files) // seq_len
    if n_seq == 0:
        print("Not enough files to form a complete sequence."); return

    # --- 4. PASS 1: FORECAST GENERATION (CACHE CHECKING) ---
    print("\n--- Pass 1: Generating/Checking Forecast Cache ---")
    for i in range(n_seq):
        seq_files = all_files[i*seq_len:(i+1)*seq_len]
        if not is_sequence_valid(seq_files, interval_minutes):
            print(f"  -> Skipping invalid sequence starting with {os.path.basename(seq_files[0])}")
            continue
            
        first_timestamp_unix = extract_timestamp(seq_files[0])
        timestamp_str = unix_to_yyyymmddhhmm(first_timestamp_unix)
        forecast_path = os.path.join(forecast_dir, f"{timestamp_str}.npy")

        if os.path.exists(forecast_path):
            print(f"  ({i+1}/{n_seq}) Cache hit for {timestamp_str}, skipping generation.")
            continue
        else:
            print(f"  ({i+1}/{n_seq}) Cache miss for {timestamp_str}, generating forecast...")
            train_norm = load_sequence(seq_files[:n_train])
            train_norm[~np.isfinite(train_norm)] = 0.0
            forecast_physical = generate_dgmr_forecast(model, train_norm, device)
            np.save(forecast_path, forecast_physical)
            print(f"      -> Saved forecast to: {forecast_path}")

    # --- 5. PASS 2: VERIFICATION & AGGREGATION ---
    print("\n--- Pass 2: Verifying All Forecasts ---")
    all_scores_model_fss, all_scores_pers_fss = [], []
    all_scores_model_cat, all_scores_pers_cat = [], []
    try:
        with h5py.File(all_files[0], 'r') as f:
            spatial_shape = f['precipitationCal'][0].shape
    except (IOError, IndexError):
        print("Error reading the first file. Exiting."); return
    total_hits_model = np.zeros(spatial_shape)
    total_misses_model = np.zeros(spatial_shape)
    total_fas_model = np.zeros(spatial_shape)

    for i in range(n_seq):
        print(f"  Verifying sequence {i+1}/{n_seq}...")
        seq_files = all_files[i*seq_len:(i+1)*seq_len]
        if not is_sequence_valid(seq_files, interval_minutes): continue
            
        first_timestamp_unix = extract_timestamp(seq_files[0])
        timestamp_str = unix_to_yyyymmddhhmm(first_timestamp_unix)
        forecast_path = os.path.join(forecast_dir, f"{timestamp_str}.npy")

        forecast_physical = np.load(forecast_path)
        obs_norm = load_sequence(seq_files[n_train:])
        obs_norm[~np.isfinite(obs_norm)] = 0.0
        obs_physical = transform_to_physical(obs_norm)
        last_train_norm = load_sequence([seq_files[n_train-1]])
        last_train_norm[~np.isfinite(last_train_norm)] = 0.0
        last_train_physical = transform_to_physical(last_train_norm)
        persistence_physical = np.stack([last_train_physical[0]] * n_obs)
        
        seq_model_fss, seq_pers_fss = np.zeros((len(fss_scales_pixels), n_obs)), np.zeros((len(fss_scales_pixels), n_obs))
        seq_model_cat = {'CSI': np.zeros(n_obs), 'POD': np.zeros(n_obs), 'FAR': np.zeros(n_obs)}
        seq_pers_cat = {'CSI': np.zeros(n_obs), 'POD': np.zeros(n_obs), 'FAR': np.zeros(n_obs)}
        
        for k in range(n_obs):
            model_cat = manual_det_cat_fct(forecast_physical[k], obs_physical[k], thr=verification_threshold)
            pers_cat = manual_det_cat_fct(persistence_physical[k], obs_physical[k], thr=verification_threshold)
            for score in ['CSI', 'POD', 'FAR']:
                seq_model_cat[score][k] = model_cat.get(score, np.nan)
                seq_pers_cat[score][k] = pers_cat.get(score, np.nan)
            
            for s, scale in enumerate(fss_scales_pixels):
                seq_model_fss[s, k] = manual_fss(forecast_physical[k], obs_physical[k], thr=verification_threshold, scale=scale)
                seq_pers_fss[s, k] = manual_fss(persistence_physical[k], obs_physical[k], thr=verification_threshold, scale=scale)
            
            contingency_arrays = get_contingency_arrays(forecast_physical[k], obs_physical[k], thr=verification_threshold)
            total_hits_model += contingency_arrays['hits']
            total_misses_model += contingency_arrays['misses']
            total_fas_model += contingency_arrays['false_alarms']
            
        all_scores_model_fss.append(seq_model_fss); all_scores_pers_fss.append(seq_pers_fss)
        all_scores_model_cat.append(seq_model_cat); all_scores_pers_cat.append(seq_pers_cat)

    if not all_scores_model_fss:
        print("No valid sequences were processed for verification."); return

    # --- 6. AVERAGING, PLOTTING, AND REPORTING ---
    print("\n--- Pass 3: Averaging Scores and Generating Plots ---")
    model_scores = {'FSS': np.nanmean(np.stack(all_scores_model_fss), axis=0), **{s: np.nanmean([d[s] for d in all_scores_model_cat], axis=0) for s in ['CSI', 'POD', 'FAR']}}
    persistence_scores = {'FSS': np.nanmean(np.stack(all_scores_pers_fss), axis=0), **{s: np.nanmean([d[s] for d in all_scores_pers_cat], axis=0) for s in ['CSI', 'POD', 'FAR']}}
    lead_times = np.arange(1, n_obs + 1) * interval_minutes
    pixel_size_km = metadata['xpixelsize'] * 111.0
    
    plot_fss(plots_dir, lead_times, model_scores, persistence_scores, fss_scales_pixels, pixel_size_km, verification_threshold, metadata['pretty_name'])
    plot_fss_csi_comparison(plots_dir, lead_times, model_scores, persistence_scores, fss_scales_pixels, verification_threshold, metadata['pretty_name'])
    plot_spatial_maps(plots_dir, total_hits_model, total_misses_model, total_fas_model, verification_threshold, metadata, shapefile_path)

    print(f"\nProcessing complete. All plots saved in '{plots_dir}'.")
    
    print_and_save_metrics(plots_dir, lead_times, model_scores, persistence_scores, fss_scales_pixels, pixel_size_km, metadata['pretty_name'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DGMR model verification for a specific region.")
    parser.add_argument('--region', type=str, choices=['south', 'ne', 'south_test', 'ne_test'], required=True, help="The geographic region to process.")
    args = parser.parse_args()
    
    main(region=args.region)