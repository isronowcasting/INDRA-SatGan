import matplotlib.pyplot as plt
import numpy as np
import re
from collections import defaultdict
import os

# --- Data from your ProGAN Model (Epoch 92) ---
log_data_progan = """
--- Epoch 92 Validation Summary ---
pred_FSS@0.5_30-min      0.965610
pers_FSS@0.5_30-min      0.971553
pred_FSS@2.5_30-min      0.919866
pers_FSS@2.5_30-min      0.951440
pred_FSS@7.6_30-min      0.796159
pers_FSS@7.6_30-min      0.908072
pred_FSS@16_30-min       0.701388
pers_FSS@16_30-min       0.866763
pred_FSS@50_30-min       0.544650
pers_FSS@50_30-min       0.854574
pred_FSS@0.5_60-min      0.965656
pers_FSS@0.5_60-min      0.939649
pred_FSS@2.5_60-min      0.941124
pers_FSS@2.5_60-min      0.898143
pred_FSS@7.6_60-min      0.859261
pers_FSS@7.6_60-min      0.786930
pred_FSS@16_60-min       0.859547
pers_FSS@16_60-min       0.774351
pred_FSS@50_60-min       0.821777
pers_FSS@50_60-min       0.655029
pred_FSS@0.5_90-min      0.955303
pers_FSS@0.5_90-min      0.918410
pred_FSS@2.5_90-min      0.911225
pers_FSS@2.5_90-min      0.852562
pred_FSS@7.6_90-min      0.774178
pers_FSS@7.6_90-min      0.702549
pred_FSS@16_90-min       0.710635
pers_FSS@16_90-min       0.649815
pred_FSS@50_90-min       0.557971
pers_FSS@50_90-min       0.419783
pred_FSS@0.5_120-min     0.939612
pers_FSS@0.5_120-min     0.902738
pred_FSS@2.5_120-min     0.864209
pers_FSS@2.5_120-min     0.805750
pred_FSS@7.6_120-min     0.600043
pers_FSS@7.6_120-min     0.623390
pred_FSS@16_120-min      0.488211
pers_FSS@16_120-min      0.531300
pred_FSS@50_120-min      0.375491
pers_FSS@50_120-min      0.370262
pred_FSS@0.5_150-min     0.915325
pers_FSS@0.5_150-min     0.881790
pred_FSS@2.5_150-min     0.787691
pers_FSS@2.5_150-min     0.754192
pred_FSS@7.6_150-min     0.462530
pers_FSS@7.6_150-min     0.518793
pred_FSS@16_150-min      0.288245
pers_FSS@16_150-min      0.336287
pred_FSS@50_150-min      0.245768
pers_FSS@50_150-min      0.297455
pred_FSS@0.5_180-min     0.884914
pers_FSS@0.5_180-min     0.865483
pred_FSS@2.5_180-min     0.698328
pers_FSS@2.5_180-min     0.710944
pred_FSS@7.6_180-min     0.378941
pers_FSS@7.6_180-min     0.450359
pred_FSS@16_180-min      0.163735
pers_FSS@16_180-min      0.292473
pred_FSS@50_180-min      0.000214
pers_FSS@50_180-min      0.286005
"""

# --- Manually extracted data from your Py-STEPS plots ("22 km" scale) ---
pysteps_fss_data = {
    "FSS@2.5":  [0.83, 0.72, 0.61, 0.53, 0.48, 0.41],
    "FSS@7.6":  [0.73, 0.55, 0.42, 0.32, 0.26, 0.21],
    "FSS@16.0": [0.61, 0.41, 0.28, 0.20, 0.15, 0.11],
    "FSS@50.0": [0.35, 0.16, 0.08, 0.06, 0.04, 0.03],
}

def parse_fss_log(log_data):
    """Parses FSS data for both prediction and persistence models."""
    metrics = defaultdict(lambda: defaultdict(list))
    # Regex to find metric name (pred/pers, score type, threshold) and value
    pattern = re.compile(r"(pred|pers)_(FSS@[\d\.]+)_\d+-min\s+([\d\.]+)")
    
    # Organize data by lead time first
    data_by_lead_time = defaultdict(dict)
    for line in log_data.strip().split('\n'):
        match = pattern.search(line)
        if match:
            model_type = "ProGAN" if match.group(1) == 'pred' else "Persistence"
            metric_name = match.group(2)
            value = float(match.group(3))
            
            # This logic is a bit complex due to the log format, but it works
            # We add to a list, assuming the log is ordered by lead time
            metrics[metric_name][model_type].append(value)
            
    # Define lead times based on how many entries we found (should be 6)
    num_entries = len(next(iter(next(iter(metrics.values())).values())))
    lead_times = [30 * (i + 1) for i in range(num_entries)]
    
    return lead_times, metrics

def plot_three_way_comparison(lead_times, progan_data, pysteps_data, thresholds):
    """Generates and saves a three-way FSS comparison plot."""
    num_thresholds = len(thresholds)
    fig, axes = plt.subplots(
        nrows=(num_thresholds + 1) // 2, 
        ncols=2, 
        figsize=(16, 5 * ((num_thresholds + 1) // 2)), 
        sharex=True, 
        sharey=True
    )
    axes = axes.flatten()
    fig.suptitle('FSS Comparison: ProGAN vs. Py-STEPS (22km) vs. Persistence', fontsize=22, y=0.98)

    for i, thr in enumerate(thresholds):
        ax = axes[i]
        progan_metric_key = f"FSS@{thr}"
        pysteps_metric_key = f"FSS@{thr}" # Assuming keys match, handle if not
        
        # --- Plotting Data ---
        # 1. ProGAN Model
        ax.plot(lead_times, progan_data[progan_metric_key]['ProGAN'], marker='o', linestyle='-', label='ProGAN (Your Model)', color='royalblue', lw=2.5, zorder=10)
        
        # 2. Persistence Baseline
        ax.plot(lead_times, progan_data[progan_metric_key]['Persistence'], marker='x', linestyle='--', label='Persistence', color='coral', lw=2.0, zorder=5)
        
        # 3. Py-STEPS Baseline
        if pysteps_metric_key in pysteps_data:
            ax.plot(lead_times, pysteps_data[pysteps_metric_key], marker='s', linestyle=':', label='Py-STEPS (22km)', color='green', lw=2.0)

        # --- Formatting ---
        ax.set_title(f'Threshold: {thr} mm/hr', fontsize=14)
        ax.grid(True, which='both', linestyle=':', linewidth=0.7)
        ax.set_xticks(lead_times)
        ax.legend(fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.set_xlim(lead_times[0]-5, lead_times[-1]+5)

    fig.text(0.5, 0.04, 'Lead Time (minutes)', ha='center', va='center', fontsize=18)
    fig.text(0.07, 0.5, 'Fractions Skill Score (FSS)', ha='center', va='center', rotation='vertical', fontsize=18)

    plt.tight_layout(rect=[0.08, 0.05, 1, 0.95])
    
    output_dir = os.path.expanduser("~/data_67/Nowcasting/vinayak/ModelSouth/test/")
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "FSS_ProGAN_vs_Baselines_Comparison.png")
    
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_filename}")
    plt.show()

# --- Main execution block ---
if __name__ == "__main__":
    # Corrected log parsing logic to handle the new format
    def parse_new_format(log_data):
        metrics = defaultdict(lambda: defaultdict(list))
        data_by_lead = {}
        # Find all metric lines
        pattern = re.compile(r"(pred|pers)_(FSS@[\d\._]+)-min\s+([\d\.]+)")
        
        # Use a temporary dict to group by lead time before ordering
        temp_metrics = defaultdict(lambda: defaultdict(list))
        lead_times = sorted(list(set([int(lt) for lt in re.findall(r'_(\d+)-min', log_data)])))

        for lt in lead_times:
            for line in log_data.strip().split('\n'):
                 # Look for metric for the current lead time
                 metric_pattern = re.compile(r"(pred|pers)_(FSS@[\d\.]+)_"+str(lt)+r"-min\s+([\d\.]+)")
                 match = metric_pattern.search(line)
                 if match:
                     model_type = "ProGAN" if match.group(1) == 'pred' else "Persistence"
                     metric_name = match.group(2)
                     value = float(match.group(3))
                     temp_metrics[metric_name][model_type].append(value)

        return lead_times, temp_metrics

    # Use the new parser
    lead_times, parsed_progan_data = parse_new_format(log_data_progan)
    
    if not lead_times or not parsed_progan_data:
        print("Could not parse the log data.")
    else:
        # Define thresholds based on the Py-STEPS plots for direct comparison
        comparison_thresholds = [2.5, 7.6, 16.0, 50.0]
        plot_three_way_comparison(lead_times, parsed_progan_data, pysteps_fss_data, comparison_thresholds)