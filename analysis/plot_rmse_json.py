import matplotlib
matplotlib.use('macosx')

import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import seaborn as sns

def plot_rmse_json(json_path, rename_map=None, xlim=None, ylim=None):
    if rename_map is None:
        rename_map = {}
    if not os.path.exists(json_path):
        print(f"Error: file '{json_path}' not found.")
        return

    print(f"Loading data from {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    timestamps = np.array(data['timestamps'])
    
    # Process combined RMSE over time
    combined_rmse = np.array([np.nan if x is None else float(x) for x in data['combined_rmse_mm']])
    valid_comb = ~np.isnan(combined_rmse)

    overall_comb_rmse = np.nan
    if np.any(valid_comb):
        # Calculate the scalar overall combined RMSE across the entire trajectory
        overall_comb_rmse = np.sqrt(np.mean(combined_rmse[valid_comb] ** 2))

    # Setup the plot
    plt.figure(figsize=(8, 4))

    drones = data.get('drones', {})
    drone_ids = list(drones.keys())

    # 1. For each drone, draw a line showing its RMSE over time with different colors
    colors = sns.color_palette("husl", max(1, len(drone_ids)))
    for i, drone_id in enumerate(drone_ids):
        rmse_data = np.array([np.nan if x is None else float(x) for x in drones[drone_id]['rmse_mm']])
        # Use the map to rename labels
        label = rename_map.get(drone_id, drone_id)
        plt.plot(timestamps, rmse_data, color=colors[i], linewidth=1.5, alpha=0.8, label=label)

    # 2. Plot a thick black line to show the overall RMSE as a function of time
    plt.plot(timestamps, combined_rmse, 'k-', linewidth=3, alpha=0.9, label='Average')

    # 3. Plot a dashed red line to show the overall combined RMSE (scalar mean)
    if not np.isnan(overall_comb_rmse):
        plt.axhline(overall_comb_rmse, color='r', linestyle='--', linewidth=2, label='RMSE Illumination')
        
        # Annotate the line with the value
        x_pos = args.annotate_x if args.annotate_x else xlim[1]
        plt.text(x_pos, overall_comb_rmse + 0.5, f'{overall_comb_rmse:.1f} mm', color='red', 
                 ha='right', va='bottom', fontsize=11, fontweight='bold')

    # Formatting
    plt.xlabel('Time (s)', fontsize=14)
    plt.ylabel('')
    plt.text(0, 1.02, 'Error (mm)', transform=plt.gca().transAxes, fontsize=14, va='bottom', ha='center')

    if xlim:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(0, timestamps[-1] if len(timestamps) > 0 else 1)
        
    if ylim:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(bottom=0)

    plt.grid(axis='y', linestyle=':', alpha=0.7)
    sns.despine()
    
    # Legend config
    handles, labels = plt.gca().get_legend_handles_labels()
    
    def sort_key(item):
        label = item[1]
        if label == 'Average':
            return (0, label)
        elif label == 'RMSE Illumination':
            return (1, label)
        else:
            return (2, label.lower())
            
    sorted_items = sorted(zip(handles, labels), key=sort_key)
    if sorted_items:
        sorted_handles, sorted_labels = zip(*sorted_items)
        plt.legend(sorted_handles, sorted_labels, loc='best', fontsize=11)
        
    plt.tight_layout()

    # Save to disk
    base_name = os.path.splitext(os.path.basename(json_path))[0]
    out_dir = os.path.dirname(json_path) or '.'
    out_file = os.path.join(out_dir, f"{base_name}_plot.png")
    
    plt.savefig(out_file, dpi=300)
    print(f"Plot saved to '{out_file}'")

    # Display window
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RMSE analysis results from JSON output")
    parser.add_argument('json_file', type=str, help='Path to the JSON results file')
    parser.add_argument('--rename', nargs='+', help='Rename drones, e.g. lb2="LightBender 1 / Top"')
    parser.add_argument('--xlim', type=float, nargs=2, help='X-axis limits (min max)')
    parser.add_argument('--ylim', type=float, nargs=2, help='Y-axis limits (min max)')
    parser.add_argument('--annotate_x', type=float, help='X-axis position of the annotation')
    args = parser.parse_args()

    rename_map = {}
    if args.rename:
        for r in args.rename:
            if '=' in r:
                k, v = r.split('=', 1)
                rename_map[k] = v

    plot_rmse_json(args.json_file, rename_map=rename_map, xlim=args.xlim, ylim=args.ylim)
