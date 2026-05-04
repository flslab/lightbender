import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

def parse_velocity(filepath):
    """
    Parses a JSON log file to extract time and vy.
    Extracts from both 'frames' and 'state' (VEL_ORI) groups.
    Aligns time such that t=0 is the timestamp of the 'start' event.
    """
    frames_times = []
    frames_vxs = []
    frames_vys = []
    frames_ys = []
    state_times = []
    state_vys = []

    with open(filepath, 'r') as f:
        try:
            log_data = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return [], [], [], [], [], []

    # Find the start time
    t0 = None
    for item in log_data:
        if item.get("type") == "start":
            t0 = item.get("data")
            break

    for item in log_data:
        if item.get("type") == "frames":
            d = item.get("data", {})
            vel = d.get("vel", [0, 0, 0])
            tvec = d.get("tvec", [0, 0, 0])
            t = d.get("time")
            if t is not None:
                if t0 is None: t0 = t
                frames_times.append(t - t0)
                frames_vxs.append(vel[0] * 1000)
                frames_vys.append(vel[1] * 1000)
                frames_ys.append(tvec[1])

        elif item.get("type") == "state" and item.get("group") == "VEL_ORI":
            d = item.get("data", {})
            vy = d.get("stateEstimate.vy", 0.0)
            t = d.get("time")
            if t is not None:
                if t0 is None: t0 = t
                state_times.append(t - t0)
                state_vys.append(vy * 1000)

    return frames_times, frames_vxs, frames_vys, frames_ys, state_times, state_vys

def extract_negative_period_pos(times, vys, ys):
    n_times = []
    n_pos = []

    current_pos = 0.0
    accumulating = False

    for i in range(1, len(vys)):
        if vys[i] < 0:
            if not accumulating:
                # Started a new consecutive negative run
                current_pos = 0.0
                accumulating = True
                # Add the 0 position point just before it went negative
                n_times.append(times[i-1])
                n_pos.append(current_pos)

            dy = ys[i] - ys[i-1]

            current_pos += dy
            n_times.append(times[i])
            n_pos.append(current_pos)
        else:
            accumulating = False

    return n_times, n_pos

def plot_velocities(files, output="velocity_plot.png", use_log_scale=False):
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    ax_frames, ax_state, ax_frames_pos = axes

    max_t = 0
    max_v = 0
    min_v = 0

    has_data = False

    for filepath, name in files:
        f_times, f_vxs, f_vys, f_ys, s_times, s_vys = parse_velocity(filepath)

        if not f_times and not s_times:
            print(f"Warning: No velocity data found in {filepath}")
            continue

        if f_times:
            max_t = max(max_t, max(f_times))
            max_v = max(max_v, max(f_vys))
            min_v = min(min_v, min(f_vys))
            ax_frames.scatter(f_times, f_vys, alpha=0.5, s=5, label=name)
            has_data = True

            f_pos_t, f_pos_y = extract_negative_period_pos(f_times, f_vys, f_ys)
            if f_pos_t:
                ax_frames_pos.scatter(f_pos_t, np.array(f_pos_y) * 1000, alpha=0.8, s=8, label=name)

        if s_times:
            max_t = max(max_t, max(s_times))
            max_v = max(max_v, max(s_vys))
            min_v = min(min_v, min(s_vys))
            ax_state.scatter(s_times, s_vys, alpha=0.5, s=5, label=name)
            has_data = True

    if has_data:
        v_margin = max(abs(max_v), abs(min_v)) * 0.2
        if v_margin == 0: v_margin = 0.1

        ylim = [-1800, 1800]
        xlim = [0, 40]

        for ax in [ax_frames, ax_state]:
            ax.set_ylim(ylim)

        for ax in [ax_frames, ax_state, ax_frames_pos]:
            ax.set_xlim(xlim)

        ax_frames_pos.set_ylim([-100, 0])

    ax_frames.set_xlabel("Time (s)", fontsize=16)
    ax_frames.set_title("Y-axis Velocity: Velocity Estimation with Vicon (mm/s)", loc="left", fontsize=16)

    ax_state.set_xlabel("Time (s)", fontsize=16)
    ax_state.set_title("Y-axis Velocity: Flight Controller Estimation (mm/s)", loc="left", fontsize=16)

    ax_frames_pos.set_xlabel("Time (s)", fontsize=16)
    ax_frames_pos.set_title("Y-axis Accumulated Position Change (Vicon) [Vy < 0] (mm)", loc="left", fontsize=16)

    for ax in [ax_frames, ax_state, ax_frames_pos]:
        if ax in [ax_frames, ax_state] and use_log_scale:
            ax.set_yscale('symlog', linthresh=1.0)
            ticks = [-1500, -1000, -100, -10, -1, 0, 1, 10, 100, 1000, 1500]
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(t) for t in ticks])
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='both', labelsize=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=16)

    plt.tight_layout()
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(output, dpi=300)
    print(f"Saved plot to {output}")
    plt.close(fig)

def plot_xy_speed(files, output="xy_speed_plot.png", velocity_threshold=None):
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    has_data = False

    for filepath, name in files:
        f_times, f_vxs, f_vys, f_ys, s_times, s_vys = parse_velocity(filepath)

        if not f_times:
            continue

        xy_speed = np.sqrt(np.array(f_vxs)**2 + np.array(f_vys)**2)
        ax.scatter(f_times, xy_speed, alpha=0.5, s=5, label=name)
        has_data = True

    if velocity_threshold is not None:
        thresh_val, thresh_label = velocity_threshold
        ax.axhline(thresh_val, color='gray', linestyle='--', linewidth=1.5, label=thresh_label)

    if has_data:
        ax.set_xlim([0, 40])
        ax.set_ylim([0, 1800])

    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_title("XY Speed: Vicon Estimation (mm/s)", loc="left", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=16)

    plt.tight_layout()
    output_dir = os.path.dirname(output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(output, dpi=300)
    print(f"Saved plot to {output}")
    plt.close(fig)

if __name__ == '__main__':
    # Load your parameters here:
    target_files = [
        # ["../../orchestrator/logs/translation_2026-04-30_11-34-03/lb11_translation_2026-04-30_11-34-03.json", 'Go To'],
        # ["../../orchestrator/logs/translation_2026-04-30_11-31-44/lb11_translation_2026-04-30_11-31-44.json", 'Pos Setpoint']
        #
        # ["../../orchestrator/logs/translation_2026-04-30_13-29-04/lb11_translation_2026-04-30_13-29-04.json", 'Go To'],
        # ["../../orchestrator/logs/translation_2026-04-30_13-33-54/lb11_translation_2026-04-30_13-33-54.json", 'Pos Setpoint']

        ["../../orchestrator/logs/translation_2026-04-30_14-55-37/lb11_translation_2026-04-30_14-55-37.json", '']
    ]

    output_plot = "../analysis/results/velocity_plot.png"
    output_xy_speed_plot = "../analysis/results/xy_speed_plot.png"

    use_log_scale = False
    velocity_threshold = [130, "S_D=S_Q"]  # [value_mm_s, label], or None to disable

    plot_velocities(target_files, output=output_plot, use_log_scale=use_log_scale)
    plot_xy_speed(target_files, output=output_xy_speed_plot, velocity_threshold=velocity_threshold)
