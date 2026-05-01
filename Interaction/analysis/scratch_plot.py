import json
import math
import time
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Global limits you can change
ACCEL_YLIM = 5000  # mm/s^2 limit for all acceleration plots
FORCE_YLIM = 2.0   # N limit for all force plots
VEL_YLIM = 1500    # mm/s limit for all velocity plots
USE_LOG_SCALE = False  # Set to True to use symlog for accel and vel plots
OUT_DIR = "./results/Shuqin"
LOG_170 = "../../orchestrator/logs/translation_2026-04-28_15-48-57/lb11_translation_2026-04-28_15-48-57.json"
LOG_340 = "../../orchestrator/logs/translation_2026-04-28_15-50-23/lb11_translation_2026-04-28_15-50-23.json"
#
# OUT_DIR = "./results/Shahram"
# LOG_170 = "../../orchestrator/logs/translation_2026-04-29_10-28-12/lb11_translation_2026-04-29_10-28-12.json"
# LOG_340 = "../../orchestrator/logs/translation_2026-04-29_10-29-41/lb11_translation_2026-04-29_10-29-41.json"

def load_records(log_path):
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip().rstrip(',')
            if not line or line in ('[', ']'):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records

def process_log(log_path):
    records = load_records(log_path)
    
    wait_time = next(
        (r['data']['time'] for r in records
         if r.get('type') == 'events' and r.get('name') == 'Waiting For User Interaction'),
        None
    )
    if wait_time is None:
        wait_time = records[0]['data']['time'] if records else 0

    # Extract frames for velocity
    frames = []
    for r in records:
        if r.get('type') == 'frames':
            vel = r['data']['vel']
            frames.append({
                'time': r['data']['time'] - wait_time,
                'vx': vel[0], 'vy': vel[1], 'vz': vel[2]
            })

    # Extract state for FC acceleration, velocity and roll
    fc_accel = []
    fc_vel = []
    fc_roll = []
    for r in records:
        if r.get('type') == 'state':
            data = r['data']
            if r.get('group') == 'POS_ACC':
                ay = data['stateEstimate.ay']
                fc_accel.append({
                    'time': data['time'] - wait_time,
                    'a': ay * 9810.0  # mm/s^2 (converted from Gs)
                })
            elif r.get('group') == 'VEL_ORI':
                vy = data['stateEstimate.vy']
                fc_vel.append({
                    'time': data['time'] - wait_time,
                    'v': vy * 1000.0  # mm/s
                })
                fc_roll.append({
                    'time': data['time'] - wait_time,
                    'roll': data['stateEstimate.roll']
                })
                
    frames_vel = [{'time': f['time'], 'v': f['vy'] * 1000.0} for f in frames]
    # Calculate instant vel-based acceleration
    vel_accel_raw = []
    for i in range(1, len(frames)):
        dt = frames[i]['time'] - frames[i-1]['time']
        if dt > 0:
            dvy = frames[i]['vy'] - frames[i-1]['vy']
            a = (dvy / dt) * 1000.0  # mm/s^2
            vel_accel_raw.append({
                'time': frames[i]['time'],
                'a': a
            })
            
    # Calculate derived FC acceleration from FC velocity
    fc_accel_derived = []
    for i in range(1, len(fc_vel)):
        dt = fc_vel[i]['time'] - fc_vel[i-1]['time']
        if dt > 0:
            dv = fc_vel[i]['v'] - fc_vel[i-1]['v']
            a = dv / dt  # v is already in mm/s, dt in s -> a in mm/s^2
            fc_accel_derived.append({
                'time': fc_vel[i]['time'],
                'a': a
            })

    # Sliding window (100ms = 0.1s)
    # Average and Median for Velocity
    vel_accel_avg = []
    vel_accel_median = []
    
    vel_times = [x['time'] for x in vel_accel_raw]
    vel_accels = [x['a'] for x in vel_accel_raw]
    
    for i in range(len(vel_accel_raw)):
        t = vel_times[i]
        window = [vel_accels[j] for j in range(len(vel_accel_raw)) if t - 0.1 <= vel_times[j] <= t]
        if window:
            vel_accel_avg.append({'time': t, 'a': np.mean(window)})
            vel_accel_median.append({'time': t, 'a': np.median(window)})

    # Average and Median for FC
    fc_accel_avg = []
    fc_accel_median = []

    fc_times = [x['time'] for x in fc_accel]
    fc_accels = [x['a'] for x in fc_accel]

    for i in range(len(fc_accel)):
        t = fc_times[i]
        window = [fc_accels[j] for j in range(len(fc_accel)) if t - 0.1 <= fc_times[j] <= t]
        if window:
            fc_accel_avg.append({'time': t, 'a': np.mean(window)})
            fc_accel_median.append({'time': t, 'a': np.median(window)})

    push_events = []
    disengage_events = []
    for r in records:
        if r.get('type') == 'events':
            if r.get('name') == 'User Pushing':
                push_events.append(r['data']['time'] - wait_time)
            elif r.get('name') == 'User Disengage':
                disengage_events.append(r['data']['time'] - wait_time)

    boundaries = [0] + disengage_events + [float('inf')]
    first_push_events = []
    for i in range(len(boundaries) - 1):
        t_from = boundaries[i]
        t_to = boundaries[i+1]
        session_pushes = [p for p in push_events if t_from <= p <= t_to]
        if session_pushes:
            first_push_events.append(session_pushes[0])

    return frames_vel, fc_vel, vel_accel_raw, vel_accel_avg, vel_accel_median, fc_accel, fc_accel_derived, fc_accel_avg, fc_accel_median, fc_roll, first_push_events, disengage_events

def create_plot(x1, y1, x2, y2, push, disengage, label1, label2, title, filename, y_lim_override=None, use_log=False):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.scatter(x1, y1, alpha=0.5, s=10, label=label1)
    ax.scatter(x2, y2, alpha=0.5, s=10, label=label2)

    for i, p in enumerate(push):
        ax.axvline(x=p, color='gray', linestyle='--', alpha=0.6, label='Push' if i == 0 else "")
    for i, d in enumerate(disengage):
        ax.axvline(x=d, color='gray', linestyle=':', alpha=0.6, label='Disengage' if i == 0 else "")

    if y_lim_override is not None:
        max_abs_y = y_lim_override
    else:
        y1_max = max(abs(v) for v in y1) if y1 else 0
        y2_max = max(abs(v) for v in y2) if y2 else 0
        max_abs_y = max(y1_max, y2_max) * 1.2
        if max_abs_y == 0:
            max_abs_y = 1

    ax.set_ylim(-max_abs_y, max_abs_y)

    if use_log:
        ax.set_yscale('symlog', linthresh=1.0)
        ticks = [0]
        for v in [1, 10, 100, 1000]:
            if v < max_abs_y:
                ticks.extend([v, -v])
        ticks.extend([max_abs_y, -max_abs_y])
        ticks = sorted(list(set(ticks)))
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    
    last_disengage = max(disengage) if disengage else 0
    if last_disengage > 0:
        max_x = last_disengage + 3.0
    else:
        max_x = max(max(x1, default=0), max(x2, default=0)) * 1.2
    
    start_x = 0
    ax.set_xlim([start_x, max_x])

    ax.set_xlabel("Time (s)", fontsize=16)
    ax.set_title(title, loc="left", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', labelsize=16)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    # Save the figure safely creating directories if needed
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=300)
    plt.close(fig)

def generate_plots(output_dir="./results"):
    log_170g = LOG_170
    log_340g = LOG_340
    
    vvel_170, fvel_170, vraw_170, vavg_170, vmed_170, fraw_170, fderiv_170, favg_170, fmed_170, froll_170, p_170, d_170 = process_log(log_170g)
    vvel_340, fvel_340, vraw_340, vavg_340, vmed_340, fraw_340, fderiv_340, favg_340, fmed_340, froll_340, p_340, d_340 = process_log(log_340g)

    # Helper for generating the plots for a given payload
    def generate_payload_plots(vvel, fvel, vraw, fraw, fderiv, vavg, favg, vmed, fmed, froll, push, disengage, mass_label, mass_kg):
        # 1. Velocity
        create_plot(
            [x['time'] for x in fvel], [x['v'] for x in fvel],
            [x['time'] for x in vvel], [x['v'] for x in vvel],
            push, disengage,
            "FC Data", "Vel Data",
            f"{mass_label} - Velocity (mm/s)",
            f"{output_dir}/{mass_label.replace(' ', '')}_velocity.png",
            y_lim_override=VEL_YLIM,
            use_log=USE_LOG_SCALE
        )
        
        # 2. Accel Raw
        create_plot(
            [x['time'] for x in fraw], [x['a'] for x in fraw],
            [x['time'] for x in vraw], [x['a'] for x in vraw],
            push, disengage,
            "FC Data", "Vel Data",
            f"{mass_label} - Acceleration (mm/s^2) - Raw",
            f"{output_dir}/{mass_label.replace(' ', '')}_accel_raw.png",
            y_lim_override=ACCEL_YLIM,
            use_log=USE_LOG_SCALE
        )
        
        # 2b. Sanity Check
        create_plot(
            [x['time'] for x in fraw], [x['a'] for x in fraw],
            [x['time'] for x in fderiv], [x['a'] for x in fderiv],
            push, disengage,
            "FC Reported Accel", "FC Derived Accel",
            f"{mass_label} - Sanity Check: FC Accel vs Derived FC Accel",
            f"{output_dir}/{mass_label.replace(' ', '')}_sanity_check.png",
            y_lim_override=ACCEL_YLIM,
            use_log=USE_LOG_SCALE
        )
        
        # 3. Accel Avg
        create_plot(
            [x['time'] for x in favg], [x['a'] for x in favg],
            [x['time'] for x in vavg], [x['a'] for x in vavg],
            push, disengage,
            "FC Data", "Vel Data",
            f"{mass_label} - Acceleration (mm/s^2) - 100ms Avg",
            f"{output_dir}/{mass_label.replace(' ', '')}_accel_avg.png",
            y_lim_override=ACCEL_YLIM,
            use_log=USE_LOG_SCALE
        )
        
        # 4. Accel Median
        create_plot(
            [x['time'] for x in fmed], [x['a'] for x in fmed],
            [x['time'] for x in vmed], [x['a'] for x in vmed],
            push, disengage,
            "FC Data", "Vel Data",
            f"{mass_label} - Acceleration (mm/s^2) - 100ms Median",
            f"{output_dir}/{mass_label.replace(' ', '')}_accel_median.png",
            y_lim_override=ACCEL_YLIM,
            use_log=USE_LOG_SCALE
        )
        
        # 5. Force (Mass * Avg Accel)
        create_plot(
            [x['time'] for x in favg], [x['a'] * mass_kg / 1000.0 for x in favg],
            [x['time'] for x in vavg], [x['a'] * mass_kg / 1000.0 for x in vavg],
            push, disengage,
            "FC Data", "Vel Data",
            f"{mass_label} - Force (N) - 100ms Avg Accel",
            f"{output_dir}/{mass_label.replace(' ', '')}_force.png",
            y_lim_override=FORCE_YLIM
        )

        # 5b. Roll Angle
        create_plot(
            [x['time'] for x in froll], [x['roll'] for x in froll],
            [], [],
            push, disengage,
            "FC Roll Angle", "",
            f"{mass_label} - Roll Angle (Degrees)",
            f"{output_dir}/{mass_label.replace(' ', '')}_roll.png",
            y_lim_override=30
        )

        # 6. Force from Roll vs Vel Accel Force
        # F_aero_y = - m * g * sin(roll). 
        create_plot(
            [x['time'] for x in froll], [-mass_kg * 9.81 * np.sin(np.radians(x['roll'])) for x in froll],
            [x['time'] for x in vavg], [x['a'] * mass_kg / 1000.0 for x in vavg],
            push, disengage,
            "FC Force (from Roll)", "Vel Force (from Accel)",
            f"{mass_label} - Force Against User (N) - Roll vs Vel Accel",
            f"{output_dir}/{mass_label.replace(' ', '')}_force_roll.png",
            y_lim_override=FORCE_YLIM
        )

    # Generate for 170g
    generate_payload_plots(vvel_170, fvel_170, vraw_170, fraw_170, fderiv_170, vavg_170, favg_170, vmed_170, fmed_170, froll_170, p_170, d_170, "170 g", 0.170)
    
    # Generate for 340g
    generate_payload_plots(vvel_340, fvel_340, vraw_340, fraw_340, fderiv_340, vavg_340, favg_340, vmed_340, fmed_340, froll_340, p_340, d_340, "340 g", 0.340)

if __name__ == '__main__':
    
    generate_plots(output_dir=OUT_DIR)
