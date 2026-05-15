import argparse
import json
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d

# ==========================================
# CONFIGURATION
# ==========================================

# Offset of camera relative to mocap (frame tvec) data for the localizing lb (in drone body frame)
CAMERA_OFFSET_BODY = np.array([0.018, -0.035, -0.025])

# Offset of marker relative to mocap data for the anchor FLS (in drone body frame)
MARKER_OFFSET_BODY = np.array([0.01, 0.035, -0.035])

def tracker_to_body(tracker_tvec):
    """
    Mapping from camera tracker xyz to mocap xyz.
    User specifies: Z->Y, X->-X, Y->-Z
    """
    x, y, z = tracker_tvec
    return np.array([-x, -z, -y])

# ==========================================
# DATA LOADING
# ==========================================

def load_mocap_data(json_path, start_time, stop_time, ignore_rpy=False):
    """
    Loads Mocap GT positions and EKF orientations from JSON.
    Returns:
        times: array of relative times (from start_time)
        pos_fn: interpolation function for position
        rpy_fns: tuple of (roll_fn, pitch_fn, yaw_fn)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    vicon_times = []
    vicon_pos = []
        
    for frame in data['frames']:
        t = frame['time']
        if start_time <= t <= stop_time:
            vicon_times.append(t)
            vicon_pos.append(frame['tvec'])
            
    vicon_times = np.array(vicon_times)
    vicon_pos = np.array(vicon_pos)
    
    rel_times = vicon_times - start_time
    if len(rel_times) == 0:
        return None, None, None

    pos_fn = interp1d(rel_times, vicon_pos, axis=0, kind='linear', fill_value="extrapolate", bounds_error=False)
    
    cf_log_group = data.get('cf', data.get('cf_ATT_RATE'))
    if not cf_log_group or ignore_rpy:
        # Default zero rotation
        r_fn = lambda t: 0.0
        p_fn = lambda t: 0.0
        y_fn = lambda t: 0.0
    else:
        ekf_time = np.array(cf_log_group['time'])
        ekf_rel_times = ekf_time - start_time
        
        roll = np.array(cf_log_group['params']['stateEstimate.roll']['data'])
        pitch = np.array(cf_log_group['params']['stateEstimate.pitch']['data'])
        yaw = np.array(cf_log_group['params']['stateEstimate.yaw']['data'])
        
        r_fn = interp1d(ekf_rel_times, np.radians(roll), fill_value="extrapolate", bounds_error=False)
        p_fn = interp1d(ekf_rel_times, np.radians(pitch), fill_value="extrapolate", bounds_error=False)
        y_fn = interp1d(ekf_rel_times, np.radians(yaw), fill_value="extrapolate", bounds_error=False)
        
    return rel_times, pos_fn, (r_fn, p_fn, y_fn)

def main():
    parser = argparse.ArgumentParser(description="Evaluate accuracy of relative localization using tracker camera.")
    parser.add_argument('log_dir', type=str, help='Path to the log directory (e.g., orchestrator/logs/test_2026-05-13_11-14-09)')
    parser.add_argument('--ignore_rpy', action='store_true', help='Ignore EKF roll, pitch, yaw data for relative calculation')
    args = parser.parse_args()
    
    log_dir = args.log_dir
    
    # 1. Discover Logs & Assign Roles
    all_json_files = glob.glob(os.path.join(log_dir, "*.json"))
    tracker_files = [f for f in all_json_files if "_tracker_" in os.path.basename(f)]
    mocap_files = [f for f in all_json_files if "_tracker_" not in os.path.basename(f) and f.startswith(os.path.join(log_dir, "lb"))]
    
    if not mocap_files:
        print("No lightbender mocap logs found.")
        return
        
    drone_ids = [os.path.basename(f).split('_')[0] for f in mocap_files]
    tracker_ids = [os.path.basename(f).split('_')[0] for f in tracker_files]
    
    localizing_id = None
    anchor_id = None
    
    if len(drone_ids) == 2 and len(tracker_ids) == 1:
        localizing_id = tracker_ids[0]
        anchor_id = [d for d in drone_ids if d != localizing_id][0]
        print(f"Auto-assigned: Localizing={localizing_id}, Anchor={anchor_id}")
    else:
        print("Available drones:")
        for d in drone_ids:
            print(f" - {d} " + ("(Has Tracker)" if d in tracker_ids else ""))
            
        anchor_id = input("Enter Anchor Drone ID: ").strip()
        localizing_id = input("Enter Localizing Drone ID: ").strip()
        
    anchor_json = next((f for f in mocap_files if os.path.basename(f).startswith(anchor_id + "_")), None)
    localizing_json = next((f for f in mocap_files if os.path.basename(f).startswith(localizing_id + "_")), None)
    tracker_json = next((f for f in tracker_files if os.path.basename(f).startswith(localizing_id + "_tracker_")), None)
    
    if not anchor_json or not localizing_json or not tracker_json:
        print("Required log files are missing.")
        return
        
    # 2. Time Range Selection
    with open(localizing_json, 'r') as f:
        loc_data = json.load(f)
        
    if 'start_times' in loc_data and 'stop_times' in loc_data:
        sample_pairs = list(zip(loc_data['start_times'], loc_data['stop_times']))
        if len(sample_pairs) > 1:
            print("\nMultiple time ranges found:")
            for i, (st, et) in enumerate(sample_pairs):
                print(f"  {i}: start={st:.4f}, end={et:.4f}")
            time_range_index = int(input("Select index: "))
        else:
            time_range_index = 0
            
        start_time = sample_pairs[time_range_index][0]
        stop_time = sample_pairs[time_range_index][1]
    else:
        start_time = loc_data['start_time']
        stop_time = loc_data['stop_time']
        
    # 3. Load Data
    print(f"Loading Mocap for {anchor_id}...")
    anchor_times, anchor_pos_fn, anchor_rpy_fns = load_mocap_data(anchor_json, start_time, stop_time, args.ignore_rpy)
    
    print(f"Loading Mocap for {localizing_id}...")
    loc_times, loc_pos_fn, loc_rpy_fns = load_mocap_data(localizing_json, start_time, stop_time, args.ignore_rpy)
    
    if anchor_pos_fn is None or loc_pos_fn is None:
        print("No valid mocap data found in the selected time range.")
        return
        
    print(f"Loading Tracker data for {localizing_id}...")
    with open(tracker_json, 'r') as f:
        track_data = json.load(f)
        
    tracker_times = []
    tracker_vecs = []
    
    # tracker timestamps are in ms (Unix time).
    for frame in track_data['frames']:
        t_sec = frame['time'] / 1000.0
        if start_time <= t_sec <= stop_time:
            # relative time
            tracker_times.append(t_sec - start_time)
            tracker_vecs.append(frame['tvec'])
            
    tracker_times = np.array(tracker_times)
    tracker_vecs = np.array(tracker_vecs)
    
    if len(tracker_times) == 0:
        print("No tracker data found in the selected time range.")
        return
        
    # 4. Compute RMSE
    # We will interpolate GT at tracker times
    errors = []
    valid_times = []
    gt_vecs = []
    act_vecs = []
    
    for i, t_rel in enumerate(tracker_times):
        # Anchor Pose
        a_pos = anchor_pos_fn(t_rel)
        if np.any(np.isnan(a_pos)): continue
        
        a_r, a_p, a_y = anchor_rpy_fns[0](t_rel), anchor_rpy_fns[1](t_rel), anchor_rpy_fns[2](t_rel)
        a_rot = R.from_euler('xyz', [a_r, a_p, a_y], degrees=False)
        
        # Localizing Pose
        l_pos = loc_pos_fn(t_rel)
        if np.any(np.isnan(l_pos)): continue
        
        l_r, l_p, l_y = loc_rpy_fns[0](t_rel), loc_rpy_fns[1](t_rel), loc_rpy_fns[2](t_rel)
        l_rot = R.from_euler('xyz', [l_r, l_p, l_y], degrees=False)
        
        # Positions in World
        marker_world = a_pos + a_rot.apply(MARKER_OFFSET_BODY)
        camera_world = l_pos + l_rot.apply(CAMERA_OFFSET_BODY)
        
        # GT Vector in World Frame
        v_world = marker_world - camera_world
        
        # GT Vector in Camera (Localizing Drone Body) Frame
        # v_body = R_l^-1 * v_world
        v_body_gt = l_rot.inv().apply(v_world)
        
        # Tracker Vector in Drone Body Frame
        v_tracker = tracker_vecs[i]
        v_body_act = tracker_to_body(v_tracker)
        
        diff = v_body_gt - v_body_act
        err = np.linalg.norm(diff)
        errors.append(err * 1000.0) # mm
        valid_times.append(t_rel)
        gt_vecs.append(v_body_gt)
        act_vecs.append(v_body_act)
        
    errors = np.array(errors)
    valid_times = np.array(valid_times)
    gt_vecs = np.array(gt_vecs) * 1000.0 # Convert to mm
    act_vecs = np.array(act_vecs) * 1000.0 # Convert to mm
    
    # 5. Output Results
    mean_error = np.mean(errors)
    true_rmse = np.sqrt(np.mean(errors ** 2))
    max_error = np.max(errors)
    
    print(f"\nResults:")
    print(f"Mean Euclidean Error: {mean_error:.2f} mm")
    print(f"True RMSE: {true_rmse:.2f} mm")
    print(f"Max Error: {max_error:.2f} mm")
    
    # Plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Calculate common range size
    err_min, err_max = np.min(errors), np.max(errors)
    x_min, x_max = min(np.min(gt_vecs[:, 0]), np.min(act_vecs[:, 0])), max(np.max(gt_vecs[:, 0]), np.max(act_vecs[:, 0]))
    y_min, y_max = min(np.min(gt_vecs[:, 1]), np.min(act_vecs[:, 1])), max(np.max(gt_vecs[:, 1]), np.max(act_vecs[:, 1]))
    z_min, z_max = min(np.min(gt_vecs[:, 2]), np.min(act_vecs[:, 2])), max(np.max(gt_vecs[:, 2]), np.max(act_vecs[:, 2]))
    
    max_range = max(err_max - err_min, x_max - x_min, y_max - y_min, z_max - z_min)
    pad = max_range * 0.1 if max_range > 0 else 10.0
    span = max_range + 2 * pad
    
    def set_ylim_centered(ax, d_min, d_max):
        mid = (d_max + d_min) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)
    
    # Error Plot
    axs[0].plot(valid_times, errors, label='Relative Error')
    axs[0].axhline(mean_error, color='r', linestyle='--', label=f'Mean Error: {mean_error:.2f} mm')
    axs[0].axhline(true_rmse, color='g', linestyle='-.', label=f'True RMSE: {true_rmse:.2f} mm')
    axs[0].set_title('Relative Localization Error Over Time')
    axs[0].set_ylabel('Error (mm)')
    axs[0].grid(True)
    axs[0].legend()
    set_ylim_centered(axs[0], err_min, err_max)
    
    # X Plot
    axs[1].plot(valid_times, gt_vecs[:, 0], label='GT X', color='black')
    axs[1].plot(valid_times, act_vecs[:, 0], label='Act X', linestyle='--')
    axs[1].set_ylabel('X (mm)')
    axs[1].grid(True)
    axs[1].legend()
    set_ylim_centered(axs[1], x_min, x_max)
    
    # Y Plot
    axs[2].plot(valid_times, gt_vecs[:, 1], label='GT Y', color='black')
    axs[2].plot(valid_times, act_vecs[:, 1], label='Act Y', linestyle='--')
    axs[2].set_ylabel('Y (mm)')
    axs[2].grid(True)
    axs[2].legend()
    set_ylim_centered(axs[2], y_min, y_max)
    
    # Z Plot
    axs[3].plot(valid_times, gt_vecs[:, 2], label='GT Z', color='black')
    axs[3].plot(valid_times, act_vecs[:, 2], label='Act Z', linestyle='--')
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('Z (mm)')
    axs[3].grid(True)
    axs[3].legend()
    set_ylim_centered(axs[3], z_min, z_max)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
