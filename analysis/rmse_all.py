import matplotlib
matplotlib.use('macosx') # Commented out to prevent errors in non-Mac environments, uncomment if needed.


import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
import glob
import os


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_ROOT = os.path.join(REPO_ROOT, 'orchestrator', 'logs')
MIN_ACTUAL_FRAME_GAP_S = 0.05
ACTUAL_FRAME_GAP_FACTOR = 5.0


def get_time_range_pairs(data):
    """Return logged start/stop pairs, closing an unfinished final range."""
    start_times = data['start_times']
    stop_times = data['stop_times']

    if len(stop_times) == len(start_times) - 1:
        stop_times = [*stop_times, data['frames'][-1]['time']]

    return list(zip(start_times, stop_times))


# ==========================================
# 1. KINEMATICS & GEOMETRY
# ==========================================

def get_led_local_positions(rod_angle_1_deg, rod_angle_2_deg):
    """
    Computes the positions of all 50 LEDs in the DRONE BODY frame.

    Args:
        rod_angle_1_deg: Angle of rod 1 in degrees.
        rod_angle_2_deg: Angle of rod 2 in degrees.

    Returns:
        np.array of shape (50, 3) representing [x, y, z] of each LED.
    """
    leds = []

    # Constants
    SPACING = 0.006  # 6mm

    # --- GEOMETRY DEFINITION ---
    # Constraint 1: Rod rotation axis is X axis (Front).
    # Constraint 2: Rods are perpendicular to rotation axis (in Y-Z plane).
    # Constraint 3: 0 deg = 3 o'clock.
    # Constraint 4: 90 deg = 6 o'clock.
    #
    # Frame Assumption (Standard Drone): X-Forward, Y-Left, Z-Up.
    # View: From Front (+X) looking at the drone (towards origin).
    #
    # Math:
    # y = r * cos(theta)
    # z = -r * sin(theta)

    # --- ROD 1 (LEDs 0 to 25) ---
    # LED 25 is at center (r=0). LED 0 is furthest out.
    theta1_rad = np.radians(rod_angle_1_deg)
    c1 = np.cos(theta1_rad)
    s1 = -np.sin(theta1_rad)  # Note the negative sign for CW rotation logic

    for i in range(26):
        r = (25 - i) * SPACING
        y = r * c1
        z = r * s1
        leds.append([0.0, y, z])

    # --- ROD 2 (LEDs 26 to 49) ---
    # LED 26 is 6mm away (1 unit). LED 49 is furthest.
    theta2_rad = np.radians(rod_angle_2_deg)
    c2 = np.cos(theta2_rad)
    s2 = -np.sin(theta2_rad)

    for i in range(26, 50):
        r = (i - 25) * SPACING
        y = r * c2
        z = r * s2
        leds.append([0.0, y, z])

    return np.array(leds)


def transform_points(points_body, drone_pos, drone_rpy_rad):
    """
    Transforms points from Body Frame to World Frame.
    """
    r = R.from_euler('xyz', drone_rpy_rad, degrees=False)
    points_world = r.apply(points_body) + drone_pos
    return points_world


def marker_position_to_body_origin(
        marker_position_world, marker_position_body, drone_rpy_rad):
    """Recover the drone body origin from an offset mocap marker.

    All positions use metres. Both the world and drone body frames are FLU,
    and ``drone_rpy_rad`` describes the body-to-world rotation.
    """
    body_to_world = R.from_euler('xyz', drone_rpy_rad, degrees=False)
    marker_offset_world = body_to_world.apply(marker_position_body)
    return np.asarray(marker_position_world) - marker_offset_world


def set_axes_equal(ax, points):
    """
    Sets the 3D axes to have equal aspect ratio based on the data bounds.
    """
    x_limits = [np.min(points[:, 0]), np.max(points[:, 0])]
    y_limits = [np.min(points[:, 1]), np.max(points[:, 1])]
    z_limits = [np.min(points[:, 2]), np.max(points[:, 2])]

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max([x_range, y_range, z_range])
    if plot_radius == 0: plot_radius = 1.0

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# ==========================================
# 2. DATA PROCESSING CLASSES
# ==========================================

class DroneProcessor:
    def __init__(
            self, drone_id, yaml_config, json_path=None, act_yaml_config=None,
            use_kinematics=False, max_v=2.0, max_a=1.0, max_j=2.0,
            max_s=10.0, ignore_rpy=False, visualize_only=False,
            time_range_index=None, mocap_marker_position=None):
        self.drone_id = drone_id
        self.yaml_config = yaml_config
        self.json_path = json_path
        self.act_yaml_config = act_yaml_config
        self.use_kinematics = use_kinematics
        self.max_v = max_v
        self.max_a = max_a
        self.max_j = max_j
        self.max_s = max_s
        self.ignore_rpy = ignore_rpy
        self.visualize_only = visualize_only
        self.time_range_index = time_range_index
        self.mocap_marker_position = np.asarray(
            [0.0, 0.0, 0.0]
            if mocap_marker_position is None else mocap_marker_position,
            dtype=float,
        )
        if self.mocap_marker_position.shape != (3,):
            raise ValueError("mocap_marker_position must contain x, y, and z")

        # Load Interpolators
        self._load_gt()
        if self.visualize_only:
            self.start_time = 0.0
            self.stop_time = self.gt_duration
            self.act_max_rel_time = self.gt_duration
        elif self.act_yaml_config:
            self._load_act_from_yaml()
        else:
            self._load_act()

    def _load_gt(self):
        base_waypoints = self.yaml_config.get('waypoints', [])
        if not len(base_waypoints):
            base_waypoints.append(self.yaml_config['target'])
            base_waypoints.append(self.yaml_config['target'])
        elif len(base_waypoints) == 1:
            wp1 = list(base_waypoints[0])
            wp2 = list(base_waypoints[0])
            if len(wp1) == 4:
                wp1.append(0.0)
                wp2.append(5.0)
            else:
                wp2[4] = 5.0
            base_waypoints = [wp1, wp2]

        base_waypoints = np.array(base_waypoints, dtype=float)  # [x, y, z, yaw, dt]
        if 'position_offset' in self.yaml_config:
            offset = np.array(self.yaml_config['position_offset'])
            base_waypoints[:, 0:3] += offset

        base_servos = self.yaml_config.get('servos', [])
        if len(base_servos) == 1:
            base_servos.append(base_servos[0])
        base_servos = np.array(base_servos)  # [rod1, rod2]

        base_pointers = self.yaml_config.get('pointers', [])
        if len(base_pointers) == 1:
            base_pointers.append(base_pointers[0])
        base_pointers = np.array(base_pointers)  # [p0, p1]

        iterations = self.yaml_config.get('iterations', 1)

        waypoints = []
        servos = []
        pointers = []
        times = [0.0]

        for it in range(iterations):
            for i in range(len(base_waypoints)):
                wp = np.copy(base_waypoints[i])
                sv = np.copy(base_servos[i]) if len(base_servos) > i else np.array([0.0, 0.0])
                pt = np.copy(base_pointers[i]) if len(base_pointers) > i else np.array([0.0, 0.0])

                if it == 0 and i == 0:
                    waypoints.append(wp)
                    servos.append(sv)
                    pointers.append(pt)
                else:
                    if i == 0:
                        dt = 0.1 # 100ms gap
                    else:
                        dt = wp[4] if len(wp) == 5 else self.yaml_config.get('delta_t', 0.0)
                    times.append(times[-1] + dt)
                    waypoints.append(wp)
                    servos.append(sv)
                    pointers.append(pt)

        waypoints = np.array(waypoints)
        servos = np.array(servos)
        pointers = np.array(pointers)

        self.gt_times = np.array(times)
        self.gt_duration = times[-1]

        self.gt_pos_fn = interp1d(self.gt_times, waypoints[:, 0:3], axis=0, kind='linear', fill_value="extrapolate")

        if self.use_kinematics:
            self.gt_pos_fn = self._apply_kinematics_filter(self.gt_times, self.gt_pos_fn, self.max_v, self.max_a, self.max_j, self.max_s)

        unwrapped_yaw = np.unwrap(np.radians(waypoints[:, 3]))
        self.gt_yaw_fn = interp1d(self.gt_times, unwrapped_yaw, kind='linear', fill_value="extrapolate")

        # Use raw servo values in degrees
        self.gt_servo_fn = interp1d(self.gt_times, servos, axis=0, kind='linear', fill_value="extrapolate")

        if len(base_pointers) > 0:
            self.gt_pointer_fn = interp1d(self.gt_times, pointers, axis=0, kind='linear', fill_value="extrapolate")
        else:
            self.gt_pointer_fn = None
        self.led_formula = self.yaml_config.get('led', {}).get('formula', None)

    def get_lit_mask(self, t_rel):
        """Returns (50,) boolean array: True for LEDs lit (non-black) at t_rel. None if unavailable."""
        if self.led_formula is None:
            return None
        import math as _math, random as _random

        if self.gt_pointer_fn is None:
            pointers = []
        else:
            pointers = self.gt_pointer_fn(t_rel)
        ctx = {'math': _math, 'random': _random, 'N': 50, 't': t_rel}
        for j, v in enumerate(pointers):
            ctx[f'p{j}'] = float(v)
        mask = np.zeros(50, dtype=bool)
        for idx in range(50):
            ctx['i'] = idx
            rgb = eval(self.led_formula, {}, ctx)
            if any(c > 0 for c in rgb):
                mask[idx] = True
        return mask

    def _apply_kinematics_filter(self, time_vals, pos_func, max_v, max_a, max_j, max_s, dt_sim=0.01):
        sim_times = np.arange(time_vals[0], time_vals[-1] + dt_sim, dt_sim)
        if len(sim_times) == 0:
            return pos_func
        
        pos = np.zeros((len(sim_times), 3))
        vel = np.zeros((len(sim_times), 3))
        acc = np.zeros((len(sim_times), 3))
        jerk = np.zeros((len(sim_times), 3))
        
        pos[0] = pos_func(sim_times[0])
        
        for i in range(1, len(sim_times)):
            target_pos = pos_func(sim_times[i])
            
            # Position error
            error = target_pos - pos[i-1]
            dist = np.linalg.norm(error)
            
            # Safe velocity (considering max_a)
            safe_v = min(max_v, np.sqrt(2 * max_a * max(0, dist)))
            if dist > 1e-6:
                v_des = (error / dist) * min(dist / dt_sim, safe_v)
            else:
                v_des = np.zeros(3)
                
            # Velocity error
            v_err = v_des - vel[i-1]
            dv_norm = np.linalg.norm(v_err)
            
            # Safe acceleration (considering max_j)
            safe_a = min(max_a, np.sqrt(2 * max_j * max(0, dv_norm)))
            if dv_norm > 1e-6:
                a_des = (v_err / dv_norm) * min(dv_norm / dt_sim, safe_a)
            else:
                a_des = np.zeros(3)
                
            # Acceleration error
            a_err = a_des - acc[i-1]
            da_norm = np.linalg.norm(a_err)
            
            # Safe jerk (considering max_s)
            safe_j = min(max_j, np.sqrt(2 * max_s * max(0, da_norm)))
            if da_norm > 1e-6:
                j_des = (a_err / da_norm) * min(da_norm / dt_sim, safe_j)
            else:
                j_des = np.zeros(3)
                
            # Jerk error
            j_err = j_des - jerk[i-1]
            dj_norm = np.linalg.norm(j_err)
            
            # Snap (control input)
            if dj_norm > 1e-6:
                snap = (j_err / dj_norm) * min(dj_norm / dt_sim, max_s)
            else:
                snap = np.zeros(3)
                
            # Step physics
            jerk[i] = jerk[i-1] + snap * dt_sim
            
            j_mag = np.linalg.norm(jerk[i])
            if j_mag > max_j and j_mag > 0:
                jerk[i] = (jerk[i] / j_mag) * max_j
                
            acc[i] = acc[i-1] + jerk[i] * dt_sim
            
            a_mag = np.linalg.norm(acc[i])
            if a_mag > max_a and a_mag > 0:
                acc[i] = (acc[i] / a_mag) * max_a
                
            vel[i] = vel[i-1] + acc[i] * dt_sim
            
            v_mag = np.linalg.norm(vel[i])
            if v_mag > max_v and v_mag > 0:
                vel[i] = (vel[i] / v_mag) * max_v
                
            pos[i] = pos[i-1] + vel[i] * dt_sim
            
        return interp1d(sim_times, pos, axis=0, kind='linear', fill_value='extrapolate')

    def _load_act_from_yaml(self):
        base_waypoints = self.act_yaml_config.get('waypoints', [])
        if not len(base_waypoints):
            base_waypoints.append(self.act_yaml_config['target'])
            base_waypoints.append(self.act_yaml_config['target'])
        elif len(base_waypoints) == 1:
            wp1 = list(base_waypoints[0])
            wp2 = list(base_waypoints[0])
            if len(wp1) == 4:
                wp1.append(0.0)
                wp2.append(5.0)
            else:
                wp2[4] = 5.0
            base_waypoints = [wp1, wp2]

        base_waypoints = np.array(base_waypoints, dtype=float)  # [x, y, z, yaw, dt]
        if 'position_offset' in self.act_yaml_config:
            offset = np.array(self.act_yaml_config['position_offset'])
            base_waypoints[:, 0:3] += offset

        base_servos = self.act_yaml_config.get('servos', [])
        if len(base_servos) == 1:
            base_servos.append(base_servos[0])
        base_servos = np.array(base_servos)  # [rod1, rod2]

        iterations = self.act_yaml_config.get('iterations', 1)

        waypoints = []
        servos = []
        times = [0.0]

        for it in range(iterations):
            for i in range(len(base_waypoints)):
                wp = np.copy(base_waypoints[i])
                sv = np.copy(base_servos[i]) if len(base_servos) > i else np.array([0.0, 0.0])
                
                if it == 0 and i == 0:
                    waypoints.append(wp)
                    servos.append(sv)
                else:
                    if i == 0:
                        dt = 0.1 # 100ms gap
                    else:
                        dt = wp[4] if len(wp) == 5 else self.act_yaml_config.get('delta_t', 0.0)
                    times.append(times[-1] + dt)
                    waypoints.append(wp)
                    servos.append(sv)

        waypoints = np.array(waypoints)
        servos = np.array(servos)

        self.act_times = np.array(times)
        
        self.act_pos_fn = interp1d(self.act_times, waypoints[:, 0:3], axis=0, kind='linear', fill_value="extrapolate")
        
        if self.use_kinematics:
            self.act_pos_fn = self._apply_kinematics_filter(self.act_times, self.act_pos_fn, self.max_v, self.max_a, self.max_j, self.max_s)

        unwrapped_yaw = np.unwrap(np.radians(waypoints[:, 3]))
        self.act_yaw_fn = interp1d(self.act_times, unwrapped_yaw, kind='linear', fill_value="extrapolate")
        
        self.act_r_fn = lambda t: 0.0
        self.act_p_fn = lambda t: 0.0
        self.act_y_fn = lambda t: self.act_yaw_fn(t)

        self.act_servo_fn = interp1d(self.act_times, servos, axis=0, kind='linear', fill_value="extrapolate")
        
        # We need start_time and stop_time for the loop bounds
        self.start_time = 0.0
        self.stop_time = self.act_times[-1]
        self.act_max_rel_time = self.act_times[-1]

    def _load_act(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)

        if 'start_times' in data and 'stop_times' in data:
            pairs = get_time_range_pairs(data)
            self.start_time = pairs[self.time_range_index][0]
            self.stop_time = pairs[self.time_range_index][1]
        else:
            self.start_time = data['start_time']
            self.stop_time = data['stop_time']

        vicon_times = []
        vicon_pos = []

        if 'viewpoint_offsets' in data and len(data['viewpoint_offsets']) > 0:
            vp_data = data['viewpoint_offsets']
            if isinstance(vp_data[0], list) and 'start_times' in data:
                self.viewpoint_offset = np.array(vp_data[self.time_range_index])
            else:
                self.viewpoint_offset = np.array(vp_data)
        else:
            self.viewpoint_offset = np.array([0.0, 0.0, 0.0])

        if 'reference_offsets' in data and len(data['reference_offsets']) > 0:
            vp_data = data['reference_offsets']
            if isinstance(vp_data[0], list) and 'start_times' in data:
                self.viewpoint_offset += np.array(vp_data[self.time_range_index])
            else:
                self.viewpoint_offset += np.array(vp_data)


        for frame in data['frames']:
            t = frame['time']
            if self.start_time <= t <= self.stop_time:
                vicon_times.append(t)
                vicon_pos.append((np.array(frame['tvec']) - self.viewpoint_offset).tolist())

        vicon_times = np.array(vicon_times)
        vicon_pos = np.array(vicon_pos)

        # Convert Vicon to interpolator for easy synchronization relative to start_time
        rel_times = vicon_times - self.start_time
        frame_intervals = np.diff(rel_times)
        positive_intervals = frame_intervals[frame_intervals > 0.0]
        nominal_frame_interval = (
            float(np.median(positive_intervals))
            if len(positive_intervals) > 0 else MIN_ACTUAL_FRAME_GAP_S
        )
        self.max_actual_frame_gap = max(
            MIN_ACTUAL_FRAME_GAP_S,
            ACTUAL_FRAME_GAP_FACTOR * nominal_frame_interval,
        )
        self.actual_frame_times = rel_times
        missing_frame_gaps = frame_intervals > self.max_actual_frame_gap
        if np.any(missing_frame_gaps):
            print(
                f"WARNING: {self.drone_id}: excluding "
                f"{np.count_nonzero(missing_frame_gaps)} actual-frame gap(s); "
                f"longest gap is {np.max(frame_intervals[missing_frame_gaps]):.3f}s."
            )
        self.act_pos_fn = interp1d(rel_times, vicon_pos, axis=0, kind='linear', fill_value="extrapolate",
                                   bounds_error=False)

        cf_log_group = data.get('cf', data.get('cf_ATT_RATE'))
        if not cf_log_group:
            raise Exception("No 'cf' log group found in data.")

        ekf_time = np.array(cf_log_group['time'])
        ekf_rel_times = ekf_time - self.start_time

        roll = np.array(cf_log_group['params']['stateEstimate.roll']['data'])
        pitch = np.array(cf_log_group['params']['stateEstimate.pitch']['data'])
        yaw = np.array(cf_log_group['params']['stateEstimate.yaw']['data'])

        self.act_r_fn = interp1d(ekf_rel_times, np.radians(roll), fill_value="extrapolate")
        self.act_p_fn = interp1d(ekf_rel_times, np.radians(pitch), fill_value="extrapolate")
        # The estimator reports yaw wrapped to [-180, 180]. Interpolating the
        # wrapped samples makes a +180 -> -180 crossing pass through zero,
        # which rotates an offset mocap marker to the wrong side of the body
        # for samples between the two attitude log entries.
        unwrapped_yaw = np.unwrap(np.radians(yaw))
        self.act_y_fn = interp1d(
            ekf_rel_times, unwrapped_yaw, fill_value="extrapolate"
        )

        self.act_max_rel_time = rel_times[-1]

    def _has_actual_frame_coverage(self, t_rel):
        """Return whether actual frames safely bracket this relative time."""
        index = np.searchsorted(self.actual_frame_times, t_rel)
        if (
                index < len(self.actual_frame_times)
                and np.isclose(
                    self.actual_frame_times[index], t_rel, rtol=0.0, atol=1e-9
                )):
            return True
        if index == 0 or index == len(self.actual_frame_times):
            return False
        return bool(
            self.actual_frame_times[index] - self.actual_frame_times[index - 1]
            <= self.max_actual_frame_gap
        )

    def get_positions_at_relative_time(self, t_rel):
        """Returns (GT_Position, Actual_Position, Valid_Bool)."""
        if t_rel < 0 or t_rel > self.gt_duration or t_rel > self.act_max_rel_time:
            return None, None, False

        gt_pos = np.asarray(self.gt_pos_fn(t_rel), dtype=float)
        if not np.all(np.isfinite(gt_pos)):
            return None, None, False

        if self.visualize_only:
            return gt_pos, None, True

        if not self.act_yaml_config and not self._has_actual_frame_coverage(t_rel):
            return None, None, False

        act_pos = np.asarray(self.act_pos_fn(t_rel), dtype=float)
        if not np.all(np.isfinite(act_pos)):
            return None, None, False

        # A mocap tvec locates the tracked marker. Convert it to the body
        # origin by rotating the body-frame marker position into world FLU.
        # This applies only to logs; YAML trajectories already describe the
        # drone body origin.
        if (
                not self.act_yaml_config
                and np.any(self.mocap_marker_position != 0.0)):
            actual_rpy = np.asarray([
                self.act_r_fn(t_rel),
                self.act_p_fn(t_rel),
                self.act_y_fn(t_rel),
            ], dtype=float)
            if not np.all(np.isfinite(actual_rpy)):
                return None, None, False
            act_pos = marker_position_to_body_origin(
                act_pos, self.mocap_marker_position, actual_rpy
            )

        return gt_pos, act_pos, True

    def get_state_at_relative_time(self, t_rel):
        """
        Returns (GT_LEDs, Act_LEDs, Valid_Bool)
        t_rel: Time in seconds relative to the mission start (yaml t=0, json t=start_time)
        """
        g_pos, a_pos, valid = self.get_positions_at_relative_time(t_rel)
        if not valid:
            return None, None, False

        # --- Ground Truth State ---
        g_yaw = self.gt_yaw_fn(t_rel)  # radians
        g_servos = self.gt_servo_fn(t_rel)  # degrees
        g_rpy = [0.0, 0.0, float(g_yaw) * 180 / np.pi]

        # print(g_rpy)
        
        if self.visualize_only:
            leds_local_gt = get_led_local_positions(g_servos[0], g_servos[1])
            leds_gt_world = transform_points(leds_local_gt, g_pos, g_rpy)
            return leds_gt_world, None, True

        # --- Actual State ---
        if self.act_yaml_config:
            if self.ignore_rpy:
                a_rpy = [0.0, 0.0, 0.0]
            else:
                a_rpy = [0.0, 0.0, float(self.act_yaw_fn(t_rel))]
            a_servos = self.act_servo_fn(t_rel)
            
            leds_local_act = get_led_local_positions(a_servos[0], a_servos[1])
            leds_act_world = transform_points(leds_local_act, a_pos, a_rpy)
            
            leds_local_gt = get_led_local_positions(g_servos[0], g_servos[1])
            leds_gt_world = transform_points(leds_local_gt, g_pos, g_rpy)
            
            return leds_gt_world, leds_act_world, True
        else:
            if self.ignore_rpy:
                a_rpy = [0.0, 0.0, 0.0]
            else:
                a_rpy = [self.act_r_fn(t_rel), self.act_p_fn(t_rel), self.act_y_fn(t_rel)]

            # --- LED Computation ---
            # Note: Actual uses GT servo angles as per prompt requirements
            leds_local = get_led_local_positions(g_servos[0], g_servos[1])

            leds_gt_world = transform_points(leds_local, g_pos, g_rpy)
            leds_act_world = transform_points(leds_local, a_pos, a_rpy)

            return leds_gt_world, leds_act_world, True


# ==========================================
# 3. MAIN ANALYSIS LOGIC
# ==========================================

def resolve_yaml_file(yaml_file, tag):
    """Resolve the mission YAML explicitly or from a run's tag directory."""
    if yaml_file:
        yaml_path = os.path.abspath(os.path.expanduser(yaml_file))
        if not os.path.isfile(yaml_path):
            raise FileNotFoundError(f"YAML file not found: {yaml_path}")
        return yaml_path

    if not tag:
        raise ValueError("Provide either --tag or --yaml.")

    tag_directories = []
    for root, directory_names, _ in os.walk(LOGS_ROOT):
        if tag in directory_names:
            tag_directories.append(os.path.join(root, tag))

    if not tag_directories:
        raise FileNotFoundError(
            f"No log directory found for tag '{tag}' under {LOGS_ROOT}"
        )
    if len(tag_directories) > 1:
        locations = ', '.join(sorted(tag_directories))
        raise ValueError(
            f"Multiple log directories found for tag '{tag}': {locations}. "
            "Provide --yaml explicitly."
        )

    tag_directory = tag_directories[0]
    yaml_files = sorted(
        os.path.join(tag_directory, filename)
        for filename in os.listdir(tag_directory)
        if filename.lower().endswith(('.yaml', '.yml'))
        and os.path.isfile(os.path.join(tag_directory, filename))
    )

    if not yaml_files:
        raise FileNotFoundError(
            f"No YAML file found in the tag directory: {tag_directory}"
        )
    if len(yaml_files) > 1:
        filenames = ', '.join(os.path.basename(path) for path in yaml_files)
        raise ValueError(
            f"Multiple YAML files found in {tag_directory}: {filenames}. "
            "Provide --yaml explicitly."
        )

    return yaml_files[0]


def get_position_mean_alignment(processors, timestamps):
    """Return the constant translation that aligns the GT and actual means."""
    gt_positions = []
    act_positions = []

    for t in timestamps:
        for p in processors:
            gt_pos, act_pos, valid = p.get_positions_at_relative_time(t)
            if valid and act_pos is not None:
                gt_positions.append(gt_pos)
                act_positions.append(act_pos)

    if not gt_positions:
        return np.zeros(3), None, None

    gt_mean = np.mean(gt_positions, axis=0)
    act_mean = np.mean(act_positions, axis=0)
    return act_mean - gt_mean, gt_mean, act_mean


def get_trajectory_positions(processor, timestamps, gt_translation=None):
    """Return matching valid GT and actual position samples for one drone."""
    if gt_translation is None:
        gt_translation = np.zeros(3)

    gt_positions = []
    act_positions = []
    for t in timestamps:
        gt_pos, act_pos, valid = processor.get_positions_at_relative_time(t)
        if valid and act_pos is not None:
            gt_positions.append(gt_pos + gt_translation)
            act_positions.append(act_pos)

    if not gt_positions:
        return np.empty((0, 3)), np.empty((0, 3))

    return np.asarray(gt_positions), np.asarray(act_positions)


def calculate_trajectory_rmse(processors, timestamps, gt_translation=None):
    """Calculate symmetric nearest-neighbor position RMSE without time pairing."""
    metrics = {'combined_rmse_mm': None, 'drones': {}}
    combined_sse = 0.0
    combined_count = 0

    for p in processors:
        gt_positions, act_positions = get_trajectory_positions(
            p, timestamps, gt_translation
        )

        if len(gt_positions) == 0:
            metrics['drones'][p.drone_id] = {
                'rmse_mm': None,
                'gt_samples': 0,
                'actual_samples': 0,
            }
            continue

        gt_to_act = cKDTree(act_positions).query(gt_positions)[0]
        act_to_gt = cKDTree(gt_positions).query(act_positions)[0]
        sse = np.sum(gt_to_act ** 2) + np.sum(act_to_gt ** 2)
        count = len(gt_to_act) + len(act_to_gt)
        rmse_mm = np.sqrt(sse / count) * 1000.0

        metrics['drones'][p.drone_id] = {
            'rmse_mm': float(rmse_mm),
            'gt_samples': len(gt_positions),
            'actual_samples': len(act_positions),
        }
        combined_sse += sse
        combined_count += count

    if combined_count > 0:
        metrics['combined_rmse_mm'] = float(
            np.sqrt(combined_sse / combined_count) * 1000.0
        )

    return metrics


def calculate_segmented_trajectory_rmse(
        processors, timestamps, segment_duration, analysis_start, analysis_end,
        gt_translation=None, accumulative=False):
    """Calculate trajectory RMSE in fixed or accumulative time segments."""
    if segment_duration <= 0.0:
        raise ValueError("trajectory RMSE segment duration must be positive")

    result = {
        'segment_duration_s': float(segment_duration),
        'accumulative': bool(accumulative),
        'segments': [],
    }
    segment_start = analysis_start
    while segment_start < analysis_end:
        segment_end = min(segment_start + segment_duration, analysis_end)
        interval_start = analysis_start if accumulative else segment_start
        first_index = np.searchsorted(timestamps, interval_start, side='left')
        last_index = np.searchsorted(timestamps, segment_end, side='left')
        segment_timestamps = timestamps[first_index:last_index]
        metrics = calculate_trajectory_rmse(
            processors, segment_timestamps, gt_translation
        )
        result['segments'].append({
            'start_time_s': float(interval_start),
            'end_time_s': float(segment_end),
            'time_s': float(
                segment_end if accumulative
                else (segment_start + segment_end) / 2.0
            ),
            **metrics,
        })
        segment_start = segment_end

    return result


def calculate_rmse(
        yaml_file, tag, compare_yaml=None, use_kinematics=False, max_v=2.0,
        max_a=1.0, max_j=2.0, max_s=10.0, ignore_rpy=False,
        lit_only=False, trim_start=0.0, trim_end=0.0,
        align_position_means=False, position_only=False,
        trajectory_rmse=False, trajectory_rmse_segment_duration=None,
        trajectory_rmse_accumulative=False, mocap_marker_position=None):
    if lit_only and position_only:
        raise ValueError("lit_only and position_only cannot be enabled together")
    if (
            trajectory_rmse_segment_duration is not None
            and trajectory_rmse_segment_duration <= 0.0):
        raise ValueError("trajectory RMSE segment duration must be positive")
    if (
            trajectory_rmse_accumulative
            and trajectory_rmse_segment_duration is None):
        raise ValueError(
            "accumulative trajectory RMSE requires a segment duration"
        )
    mocap_marker_position = np.asarray(
        [0.0, 0.0, 0.0]
        if mocap_marker_position is None else mocap_marker_position,
        dtype=float,
    )
    if mocap_marker_position.shape != (3,):
        raise ValueError("mocap_marker_position must contain x, y, and z")

    print(f"Loading Configuration from {yaml_file}...")
    with open(yaml_file, 'r') as f:
        yaml_data = yaml.safe_load(f)

    drones_config = yaml_data.get('drones', {})
    processors = []
    
    visualize_only = (tag is None and compare_yaml is None)

    if compare_yaml:
        print(f"Loading Comparison Configuration from {compare_yaml}...")
        with open(compare_yaml, 'r') as f:
            compare_yaml_data = yaml.safe_load(f)
        compare_drones = compare_yaml_data.get('drones', {})
        tag = compare_yaml.split("/")[-1].split(".")[0] # Override output tag

    # 1. Find all JSON log files and check for start_times/stop_times lists
    time_range_index = None
    if not visualize_only and not compare_yaml:
        sample_pairs = None
        for drone_id in drones_config:
            search_pattern = f"/Users/hamed/Documents/Holodeck/lightbender/orchestrator/logs/*/{drone_id}_{tag}.json"
            files = glob.glob(search_pattern) or glob.glob(f"{drone_id}_{tag}.json")
            if files:
                with open(files[0], 'r') as f:
                    d = json.load(f)
                if 'start_times' in d and 'stop_times' in d:
                    sample_pairs = get_time_range_pairs(d)
                    break
        if sample_pairs is not None:
            print("\nMultiple time ranges found in log files:")
            for i, (st, et) in enumerate(sample_pairs):
                print(f"  {i}: start={st:.4f}, end={et:.4f}")
            time_range_index = int(input("Select index: "))

    # 2. Initialize Processors (Find logs and parse)
    for drone_id, config in drones_config.items():
        if visualize_only:
            try:
                p = DroneProcessor(drone_id, config, use_kinematics=use_kinematics, max_v=max_v, max_a=max_a, max_j=max_j, max_s=max_s, ignore_rpy=ignore_rpy, visualize_only=True, mocap_marker_position=mocap_marker_position)
                processors.append(p)
            except Exception as e:
                print(f"Error loading data for {drone_id}: {e}")
            continue

        if compare_yaml:
            act_config = compare_drones.get(drone_id)
            if not act_config:
                print(f"WARNING: Drone '{drone_id}' not found in {compare_yaml}. Skipping.")
                continue
            try:
                p = DroneProcessor(drone_id, config, json_path=None, act_yaml_config=act_config, use_kinematics=use_kinematics, max_v=max_v, max_a=max_a, max_j=max_j, max_s=max_s, ignore_rpy=ignore_rpy, mocap_marker_position=mocap_marker_position)
                processors.append(p)
            except Exception as e:
                print(f"Error loading data for {drone_id}: {e}")
            continue

        # Search for log file starting with drone_id
        search_pattern = f"/Users/hamed/Documents/Holodeck/lightbender/orchestrator/logs/*/{drone_id}_{tag}.json"
        files = glob.glob(search_pattern)

        if not files:
            # Fallback to current directory for user testing if logs/ doesn't exist or is empty
            search_pattern_local = f"{drone_id}_{tag}*.json"
            files = glob.glob(search_pattern_local)

        if not files:
            print(f"WARNING: No log file found for drone '{drone_id}' (Pattern: {search_pattern}). Skipping.")
            continue

        # Pick the first match (assuming one log per drone in directory)
        json_file = files[0]
        print(f"Found log for {drone_id}: {json_file}")

        try:
            p = DroneProcessor(drone_id, config, json_file, compare_yaml, use_kinematics, max_v, max_a, max_j, max_s, ignore_rpy, time_range_index=time_range_index, mocap_marker_position=mocap_marker_position)
            processors.append(p)
        except Exception as e:
            print(f"Error loading data for {drone_id}: {e}")

    if not processors:
        print("No valid drone data found.")
        return

    # 2. Define Master Timeline
    # We want to cover the extent of the longest GT plan
    max_duration = max([p.gt_duration for p in processors])

    if trim_end > 0.0:
        max_duration = min(max_duration, trim_end)

    # Sampling rate for analysis (100Hz)
    dt_analysis = 0.01
    timestamps = np.arange(trim_start, max_duration, dt_analysis)

    position_alignment = np.zeros(3)
    gt_position_mean = None
    act_position_mean = None
    if align_position_means:
        if visualize_only:
            print("WARNING: --align-position-means requires actual data; alignment is disabled in YAML-only visualization mode.")
        else:
            position_alignment, gt_position_mean, act_position_mean = get_position_mean_alignment(processors, timestamps)
            if gt_position_mean is None:
                print("WARNING: No matching finite GT and actual positions were found; position-mean alignment was not applied.")
            else:
                print(
                    "Aligning mean GT position "
                    f"{np.round(gt_position_mean, 5).tolist()} with mean actual position "
                    f"{np.round(act_position_mean, 5).tolist()}; translating GT by "
                    f"{np.round(position_alignment, 5).tolist()} m."
                )

    results = {
        'timestamps': timestamps,
        'combined_rmse': [],
        'drones': {p.drone_id: {'rmse': [], 'leds_gt': [], 'leds_act': []} for p in processors}
    }

    print(f"Analyzing {len(processors)} drone(s) over {max_duration:.2f}s...")

    # 3. Time Loop
    for t in timestamps:
        frame_total_sse = 0
        frame_total_count = 0

        # Iterate over all drones for this specific time step
        for p in processors:
            if position_only:
                gt_pos, act_pos, valid = p.get_positions_at_relative_time(t)
                gt_leds = None if gt_pos is None else np.atleast_2d(gt_pos)
                act_leds = None if act_pos is None else np.atleast_2d(act_pos)
            else:
                gt_leds, act_leds, valid = p.get_state_at_relative_time(t)

            if valid:
                gt_leds = gt_leds + position_alignment

                if position_only:
                    gt_leds_f, act_leds_f, count = gt_leds, act_leds, 1
                elif lit_only:
                    mask = p.get_lit_mask(t)

                    if mask is not None:
                        gt_leds_f = gt_leds[mask]
                        act_leds_f = act_leds[mask] if act_leds is not None else None
                        count = int(mask.sum())
                    else:
                        gt_leds_f, act_leds_f, count = gt_leds, act_leds, len(gt_leds)
                else:
                    gt_leds_f, act_leds_f, count = gt_leds, act_leds, len(gt_leds)

                if act_leds_f is not None:
                    diff = gt_leds_f - act_leds_f
                    sse = np.sum(diff ** 2)

                    # Per drone metrics
                    rmse_val = np.sqrt(sse / count) * 1000.0  # mm

                    results['drones'][p.drone_id]['rmse'].append(rmse_val)

                    # Accumulate for Combined Metric
                    frame_total_sse += sse
                    frame_total_count += count
                else:
                    results['drones'][p.drone_id]['rmse'].append(np.nan)

                # Store subsample for vis (every 10th step)
                if len(results['drones'][p.drone_id]['rmse']) % 10 == 0:
                    results['drones'][p.drone_id]['leds_gt'].append(gt_leds_f)
                    results['drones'][p.drone_id]['leds_act'].append(act_leds_f)
            else:
                # Store NaN to keep array length consistent with timestamps
                results['drones'][p.drone_id]['rmse'].append(np.nan)

                if len(results['drones'][p.drone_id]['rmse']) % 10 == 0:
                    results['drones'][p.drone_id]['leds_gt'].append(None)
                    results['drones'][p.drone_id]['leds_act'].append(None)

        # Compute Combined RMSE for this frame (if any drone was active)
        if frame_total_count > 0:
            comb_rmse = np.sqrt(frame_total_sse / frame_total_count) * 1000.0  # mm
            results['combined_rmse'].append(comb_rmse)
        else:
            results['combined_rmse'].append(np.nan)

    # 4. Compute and Print Overall Stats
    print("\n=== RESULTS ===")

    comb_arr = np.array(results['combined_rmse'])
    valid_comb = ~np.isnan(comb_arr)

    if np.any(valid_comb):
        # Overall RMSE across entire trajectory
        overall_comb_rmse = np.sqrt(np.mean(comb_arr[valid_comb] ** 2))
        max_comb_rmse = np.nanmax(comb_arr)
        print(f"COMBINED (All Drones): Overall RMSE {overall_comb_rmse:.2f} mm")
        print(f"COMBINED (All Drones): Max RMSE {max_comb_rmse:.2f} mm")
    else:
        print("COMBINED: No flight data provided.")

    for p in processors:
        d_rmse = np.array(results['drones'][p.drone_id]['rmse'])
        valid = ~np.isnan(d_rmse)
        if np.any(valid):
            # This is 'Average of RMSEs' which is slightly different from 'RMSE of all points',
            # but standard for reporting time-series performance.
            max_d = np.nanmax(d_rmse)
            mean_d = np.nanmean(d_rmse)
            print(f"Drone {p.drone_id}: Max RMSE {max_d:.2f} mm, Mean RMSE {mean_d:.2f} mm")

    trajectory_metrics = None
    if trajectory_rmse:
        trajectory_metrics = calculate_trajectory_rmse(
            processors, timestamps, position_alignment
        )
        print("\n=== TIME-INDEPENDENT TRAJECTORY RMSE ===")
        combined_trajectory_rmse = trajectory_metrics['combined_rmse_mm']
        if combined_trajectory_rmse is None:
            print("COMBINED: No flight data provided.")
        else:
            print(
                "COMBINED (All Drones): Trajectory RMSE "
                f"{combined_trajectory_rmse:.2f} mm"
            )
        for p in processors:
            drone_trajectory_rmse = trajectory_metrics['drones'][p.drone_id]['rmse_mm']
            if drone_trajectory_rmse is not None:
                print(
                    f"Drone {p.drone_id}: Trajectory RMSE "
                    f"{drone_trajectory_rmse:.2f} mm"
                )

    segmented_trajectory_metrics = None
    if trajectory_rmse_segment_duration is not None:
        segmented_trajectory_metrics = calculate_segmented_trajectory_rmse(
            processors,
            timestamps,
            trajectory_rmse_segment_duration,
            trim_start,
            max_duration,
            position_alignment,
            trajectory_rmse_accumulative,
        )
        segments = segmented_trajectory_metrics['segments']
        valid_segments = sum(
            segment['combined_rmse_mm'] is not None for segment in segments
        )
        print(
            "\n=== TRAJECTORY RMSE OVER TIME ===\n"
            f"Computed {valid_segments}/{len(segments)} valid "
            f"{trajectory_rmse_segment_duration:g}s "
            f"{'accumulative interval(s)' if trajectory_rmse_accumulative else 'segment(s)'}."
        )

    if not visualize_only:
        # ==========================================
        # 5. EXPORT DATA TO JSON
        # ==========================================
        if position_only:
            output_filename = f"{tag}_position_rmse.json"
        elif lit_only:
            output_filename = f"{tag}_absolute_rmse_lit_only.json"
        else:
            output_filename = f"{tag}_absolute_rmse.json"
        if align_position_means:
            output_filename = output_filename.replace(".json", "_position_aligned.json")
        print(f"\nExporting raw data to {output_filename}...")

        # Structure data for export (handle numpy types)
        export_data = {
            "timestamps": results['timestamps'].tolist(),
            "combined_rmse_mm": [None if np.isnan(x) else float(x) for x in results['combined_rmse']],
            "metric": "drone_position" if position_only else "led_position",
            "mocap_marker_position_body_m": mocap_marker_position.tolist(),
            "time_independent_trajectory": trajectory_metrics,
            "trajectory_rmse_over_time": segmented_trajectory_metrics,
            "position_mean_alignment": {
                "enabled": bool(align_position_means and gt_position_mean is not None),
                "gt_mean_m": None if gt_position_mean is None else gt_position_mean.tolist(),
                "actual_mean_m": None if act_position_mean is None else act_position_mean.tolist(),
                "gt_translation_m": position_alignment.tolist()
            },
            "drones": {}
        }

        for p in processors:
            d_data = results['drones'][p.drone_id]

            # Helper to clean numpy point arrays for JSON output
            def clean_points(point_list):
                cleaned = []
                for item in point_list:
                    if item is None:
                        cleaned.append(None)
                    else:
                        # Rounding to save space, remove round() if max precision needed
                        cleaned.append(np.round(item, 5).tolist())
                return cleaned

            drone_export = {
                "rmse_mm": [None if np.isnan(x) else float(x) for x in d_data['rmse']]
            }
            if position_only:
                drone_export["subsampled_gt_positions"] = clean_points(d_data['leds_gt'])
                drone_export["subsampled_actual_positions"] = clean_points(d_data['leds_act'])
            else:
                drone_export["subsampled_gt_leds"] = clean_points(d_data['leds_gt'])
                drone_export["subsampled_act_leds"] = clean_points(d_data['leds_act'])
            export_data["drones"][p.drone_id] = drone_export

        with open(output_filename, 'w') as f:
            json.dump(export_data, f, indent=4)
        print("Export complete.")

    # ==========================================
    # 5. VISUALIZATION
    # ==========================================

    fig = plt.figure(figsize=(18, 8))
    import matplotlib.gridspec as gridspec
    
    N = len(processors) if len(processors) > 0 else 1
    gs = gridspec.GridSpec(N, 2, figure=fig)
    
    left_texts = {}
    time_lines = {}
    colors = plt.cm.jet(np.linspace(0, 1, len(processors)))

    if not visualize_only:
        ax1 = fig.add_subplot(gs[:, 0])

        if segmented_trajectory_metrics is not None:
            segments = segmented_trajectory_metrics['segments']
            segment_times = [segment['time_s'] for segment in segments]
            combined_segment_rmse = [
                np.nan if segment['combined_rmse_mm'] is None
                else segment['combined_rmse_mm']
                for segment in segments
            ]
            ax1.plot(
                segment_times,
                combined_segment_rmse,
                'ko-',
                linewidth=3,
                alpha=0.8,
                label='All Drones (Combined)',
            )
            for i, p in enumerate(processors):
                drone_segment_rmse = [
                    np.nan if segment['drones'][p.drone_id]['rmse_mm'] is None
                    else segment['drones'][p.drone_id]['rmse_mm']
                    for segment in segments
                ]
                ax1.plot(
                    segment_times,
                    drone_segment_rmse,
                    marker='o',
                    color=colors[i],
                    linewidth=1.5,
                    label=p.drone_id,
                )
            duration = segmented_trajectory_metrics['segment_duration_s']
            interval_label = (
                'accumulative intervals'
                if segmented_trajectory_metrics['accumulative'] else 'segments'
            )
            ax1.set_title(
                f'Trajectory RMSE over Time ({duration:g}s {interval_label})'
            )
        else:
            ax1.plot(timestamps, results['combined_rmse'], 'k-', linewidth=3, alpha=0.8, label='All Drones (Combined)')
            if np.any(valid_comb):
                ax1.axhline(overall_comb_rmse, color='r', linestyle='--', label=f'Overall: {overall_comb_rmse:.3f}mm')

            for i, p in enumerate(processors):
                rmse_data = results['drones'][p.drone_id]['rmse']
                ax1.plot(timestamps, rmse_data, color=colors[i], linewidth=1, label=f'{p.drone_id}')

            metric_label = 'Drone Position' if position_only else 'LED Position'
            ax1.set_title(f'{metric_label} RMSE over Time (mm)')

        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Error (mm)')

        if trajectory_metrics is not None:
            trajectory_value = trajectory_metrics['combined_rmse_mm']
            if trajectory_value is not None:
                ax1.axhline(
                    trajectory_value,
                    color='purple',
                    linestyle=':',
                    linewidth=2.5,
                    label=f'Trajectory RMSE: {trajectory_value:.2f} mm',
                )
                ax1.annotate(
                    f'Trajectory RMSE: {trajectory_value:.2f} mm',
                    xy=(1.0, trajectory_value),
                    xycoords=ax1.get_yaxis_transform(),
                    xytext=(-8, 5),
                    textcoords='offset points',
                    ha='right',
                    va='bottom',
                    color='purple',
                    fontsize=9,
                )

        ax1.legend()
        ax1.grid(True)
    else:
        for i, p in enumerate(processors):
            ax = fig.add_subplot(gs[i, 0])
            pos_data = p.gt_pos_fn(timestamps)
            
            ax.plot(timestamps, pos_data[:, 0], color='r', label='X')
            ax.plot(timestamps, pos_data[:, 1], color='g', label='Y')
            ax.plot(timestamps, pos_data[:, 2], color='b', label='Z')
            
            ax.set_ylabel(f'{p.drone_id} (m)')
            if i == N - 1:
                ax.set_xlabel('Time (s)')
            if i == 0:
                ax.set_title('Drone GT Positions (X, Y, Z)')
                ax.legend(loc='upper right', fontsize=8)
                
            ax.grid(True, linestyle=':', alpha=0.6)
            txt = ax.text(0.02, 0.90, '', transform=ax.transAxes, verticalalignment='top', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
            left_texts[p.drone_id] = txt
            
            vl = ax.axvline(0, color='k', linestyle='--', alpha=0.5)
            time_lines[p.drone_id] = vl

    # Plot 2: 3D Animation
    ax_3d = fig.add_subplot(gs[:, 1], projection='3d')

    # Collect all valid points to set global bounds
    all_vis_pts = []
    trajectory_paths = {}
    for p in processors:
        gts = results['drones'][p.drone_id]['leds_gt']
        acts = results['drones'][p.drone_id]['leds_act']
        valid_pts = [x for x in gts if x is not None] + [x for x in acts if x is not None]
        if valid_pts:
            all_vis_pts.append(np.vstack(valid_pts))

        if trajectory_rmse or segmented_trajectory_metrics is not None:
            gt_path, act_path = get_trajectory_positions(
                p, timestamps, position_alignment
            )
            if len(gt_path) > 0:
                trajectory_paths[p.drone_id] = (gt_path, act_path)
                all_vis_pts.extend([gt_path, act_path])

    if all_vis_pts:
        set_axes_equal(ax_3d, np.vstack(all_vis_pts))

    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    replay_label = 'Position' if position_only else 'LED'
    replay_title = f'Multi-Drone {replay_label} Replay'
    if trajectory_rmse or segmented_trajectory_metrics is not None:
        replay_title += ' with Trajectories'
    ax_3d.set_title(replay_title)

    # Draw the complete center-position paths behind the animated markers.
    for i, p in enumerate(processors):
        if p.drone_id not in trajectory_paths:
            continue
        gt_path, act_path = trajectory_paths[p.drone_id]
        ax_3d.plot(
            gt_path[:, 0], gt_path[:, 1], gt_path[:, 2],
            color=colors[i], linewidth=1.0, linestyle='-', alpha=0.5,
            label=f'{p.drone_id} GT trajectory',
        )
        ax_3d.plot(
            act_path[:, 0], act_path[:, 1], act_path[:, 2],
            color=colors[i], linewidth=1.0, linestyle='--', alpha=0.9,
            label=f'{p.drone_id} Act trajectory',
        )

    # Create Scatter Objects
    scatters = {}
    for i, p in enumerate(processors):
        # GT = Solid Circle, Act = Triangle
        sc_gt = ax_3d.scatter([], [], [], color=colors[i], marker='o', s=15, alpha=0.6, label=f'{p.drone_id} GT')
        sc_act = ax_3d.scatter([], [], [], color=colors[i], marker='^', s=15, label=f'{p.drone_id} Act')
        scatters[p.drone_id] = (sc_gt, sc_act)

    ax_3d.legend()

    # --- ANIMATION CONTROLS ---
    plt.subplots_adjust(bottom=0.25)

    # Determine number of frames in the visual subsample
    # Pick the first drone's list length (they should be identical due to uniform loop)
    num_frames = len(results['drones'][processors[0].drone_id]['leds_gt'])

    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Frame', 0, num_frames - 1, valinit=0, valstep=1)

    ax_play = plt.axes([0.05, 0.1, 0.1, 0.04])
    btn_play = Button(ax_play, 'Pause')  # Start playing by default

    def update_anim(val):
        idx = int(val)

        # Approximate time for display (since we subsampled by 10)
        t_disp = timestamps[min(idx * 10, len(timestamps) - 1)]
        ax_3d.set_title(f"{replay_title} t={t_disp:.2f}s")
        
        if visualize_only:
            for p in processors:
                if p.drone_id in left_texts:
                    pos = p.gt_pos_fn(t_disp)
                    txt = left_texts[p.drone_id]
                    txt.set_text(f"x: {pos[0]:.2f}    y: {pos[1]:.2f}    z: {pos[2]:.2f}")
                if p.drone_id in time_lines:
                    time_lines[p.drone_id].set_xdata([t_disp, t_disp])

        for p in processors:
            gt_list = results['drones'][p.drone_id]['leds_gt']
            act_list = results['drones'][p.drone_id]['leds_act']

            s_gt, s_act = scatters[p.drone_id]

            if idx < len(gt_list) and gt_list[idx] is not None:
                gt_pts = gt_list[idx]
                act_pts = act_list[idx]
                s_gt._offsets3d = (gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2])
                if act_pts is not None:
                    s_act._offsets3d = (act_pts[:, 0], act_pts[:, 1], act_pts[:, 2])
                else:
                    s_act._offsets3d = ([], [], [])
            else:
                # Hide if invalid for this frame
                s_gt._offsets3d = ([], [], [])
                s_act._offsets3d = ([], [], [])

        fig.canvas.draw_idle()

    slider.on_changed(update_anim)

    class Player:
        def __init__(self):
            self.playing = True
            self.anim = None

        def toggle(self, event):
            if self.playing:
                self.anim.event_source.stop()
                btn_play.label.set_text('Play')
                self.playing = False
            else:
                self.anim.event_source.start()
                btn_play.label.set_text('Pause')
                self.playing = True

    player = Player()
    btn_play.on_clicked(player.toggle)

    def animate_step(i):
        # Update slider which triggers update_anim
        slider.set_val(i)

    player.anim = animation.FuncAnimation(fig, animate_step, frames=num_frames, interval=50, blit=False)

    plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RMSE Analysis Toolkit")
    parser.add_argument('--yaml', type=str, default=None, help='Path to mission YAML (automatically resolved from the tag directory when omitted)')
    parser.add_argument('--tag', type=str, default=None, help='Log file tag; may be used without --yaml')
    parser.add_argument('--compare_yaml', type=str, default=None, help='Compare two yaml files directly without JSON logs')
    parser.add_argument('--kinematics', action='store_true', help='Enable kinematic modeling of ground truth')
    parser.add_argument('--max_v', type=float, default=1.0, help='Maximum velocity (m/s)')
    parser.add_argument('--max_a', type=float, default=0.25, help='Maximum acceleration (m/s^2)')
    parser.add_argument('--max_j', type=float, default=17.0, help='Maximum jerk (m/s^3)')
    parser.add_argument('--max_s', type=float, default=550.0, help='Maximum snap (m/s^4)')
    parser.add_argument('--trim-start', type=float, default=0.0, help='Trim start of data (seconds)')
    parser.add_argument('--trim-end', type=float, default=0.0, help='End analysis at this relative time in seconds')
    parser.add_argument('--ignore-rpy', action='store_true', dest='ignore_rpy', help='Ignore actual roll, pitch, and yaw data')
    parser.add_argument(
        '--mocap-marker-position',
        type=float,
        nargs=3,
        metavar=('X', 'Y', 'Z'),
        default=(0.0, 0.0, 0.0),
        help=(
            'Mocap marker position in the drone body FLU frame, in metres; '
            'used with actual attitude to recover the body origin '
            '(default: 0 0 0)'
        ),
    )
    metric_group = parser.add_mutually_exclusive_group()
    metric_group.add_argument('--lit-only', action='store_true', dest='lit_only', help='Only include lit (non-black) LEDs in RMSE computation and visualization')
    metric_group.add_argument('--position-only', action='store_true', help='Compute RMSE from drone center positions instead of LED positions')
    parser.add_argument('--align-position-means', action='store_true', help='Translate all GT positions by one constant offset so their mean matches the mean actual position')
    parser.add_argument('--trajectory-rmse', action='store_true', help='Report symmetric nearest-path position RMSE without matching samples by time')
    parser.add_argument(
        '--trajectory-rmse-over-time',
        type=float,
        metavar='SECONDS',
        help='Plot trajectory RMSE for consecutive segments of this duration',
    )
    parser.add_argument(
        '--trajectory-rmse-over-time-accumulative',
        '--trajectory-rmse-over-time-cumulative',
        action='store_true',
        dest='trajectory_rmse_over_time_accumulative',
        help=(
            'Make each trajectory-RMSE interval run from the analysis start '
            'through the end of the current segment'
        ),
    )

    args = parser.parse_args()

    if (
            args.trajectory_rmse_over_time_accumulative
            and args.trajectory_rmse_over_time is None):
        parser.error(
            '--trajectory-rmse-over-time-accumulative requires '
            '--trajectory-rmse-over-time SECONDS'
        )

    try:
        yaml_file = resolve_yaml_file(args.yaml, args.tag)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    calculate_rmse(
        yaml_file,
        args.tag,
        args.compare_yaml,
        args.kinematics,
        args.max_v,
        args.max_a,
        args.max_j,
        args.max_s,
        args.ignore_rpy,
        args.lit_only,
        args.trim_start,
        args.trim_end,
        align_position_means=args.align_position_means,
        position_only=args.position_only,
        trajectory_rmse=args.trajectory_rmse,
        trajectory_rmse_segment_duration=args.trajectory_rmse_over_time,
        trajectory_rmse_accumulative=(
            args.trajectory_rmse_over_time_accumulative
        ),
        mocap_marker_position=args.mocap_marker_position,
    )
