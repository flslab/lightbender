import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "analyze_interaction_logs.py"
SPEC = importlib.util.spec_from_file_location("analyze_interaction_logs", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AnalyzerTests(unittest.TestCase):
    def test_pointcloud_frames_use_nearest_onboard_velocity(self):
        frames = [{"time": 1.01, "tvec": [0, 0, 1]}]
        records = [{
            "type": "state", "group": "VEL_ORI", "data": {
                "time": 1.0,
                "stateEstimate.vx": 0.1,
                "stateEstimate.vy": -0.2,
                "stateEstimate.vz": 0.3,
            },
        }]
        result = MODULE.attach_onboard_velocities(frames, records)
        self.assertEqual(result[0]["vel"], [0.1, -0.2, 0.3])
        self.assertEqual(
            result[0]["velocity_source"], "crazyflie_state_estimate"
        )

    def test_directory_selection_ignores_tracker_and_reference_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lb11_flight.json", "lb11_tracker_flight.json", "vicon_flight.json"):
                (root / name).write_text("[]", encoding="utf-8")
            selected = MODULE.json_logs(root)
        self.assertEqual([path.name for path in selected], ["lb11_flight.json"])

    def test_recovers_unclosed_array_and_discards_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flight.json"
            path.write_text('[\n{"type":"start","data":1},\n{"type":"frames"', encoding="utf-8")
            records, integrity = MODULE.load_records(path)
        self.assertEqual(records, [{"type": "start", "data": 1}])
        self.assertFalse(integrity["valid_json"])
        self.assertFalse(integrity["closed_array"])
        self.assertEqual(integrity["discarded_lines"], 1)

    def test_flags_dropout_followed_by_motor_cutoff(self):
        records = [
            {"type": "configs", "name": "Translation Config", "data": {"delta_v": 0.13}},
            {"type": "events", "name": "Waiting For User Interaction", "data": {"time": 10.0}},
        ]
        for index in range(30):
            records.append({"type": "frames", "data": {
                "time": 10.0 + index * 0.01, "tvec": [0, 0, 1], "vel": [0.01, 0, 0]
            }})
        for index in range(100):
            value = 1000 if index < 70 else 0
            records.append({"type": "state", "group": "MOT_BAT", "data": {
                "time": 10.0 + index * 0.01,
                "motor.m1": value, "motor.m2": value, "motor.m3": value, "motor.m4": value,
            }})
        report = MODULE.analyze_records(Path("flight.json"), records, {
            "valid_json": True, "closed_array": True, "recovered": False, "discarded_lines": 0,
        })
        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("LOCALIZATION_DROPOUT", codes)
        self.assertIn("MOTOR_CUTOFF_AFTER_DROPOUT", codes)

    def test_localization_noise_excludes_commanded_excitation(self):
        records = [
            {"type": "events", "name": "Waiting For User Interaction",
             "data": {"time": 10.0}},
            {"type": "events", "name": "Wrench Calibration Excitation Started",
             "data": {"time": 11.0}},
        ]
        for index in range(300):
            time_s = 10.0 + index * 0.01
            speed = 0.01 if time_s < 11.0 else 0.2
            records.append({"type": "frames", "data": {
                "time": time_s,
                "tvec": [speed * index * 0.01, 0, 1],
                "vel": [speed, 0, 0],
            }})
        report = MODULE.analyze_records(Path("flight.json"), records, {
            "valid_json": True, "closed_array": True, "recovered": False,
            "discarded_lines": 0,
        })
        self.assertAlmostEqual(report["localization_noise"]["speed_max_m_s"], 0.01)
        self.assertNotIn(
            "LOW_NOISE_MARGIN",
            {finding["code"] for finding in report["findings"]},
        )

    def test_summarizes_wrench_shadow_log_and_uncalibrated_angular_model(self):
        records = [{
            "type": "configs", "name": "Wrench Interaction Config", "data": {
                "nominal_position": [0, 0, 1], "nominal_yaw_deg": 0,
                "config": {
                    "shadow_mode": True,
                    "motor_model": {"angular_accel_scale": [0, 0, 0]},
                    "safety": {"max_frame_age_s": 0.1, "max_motor_age_s": 0.15},
                },
            },
        }]
        for index in range(20):
            baseline = [0.1, 0, 1] if index == 5 else [0, 0, 1]
            records.append({"type": "wrench_observer", "data": {
                "calibrated": True,
                "measurement_rejected": False,
                "external_force_N": [0.01, -0.02, 0.03],
                "external_torque_Nm": [0.0001, -0.0002, 0.0003],
                "translation_contact": {"active": False},
                "yaw_contact": {"active": False},
                "frame_age_s": 0.01,
                "motor_data_age_s": 0.02,
                "motor_pose_skew_s": 0.003,
                "force_bias_N": [0, 0, 0],
                "torque_bias_Nm": [0, 0, 0],
                "shadow_mode": True,
                "baseline_position_m": baseline,
                "baseline_yaw_deg": 0,
                "calibration_excitation_active": index == 5,
                "command_position_m": baseline,
                "command_yaw_deg": 0,
            }})
        report = MODULE.analyze_records(Path("flight.json"), records, {
            "valid_json": True, "closed_array": True, "recovered": False,
            "discarded_lines": 0,
        })
        self.assertEqual(report["wrench"]["sample_count"], 20)
        self.assertEqual(report["wrench"]["shadow_command_change_count"], 0)
        self.assertEqual(report["wrench"]["quiet_abs_force_p99_N"], [0.01, 0.02, 0.03])
        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("ANGULAR_MODEL_UNCALIBRATED", codes)

    def test_disabled_yaw_omits_yaw_model_warnings(self):
        records = [{
            "type": "configs", "name": "Wrench Interaction Config", "data": {
                "config": {
                    "shadow_mode": True,
                    "detection": {"yaw": {"enabled": False}},
                    "motor_model": {"angular_accel_scale": [0, 0, 0]},
                    "safety": {},
                },
            },
        }, {
            "type": "wrench_observer", "data": {
                "calibrated": True,
                "measurement_rejected": False,
                "external_force_N": [0, 0, 0],
                "external_torque_Nm": [0, 0, 0],
                "translation_contact": {"active": False},
                "yaw_contact": {"active": False},
                "shadow_mode": True,
            },
        }]
        report = MODULE.analyze_records(Path("flight.json"), records, {
            "valid_json": True, "closed_array": True, "recovered": False,
            "discarded_lines": 0,
        })
        codes = {finding["code"] for finding in report["findings"]}
        self.assertFalse(report["wrench"]["yaw_detection_enabled"])
        self.assertFalse(codes & {
            "ANGULAR_MODEL_UNCALIBRATED", "YAW_CONTROL_LOG_MISSING",
            "YAW_DYNAMICS_FIT_WEAK", "ANGULAR_MODEL_FIT_WEAK",
        })

    def test_summarizes_onboard_wrench_state_source(self):
        records = [{
            "type": "configs", "name": "Onboard Wrench Interaction Config",
            "data": {
                "state_source": "crazyflie_state_estimate",
                "nominal_position": [0, 0, 1],
                "nominal_yaw_deg": 0,
                "config": {
                    "shadow_mode": True,
                    "motor_model": {"angular_accel_scale": [1, 1, 1]},
                    "safety": {
                        "max_state_age_s": 0.1,
                        "max_motor_age_s": 0.15,
                        "max_motor_state_skew_s": 0.03,
                    },
                },
            },
        }, {
            "type": "wrench_observer", "data": {
                "state_source": "crazyflie_state_estimate",
                "calibrated": True,
                "measurement_rejected": False,
                "external_force_N": [0, 0, 0],
                "external_torque_Nm": [0, 0, 0],
                "translation_contact": {"active": False},
                "yaw_contact": {"active": False},
                "state_age_s": 0.01,
                "state_group_skew_s": 0.002,
                "motor_data_age_s": 0.02,
                "motor_pose_skew_s": 0.003,
                "force_bias_N": [0, 0, 0],
                "torque_bias_Nm": [0, 0, 0],
                "shadow_mode": True,
                "baseline_position_m": [0, 0, 1],
                "baseline_yaw_deg": 0,
                "command_position_m": [0, 0, 1],
                "command_yaw_deg": 0,
            },
        }]
        report = MODULE.analyze_records(Path("flight.json"), records, {
            "valid_json": True, "closed_array": True, "recovered": False,
            "discarded_lines": 0,
        })
        self.assertEqual(
            report["wrench"]["state_source"],
            "crazyflie_state_estimate",
        )
        self.assertEqual(report["wrench"]["max_state_age_s"], 0.01)

    def test_fits_angular_motor_scale_from_quiet_shadow_rows(self):
        config = {
            "inertia": [0.002, 0.0025, 0.003],
            "motor_model": {
                "hover_pwm": 30000, "hover_voltage": 8.0,
                "angular_accel_scale": [0, 0, 0],
            },
        }
        mixer = [
            [0.25, 0.25, -0.25, -0.25],
            [0.25, -0.25, -0.25, 0.25],
            [-0.25, 0.25, -0.25, 0.25],
        ]
        true_scale = [120.0, -90.0, 70.0]
        rows = []
        for index in range(100):
            pwm = [
                30000 + 1800 * math.sin(index * 0.11 + phase)
                for phase in (0.0, 0.7, 1.8, 2.6)
            ]
            thrust = [(value / 30000) ** 2 for value in pwm]
            mix = [sum(mixer[axis][motor] * thrust[motor] for motor in range(4))
                   for axis in range(3)]
            torque = [config["inertia"][axis] * true_scale[axis] * mix[axis]
                      for axis in range(3)]
            rows.append({
                "motor_pwm": pwm, "battery_voltage_V": 8.0,
                "external_torque_Nm": torque,
            })
        fit = MODULE.fit_angular_motor_scale(rows, config)
        self.assertIsNotNone(fit)
        for actual, expected in zip(fit["estimated_angular_accel_scale"], true_scale):
            self.assertAlmostEqual(actual, expected, places=8)
        for value in fit["r_squared"]:
            self.assertAlmostEqual(value, 1.0, places=8)

    def test_fits_causal_yaw_dynamics_with_command_delay(self):
        dt = 0.01
        delay_samples = 4
        gain = 0.003
        damping = 0.8
        bias = 0.02
        commands = []
        rates = [0.0]
        rows = []
        for index in range(1200):
            time_s = index * dt
            frequency = 0.08 + 0.4 * time_s / 12.0
            command = 2500.0 * math.sin(2.0 * math.pi * frequency * time_s)
            commands.append(command)
            if index:
                delayed = commands[max(0, index - 1 - delay_samples)]
                rates.append(rates[-1] + dt * (
                    gain * delayed - damping * rates[-1] + bias
                ))
            rows.append({
                "state_time": time_s,
                "controller_yaw_command": command,
                "controller_yaw_rate_rad_s": rates[-1],
            })
        fit = MODULE.fit_causal_yaw_dynamics(rows)
        self.assertIsNotNone(fit)
        self.assertLessEqual(abs(fit["delay_samples"] - delay_samples), 1)
        self.assertAlmostEqual(fit["yaw_accel_per_command"], gain, places=4)
        self.assertAlmostEqual(fit["yaw_damping_per_s"], damping, places=1)
        self.assertGreater(fit["test_r_squared"], 0.95)


if __name__ == "__main__":
    unittest.main()
