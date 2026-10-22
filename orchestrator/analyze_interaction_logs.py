#!/usr/bin/env python3
"""Crash-tolerant diagnostics for LightBender interaction flight logs.

The input may be either a session directory or one controller JSON log.  The
tool deliberately accepts the unterminated JSON arrays produced when an RPi or
controller exits before ``LiveLogger.close()`` runs.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def load_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load a normal JSON array or recover complete objects line-by-line."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    integrity = {
        "valid_json": True,
        "closed_array": raw.rstrip().endswith("]"),
        "recovered": False,
        "discarded_lines": 0,
    }
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("root is not an array")
        return parsed, integrity
    except (json.JSONDecodeError, ValueError):
        integrity["valid_json"] = False
        integrity["recovered"] = True

    records: list[dict[str, Any]] = []
    for line in raw.splitlines():
        text = line.strip().rstrip(",")
        if not text or text in ("[", "]"):
            continue
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            integrity["discarded_lines"] += 1
            continue
        if isinstance(value, dict):
            records.append(value)
    return records, integrity


def magnitude(values: Iterable[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values))


def record_time(record: dict[str, Any]) -> float | None:
    data = record.get("data")
    if isinstance(data, dict) and isinstance(data.get("time"), (int, float)):
        return float(data["time"])
    if record.get("type") == "start" and isinstance(data, (int, float)):
        return float(data)
    return None


def attach_onboard_velocities(
        frames: list[dict[str, Any]],
        records: list[dict[str, Any]],
        max_skew_s: float = 0.05,
) -> list[dict[str, Any]]:
    """Fill pointcloud-frame velocity from nearest onboard VEL_ORI packet."""
    velocity_packets = [
        record["data"] for record in records
        if record.get("type") == "state"
        and record.get("group") == "VEL_ORI"
        and isinstance(record.get("data"), dict)
        and isinstance(record["data"].get("time"), (int, float))
        and all(isinstance(record["data"].get(name), (int, float)) for name in (
            "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz"
        ))
    ]
    velocity_packets.sort(key=lambda packet: float(packet["time"]))
    velocity_times = [float(packet["time"]) for packet in velocity_packets]
    result = []
    for original in frames:
        frame = original.copy()
        if isinstance(frame.get("vel"), list) or not velocity_packets:
            result.append(frame)
            continue
        frame_time = float(frame["time"])
        index = bisect.bisect_left(velocity_times, frame_time)
        candidate_indices = [
            candidate for candidate in (index - 1, index)
            if 0 <= candidate < len(velocity_packets)
        ]
        nearest_index = min(
            candidate_indices,
            key=lambda candidate: abs(velocity_times[candidate] - frame_time),
        )
        if abs(velocity_times[nearest_index] - frame_time) <= max_skew_s:
            packet = velocity_packets[nearest_index]
            frame["vel"] = [
                float(packet["stateEstimate.vx"]),
                float(packet["stateEstimate.vy"]),
                float(packet["stateEstimate.vz"]),
            ]
            frame["velocity_source"] = "crazyflie_state_estimate"
        result.append(frame)
    return result


def interaction_sessions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sessions: list[dict[str, Any]] = []
    pushing: list[dict[str, Any]] = []
    for record in records:
        if record.get("type") != "events":
            continue
        if record.get("name") == "User Pushing":
            pushing.append(record)
        elif record.get("name") == "User Disengage" and pushing:
            start = float(pushing[0]["data"]["time"])
            end = float(record["data"]["time"])
            peak = max(float(item["data"].get("speed", 0.0)) for item in pushing)
            sessions.append({
                "start": start,
                "end": end,
                "duration_s": end - start,
                "push_event_count": len(pushing),
                "peak_event_speed_m_s": peak,
                "start_position_m": pushing[0]["data"].get("Pos"),
                "end_position_m": record["data"].get("Pos"),
            })
            pushing = []
    return sessions


def sustained_motor_cutoff(states: list[dict[str, Any]]) -> float | None:
    motor_rows = [
        item["data"] for item in states
        if item.get("group") == "MOT_BAT" and isinstance(item.get("data"), dict)
    ]
    was_flying = False
    for index, row in enumerate(motor_rows):
        nonzero = any(float(row.get(f"motor.m{motor}", 0)) > 0 for motor in range(1, 5))
        was_flying = was_flying or nonzero
        if was_flying and not nonzero:
            following = motor_rows[index:index + 20]
            if len(following) >= 10 and all(
                all(float(candidate.get(f"motor.m{motor}", 0)) == 0 for motor in range(1, 5))
                for candidate in following
            ):
                return float(row["time"])
    return None


def component_percentiles(rows: list[list[float]], q: float) -> list[float | None]:
    if not rows:
        return [None, None, None]
    width = len(rows[0])
    return [percentile([abs(float(row[axis])) for row in rows], q) for axis in range(width)]


def fit_angular_motor_scale(
        rows: list[dict[str, Any]],
        config: dict[str, Any],
) -> dict[str, Any] | None:
    """Fit diagonal angular-acceleration/motor-mixer scales from quiet rows."""
    motor = config.get("motor_model", {})
    inertia = config.get("inertia", [])
    old_scale = motor.get("angular_accel_scale", [])
    if len(inertia) != 3 or len(old_scale) != 3:
        return None
    hover_pwm = float(motor.get("hover_pwm", 0.0))
    hover_voltage = float(motor.get("hover_voltage", 0.0))
    if hover_pwm <= 0 or hover_voltage <= 0:
        return None
    mixer = motor.get("motor_mixer", [
        [0.25, 0.25, -0.25, -0.25],
        [0.25, -0.25, -0.25, 0.25],
        [-0.25, 0.25, -0.25, 0.25],
    ])

    samples: list[tuple[list[float], list[float]]] = []
    for row in rows:
        pwm = row.get("motor_pwm")
        voltage = row.get("battery_voltage_V")
        torque = row.get("external_torque_Nm")
        if not isinstance(pwm, list) or len(pwm) != 4 or not isinstance(torque, list) or len(torque) != 3:
            continue
        if not isinstance(voltage, (int, float)) or any(not isinstance(value, (int, float)) for value in pwm):
            continue
        thrust = [((float(value) / hover_pwm) * (float(voltage) / hover_voltage)) ** 2 for value in pwm]
        mix = [sum(float(mixer[axis][motor_index]) * thrust[motor_index] for motor_index in range(4))
               for axis in range(3)]
        total_acceleration = [
            float(torque[axis]) / float(inertia[axis]) + float(old_scale[axis]) * mix[axis]
            for axis in range(3)
        ]
        samples.append((mix, total_acceleration))
    if len(samples) < 20:
        return None

    scales: list[float | None] = []
    r_squared: list[float | None] = []
    input_std: list[float] = []
    for axis in range(3):
        x = [sample[0][axis] for sample in samples]
        y = [sample[1][axis] for sample in samples]
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        xx = sum((value - x_mean) ** 2 for value in x)
        yy = sum((value - y_mean) ** 2 for value in y)
        xy = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
        input_std.append(statistics.pstdev(x))
        if xx <= 1e-12 or yy <= 1e-12:
            scales.append(None)
            r_squared.append(None)
        else:
            scales.append(xy / xx)
            r_squared.append((xy * xy) / (xx * yy))
    return {
        "sample_count": len(samples),
        "estimated_angular_accel_scale": scales,
        "r_squared": r_squared,
        "mixer_input_std": input_std,
    }


def _fit_two_feature_regression(
        rows: list[tuple[float, float, float]],
) -> tuple[list[float], float] | None:
    """Fit y = b0*x0 + b1*x1 + intercept without external dependencies."""
    if len(rows) < 10:
        return None
    x0 = [row[0] for row in rows]
    x1 = [row[1] for row in rows]
    y = [row[2] for row in rows]
    means = [statistics.mean(x0), statistics.mean(x1), statistics.mean(y)]
    a = [value - means[0] for value in x0]
    b = [value - means[1] for value in x1]
    c = [value - means[2] for value in y]
    aa = sum(value * value for value in a)
    bb = sum(value * value for value in b)
    ab = sum(left * right for left, right in zip(a, b))
    ac = sum(left * right for left, right in zip(a, c))
    bc = sum(left * right for left, right in zip(b, c))
    determinant = aa * bb - ab * ab
    if abs(determinant) <= 1e-18:
        return None
    beta0 = (ac * bb - bc * ab) / determinant
    beta1 = (bc * aa - ac * ab) / determinant
    intercept = means[2] - beta0 * means[0] - beta1 * means[1]
    predictions = [
        beta0 * left + beta1 * right + intercept
        for left, right in zip(x0, x1)
    ]
    residual = sum((actual - predicted) ** 2 for actual, predicted in zip(y, predictions))
    total = sum((actual - means[2]) ** 2 for actual in y)
    r_squared = 1.0 - residual / total if total > 1e-18 else None
    return [beta0, beta1, intercept], r_squared


def _regression_r_squared(
        rows: list[tuple[float, float, float]], coefficients: list[float],
) -> float | None:
    if not rows:
        return None
    actual = [row[2] for row in rows]
    mean = statistics.mean(actual)
    predicted = [
        coefficients[0] * row[0] + coefficients[1] * row[1] + coefficients[2]
        for row in rows
    ]
    residual = sum((value - estimate) ** 2 for value, estimate in zip(actual, predicted))
    total = sum((value - mean) ** 2 for value in actual)
    return 1.0 - residual / total if total > 1e-18 else None


def fit_causal_yaw_dynamics(
        rows: list[dict[str, Any]],
        window_s: float = 0.10,
        max_delay_s: float = 0.20,
) -> dict[str, Any] | None:
    """Identify yaw actuation from PID output using causal rate increments.

    The integrated model avoids differentiating the gyro sample-by-sample:

      delta(omega) = ku * integral(cmd_yaw)
                     - damping * integral(omega) + bias * duration

    Delay selection uses a chronological train/validation split and the final
    metrics are reported on a held-out high-frequency tail of the chirp.
    """
    samples = []
    for row in rows:
        timestamp = row.get("state_time", row.get("time"))
        command = row.get("controller_yaw_command")
        rate = row.get("controller_yaw_rate_rad_s")
        if not isinstance(rate, (int, float)):
            angular_velocity = row.get("angular_velocity_rad_s")
            rate = (
                angular_velocity[2]
                if isinstance(angular_velocity, list) and len(angular_velocity) == 3
                else None
            )
        if all(isinstance(value, (int, float)) and math.isfinite(float(value))
               for value in (timestamp, command, rate)):
            samples.append((float(timestamp), float(command), float(rate)))
    samples.sort(key=lambda value: value[0])
    if len(samples) < 100:
        return None
    positive_dt = [
        later[0] - earlier[0] for earlier, later in zip(samples, samples[1:])
        if 0.0 < later[0] - earlier[0] <= 0.05
    ]
    if not positive_dt:
        return None
    nominal_dt = statistics.median(positive_dt)
    window_samples = max(2, round(float(window_s) / nominal_dt))
    max_delay_samples = max(0, round(float(max_delay_s) / nominal_dt))

    def integrated_rows(delay_samples: int) -> list[tuple[float, float, float]]:
        result = []
        for start in range(delay_samples, len(samples) - window_samples):
            end = start + window_samples
            command_integral = 0.0
            negative_rate_integral = 0.0
            duration = 0.0
            valid = True
            for index in range(start, end):
                dt = samples[index + 1][0] - samples[index][0]
                if dt <= 0.0 or dt > 0.05:
                    valid = False
                    break
                command_integral += dt * samples[index - delay_samples][1]
                negative_rate_integral -= dt * samples[index][2]
                duration += dt
            if valid and duration > 0.0:
                delta_rate = samples[end][2] - samples[start][2]
                result.append((command_integral, negative_rate_integral, delta_rate))
        return result

    best = None
    for delay_samples in range(max_delay_samples + 1):
        candidates = integrated_rows(delay_samples)
        if len(candidates) < 60:
            continue
        train_end = max(30, int(0.60 * len(candidates)))
        validation_end = max(train_end + 15, int(0.80 * len(candidates)))
        fitted = _fit_two_feature_regression(candidates[:train_end])
        if fitted is None:
            continue
        coefficients, train_r_squared = fitted
        validation_r_squared = _regression_r_squared(
            candidates[train_end:validation_end], coefficients
        )
        score = -math.inf if validation_r_squared is None else validation_r_squared
        if best is None or score > best[0]:
            best = (
                score, delay_samples, candidates, train_end, validation_end,
                train_r_squared, validation_r_squared,
            )
    if best is None:
        return None

    _, delay_samples, candidates, _train_end, validation_end, _, validation_r_squared = best
    final_fit = _fit_two_feature_regression(candidates[:validation_end])
    if final_fit is None:
        return None
    coefficients, fit_r_squared = final_fit
    test_r_squared = _regression_r_squared(candidates[validation_end:], coefficients)
    mean_window_duration = window_samples * nominal_dt
    return {
        "sample_count": len(candidates),
        "source": "controller.cmd_yaw/controller.r_yaw",
        "window_s": mean_window_duration,
        "delay_samples": delay_samples,
        "delay_s": delay_samples * nominal_dt,
        "yaw_accel_per_command": coefficients[0],
        "yaw_damping_per_s": coefficients[1],
        "yaw_bias_rad_s2": coefficients[2] / mean_window_duration,
        "fit_r_squared": fit_r_squared,
        "validation_r_squared": validation_r_squared,
        "test_r_squared": test_r_squared,
    }


def analyze_wrench_records(
        records: list[dict[str, Any]],
        wrench_config: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    rows = [
        record["data"] for record in records
        if record.get("type") == "wrench_observer"
        and isinstance(record.get("data"), dict)
    ]
    if not rows:
        return None, []

    calibrated = [row for row in rows if row.get("calibrated")]
    accepted = [row for row in calibrated if not row.get("measurement_rejected")]
    quiet = [
        row for row in accepted
        if not (row.get("translation_contact") or {}).get("active", False)
        and not (row.get("yaw_contact") or {}).get("active", False)
    ]
    force_rows = [row["external_force_N"] for row in quiet if row.get("external_force_N")]
    torque_rows = [row["external_torque_Nm"] for row in quiet if row.get("external_torque_Nm")]
    state_source = rows[-1].get(
        "state_source", wrench_config.get("state_source", "mocap_full_pose")
    )
    frame_ages = [
        float(row.get("state_age_s", row.get("frame_age_s"))) for row in rows
        if isinstance(row.get("state_age_s", row.get("frame_age_s")), (int, float))
    ]
    state_group_skews = [
        float(row["state_group_skew_s"]) for row in rows
        if isinstance(row.get("state_group_skew_s"), (int, float))
    ]
    motor_ages = [float(row["motor_data_age_s"]) for row in rows if isinstance(row.get("motor_data_age_s"), (int, float))]
    motor_pose_skews = [float(row["motor_pose_skew_s"]) for row in rows if isinstance(row.get("motor_pose_skew_s"), (int, float))]
    yaw_control_skews = [
        float(row["controller_yaw_skew_s"]) for row in rows
        if isinstance(row.get("controller_yaw_skew_s"), (int, float))
    ]

    shadow_mode = bool(rows[-1].get("shadow_mode", wrench_config.get("shadow_mode", True)))
    nominal = wrench_config.get("nominal_position")
    nominal_yaw = wrench_config.get("nominal_yaw_deg")
    shadow_command_changes = 0
    if shadow_mode and nominal is not None:
        for row in rows:
            command = row.get("command_position_m")
            command_yaw = row.get("command_yaw_deg")
            expected_position = row.get("baseline_position_m", nominal)
            expected_yaw = row.get("baseline_yaw_deg", nominal_yaw if nominal_yaw is not None else 0.0)
            if command and (
                any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(command, expected_position))
                or (command_yaw is not None and abs(float(command_yaw) - float(expected_yaw)) > 1e-6)
            ):
                shadow_command_changes += 1

    def active_count(name: str) -> int:
        return sum(bool((row.get(name) or {}).get("active")) for row in calibrated)

    summary = {
        "sample_count": len(rows),
        "state_source": state_source,
        "calibrated_sample_count": len(calibrated),
        "quiet_accepted_sample_count": len(quiet),
        "shadow_mode": shadow_mode,
        "shadow_command_change_count": shadow_command_changes,
        "measurement_rejection_count": len(calibrated) - len(accepted),
        "measurement_rejection_fraction": (
            (len(calibrated) - len(accepted)) / len(calibrated) if calibrated else None
        ),
        "translation_contact_sample_count": active_count("translation_contact"),
        "yaw_contact_sample_count": active_count("yaw_contact"),
        "calibration_excitation_sample_count": sum(
            bool(row.get("calibration_excitation_active")) for row in calibrated
        ),
        "quiet_abs_force_p95_N": component_percentiles(force_rows, 0.95),
        "quiet_abs_force_p99_N": component_percentiles(force_rows, 0.99),
        "quiet_abs_torque_p95_Nm": component_percentiles(torque_rows, 0.95),
        "quiet_abs_torque_p99_Nm": component_percentiles(torque_rows, 0.99),
        "max_frame_age_s": max(frame_ages) if frame_ages else None,
        "max_state_age_s": max(frame_ages) if frame_ages else None,
        "max_state_group_skew_s": max(state_group_skews) if state_group_skews else None,
        "max_motor_data_age_s": max(motor_ages) if motor_ages else None,
        "max_motor_pose_skew_s": max(motor_pose_skews) if motor_pose_skews else None,
        "max_yaw_control_skew_s": max(yaw_control_skews) if yaw_control_skews else None,
        "final_force_bias_N": rows[-1].get("force_bias_N"),
        "final_torque_bias_Nm": rows[-1].get("torque_bias_Nm"),
    }

    findings: list[dict[str, str]] = []
    if not calibrated:
        findings.append({"severity": "critical", "code": "WRENCH_CALIBRATION_INCOMPLETE",
                         "message": "The wrench observer never completed its startup bias calibration."})
    rejection_fraction = summary["measurement_rejection_fraction"]
    if rejection_fraction is not None and rejection_fraction > 0.05:
        findings.append({"severity": "warning", "code": "HIGH_WRENCH_MEASUREMENT_REJECTION",
                         "message": f"The observer rejected {100 * rejection_fraction:.1f}% of calibrated pose updates."})
    if shadow_command_changes:
        findings.append({"severity": "critical", "code": "SHADOW_MODE_COMMAND_CHANGED",
                         "message": f"Shadow mode changed {shadow_command_changes} commanded references."})

    nested_config = wrench_config.get("config", wrench_config)
    yaw_detection_enabled = bool(
        nested_config.get("detection", {}).get("yaw", {}).get("enabled", True)
    )
    summary["yaw_detection_enabled"] = yaw_detection_enabled
    # Prefer the explicit contact-free excitation interval. Fall back to all
    # accepted rows for older logs that predate the excitation marker.
    excitation_rows = [row for row in accepted if row.get("calibration_excitation_active")]
    identification_rows = excitation_rows if len(excitation_rows) >= 100 else accepted
    yaw_dynamics_fit = (
        fit_causal_yaw_dynamics(identification_rows)
        if yaw_detection_enabled else None
    )
    summary["yaw_dynamics_fit"] = yaw_dynamics_fit
    angular_fit = (
        fit_angular_motor_scale(
            excitation_rows if len(excitation_rows) >= 20 else accepted,
            nested_config,
        )
        if yaw_detection_enabled else None
    )
    summary["angular_motor_scale_fit"] = angular_fit
    configured_motor_model = nested_config.get("motor_model", {})
    angular_scale = configured_motor_model.get("angular_accel_scale", [])
    yaw_command_config = configured_motor_model.get("yaw_command_model", {})
    yaw_command_model_ready = (
        bool(yaw_command_config.get("enabled", False))
        and isinstance(yaw_command_config.get("accel_per_command"), (int, float))
        and abs(float(yaw_command_config["accel_per_command"])) > 1e-12
    )
    legacy_yaw_model_ready = (
        len(angular_scale) >= 3 and abs(float(angular_scale[2])) > 1e-12
    )
    if yaw_detection_enabled and not (yaw_command_model_ready or legacy_yaw_model_ready):
        findings.append({"severity": "warning", "code": "ANGULAR_MODEL_UNCALIBRATED",
                         "message": "No calibrated yaw actuator model is enabled; commanded yaw can look like external torque."})
    if yaw_detection_enabled and yaw_dynamics_fit is None:
        findings.append({"severity": "warning", "code": "YAW_CONTROL_LOG_MISSING",
                         "message": "No usable synchronized controller.cmd_yaw/controller.r_yaw data was available for causal yaw identification."})
    elif yaw_detection_enabled and (
        yaw_dynamics_fit["validation_r_squared"] is None
        or yaw_dynamics_fit["test_r_squared"] is None
        or yaw_dynamics_fit["validation_r_squared"] < 0.50
        or yaw_dynamics_fit["test_r_squared"] < 0.50
    ):
        findings.append({"severity": "warning", "code": "YAW_DYNAMICS_FIT_WEAK",
                         "message": "The causal yaw dynamics model did not generalize to held-out chirp data; do not enable active yaw response from this fit."})
    if yaw_detection_enabled and yaw_dynamics_fit is None and angular_fit is not None and (
        angular_fit["r_squared"][2] is None
        or angular_fit["r_squared"][2] < 0.30
    ):
        findings.append({"severity": "warning", "code": "ANGULAR_MODEL_FIT_WEAK",
                         "message": "Quiet-flight yaw excitation is insufficient or the yaw motor model fits poorly; do not enable active yaw response from this fit."})
    safety = nested_config.get("safety", {})
    onboard_state = state_source == "crazyflie_state_estimate"
    frame_limit = safety.get(
        "max_state_age_s" if onboard_state else "max_frame_age_s"
    )
    motor_limit = safety.get("max_motor_age_s")
    skew_limit = safety.get(
        "max_motor_state_skew_s" if onboard_state else "max_motor_pose_skew_s"
    )
    if frame_limit and summary["max_frame_age_s"] is not None and summary["max_frame_age_s"] > 0.8 * float(frame_limit):
        source_label = "onboard state" if onboard_state else "mocap"
        findings.append({"severity": "warning", "code": "STATE_AGE_NEAR_LIMIT" if onboard_state else "MOCAP_AGE_NEAR_LIMIT",
                         "message": f"Maximum {source_label} age {summary['max_frame_age_s']:.3f}s approached its {float(frame_limit):.3f}s limit."})
    if motor_limit and summary["max_motor_data_age_s"] is not None and summary["max_motor_data_age_s"] > 0.8 * float(motor_limit):
        findings.append({"severity": "warning", "code": "MOTOR_AGE_NEAR_LIMIT",
                         "message": f"Maximum motor-data age {summary['max_motor_data_age_s']:.3f}s approached its {float(motor_limit):.3f}s limit."})
    if skew_limit and summary["max_motor_pose_skew_s"] is not None and summary["max_motor_pose_skew_s"] > 0.8 * float(skew_limit):
        skew_label = "motor/state" if onboard_state else "motor/pose"
        findings.append({"severity": "warning", "code": "MOTOR_STATE_SKEW_NEAR_LIMIT" if onboard_state else "MOTOR_POSE_SKEW_NEAR_LIMIT",
                         "message": f"Maximum {skew_label} skew {summary['max_motor_pose_skew_s']:.3f}s approached its {float(skew_limit):.3f}s limit."})
    return summary, findings


def analyze_records(path: Path, records: list[dict[str, Any]], integrity: dict[str, Any]) -> dict[str, Any]:
    types = collections.Counter(record.get("type", "unknown") for record in records)
    frames = [
        record["data"].copy() for record in records
        if record.get("type") == "frames"
        and isinstance(record.get("data"), dict)
        and isinstance(record["data"].get("tvec"), list)
    ]
    frames = attach_onboard_velocities(frames, records)
    # A frame without either the legacy external KF velocity or a synchronized
    # onboard velocity cannot contribute to the speed/dropout analysis.
    frames = [frame for frame in frames if isinstance(frame.get("vel"), list)]
    states = [record for record in records if record.get("type") == "state"]
    commands = [record for record in records if record.get("type") == "commands"]
    events = [record for record in records if record.get("type") == "events"]
    config_record = next(
        (record for record in records if record.get("type") == "configs"
         and record.get("name") == "Translation Config"),
        None,
    )
    config = config_record.get("data", {}) if config_record else {}
    wrench_config_record = next(
        (record for record in records if record.get("type") == "configs"
         and record.get("name") in (
             "Wrench Interaction Config", "Onboard Wrench Interaction Config"
         )),
        None,
    )
    wrench_config = wrench_config_record.get("data", {}) if wrench_config_record else {}
    threshold = float(config.get("delta_v", 0.13))
    grace = float(config.get("Grace Period", 0.0))
    wait_time = next(
        (float(record["data"]["time"]) for record in events
         if record.get("name") == "Waiting For User Interaction"),
        None,
    )
    excitation_start_time = next(
        (float(record["data"]["time"]) for record in events
         if record.get("name") == "Wrench Calibration Excitation Started"),
        None,
    )
    sessions = interaction_sessions(records)

    # Brief, low-amplitude threshold crossings are normally point-cloud/KF
    # nuisance triggers rather than deliberate pushes.
    noise_peak_limit = max(0.2, 1.5 * threshold)
    for session in sessions:
        session["likely_noise_trigger"] = (
            session["duration_s"] < 0.15
            and session["push_event_count"] < 10
            and session["peak_event_speed_m_s"] < noise_peak_limit
        )
        if wait_time is not None:
            session["start_after_wait_s"] = session["start"] - wait_time
            session["end_after_wait_s"] = session["end"] - wait_time

    substantive = next((session for session in sessions if not session["likely_noise_trigger"]), None)
    baseline_end = substantive["start"] if substantive else (wait_time + 5.0 if wait_time else None)
    # A commanded identification manoeuvre is motion, not localization noise.
    # Restrict the quiet baseline to the interval before excitation starts.
    if excitation_start_time is not None and baseline_end is not None:
        baseline_end = min(baseline_end, excitation_start_time)
    excluded: list[tuple[float, float]] = [
        (session["start"] - 0.1, session["end"] + grace + 0.2)
        for session in sessions if baseline_end is not None and session["start"] < baseline_end
    ]
    quiet_frames = []
    if wait_time is not None and baseline_end is not None:
        quiet_frames = [
            frame for frame in frames
            if wait_time <= float(frame["time"]) < baseline_end
            and not any(lo <= float(frame["time"]) <= hi for lo, hi in excluded)
        ]
    if len(quiet_frames) < 20 and wait_time is not None:
        quiet_frames = [frame for frame in frames if wait_time <= float(frame["time"]) <= wait_time + 0.5]

    quiet_speeds = [magnitude(frame["vel"]) for frame in quiet_frames]
    noise = {
        "sample_count": len(quiet_frames),
        "speed_p50_m_s": percentile(quiet_speeds, 0.50),
        "speed_p95_m_s": percentile(quiet_speeds, 0.95),
        "speed_p99_m_s": percentile(quiet_speeds, 0.99),
        "speed_max_m_s": max(quiet_speeds) if quiet_speeds else None,
        "threshold_m_s": threshold,
        "threshold_margin_over_p99_m_s": (
            threshold - percentile(quiet_speeds, 0.99) if quiet_speeds else None
        ),
        "samples_over_threshold": sum(speed > threshold for speed in quiet_speeds),
        "position_span_m": [
            max(float(frame["tvec"][axis]) for frame in quiet_frames)
            - min(float(frame["tvec"][axis]) for frame in quiet_frames)
            for axis in range(3)
        ] if quiet_frames else None,
    }

    frame_times = [float(frame["time"]) for frame in frames]
    state_times = [time for time in map(record_time, states) if time is not None]
    frame_dts = [b - a for a, b in zip(frame_times, frame_times[1:]) if 0 < b - a < 1]
    last_frame_time = frame_times[-1] if frame_times else None
    last_state_time = max(state_times) if state_times else None
    cutoff = sustained_motor_cutoff(states)
    dropout_to_cutoff = (
        cutoff - last_frame_time if cutoff is not None and last_frame_time is not None else None
    )
    tracking = {
        "frame_count": len(frames),
        "duration_s": frame_times[-1] - frame_times[0] if len(frame_times) > 1 else None,
        "rate_hz": 1.0 / statistics.median(frame_dts) if frame_dts else None,
        "largest_frame_gap_s": max(frame_dts) if frame_dts else None,
        "last_frame_time": last_frame_time,
        "last_state_time": last_state_time,
        "telemetry_after_last_frame_s": (
            last_state_time - last_frame_time
            if last_state_time is not None and last_frame_time is not None else None
        ),
        "last_position_m": frames[-1]["tvec"] if frames else None,
        "last_velocity_m_s": frames[-1]["vel"] if frames else None,
        "last_speed_m_s": magnitude(frames[-1]["vel"]) if frames else None,
    }

    z_commands = [
        command["data"] for command in commands
        if command.get("name") == "Commander.send_zdistance_setpoint"
        and isinstance(command.get("data", {}).get("args"), list)
        and len(command["data"]["args"]) >= 2
    ]
    zero_horizontal = sum(
        abs(float(command["args"][0])) < 1e-9 and abs(float(command["args"][1])) < 1e-9
        for command in z_commands
    )
    # CommandWrapper timestamps are relative to its construction, while frame
    # and event timestamps are epoch seconds. The first position hold is issued
    # immediately after the Waiting event, giving us a reliable clock bridge.
    first_position_command_time = next((
        float(command["data"]["time"]) for command in commands
        if command.get("name") == "Commander.send_position_setpoint"
        and isinstance(command.get("data", {}).get("time"), (int, float))
        and float(command["data"]["time"]) < 1_000_000
    ), None)
    timed_z_commands: list[tuple[float, dict[str, Any]]] = []
    if wait_time is not None and first_position_command_time is not None:
        timed_z_commands = [
            (wait_time + float(command["time"]) - first_position_command_time, command)
            for command in z_commands
            if isinstance(command.get("time"), (int, float)) and float(command["time"]) < 1_000_000
        ]
    pushing_commands = [
        command for absolute_time, command in timed_z_commands
        if any(session["start"] <= absolute_time <= session["end"] for session in sessions)
    ]
    zero_pushing_commands = sum(
        abs(float(command["args"][0])) < 1e-9 and abs(float(command["args"][1])) < 1e-9
        for command in pushing_commands
    )
    before_dropout = [command for absolute_time, command in timed_z_commands
                      if last_frame_time is not None and absolute_time <= last_frame_time]
    dropout_to_cutoff_commands = [
        command for absolute_time, command in timed_z_commands
        if last_frame_time is not None and cutoff is not None and last_frame_time < absolute_time <= cutoff
    ]
    after_cutoff_commands = [command for absolute_time, command in timed_z_commands
                             if cutoff is not None and absolute_time > cutoff]

    def max_attitude(command_rows: list[dict[str, Any]]) -> list[float | None]:
        return [
            max((abs(float(command["args"][axis])) for command in command_rows), default=None)
            for axis in (0, 1)
        ]

    control = {
        "zdistance_command_count": len(z_commands),
        "zero_roll_pitch_command_count": zero_horizontal,
        "max_abs_roll_command_deg": max((abs(float(command["args"][0])) for command in z_commands), default=None),
        "max_abs_pitch_command_deg": max((abs(float(command["args"][1])) for command in z_commands), default=None),
        "pushing_zdistance_command_count": len(pushing_commands),
        "pushing_zero_roll_pitch_count": zero_pushing_commands,
        "max_abs_roll_pitch_before_dropout_deg": max_attitude(before_dropout),
        "max_abs_roll_pitch_dropout_to_cutoff_deg": max_attitude(dropout_to_cutoff_commands),
        "commands_after_motor_cutoff": len(after_cutoff_commands),
    }

    findings: list[dict[str, str]] = []
    if not integrity["closed_array"]:
        findings.append({"severity": "warning", "code": "UNCLOSED_LOG",
                         "message": "Log has no closing JSON array bracket; complete records were recovered."})
    margin = noise["threshold_margin_over_p99_m_s"]
    if margin is not None and margin < 0.03:
        findings.append({"severity": "warning", "code": "LOW_NOISE_MARGIN",
                         "message": f"Interaction threshold is only {margin:.3f} m/s above quiet p99 speed."})
    nuisance_count = sum(session["likely_noise_trigger"] for session in sessions)
    if nuisance_count:
        findings.append({"severity": "warning", "code": "NUISANCE_TRIGGERS",
                         "message": f"Detected {nuisance_count} brief, low-amplitude interaction trigger(s)."})
    telemetry_tail = tracking["telemetry_after_last_frame_s"]
    if telemetry_tail is not None and telemetry_tail > 0.25:
        findings.append({"severity": "critical", "code": "LOCALIZATION_DROPOUT",
                         "message": f"Mocap ended {telemetry_tail:.3f} s before flight-controller telemetry."})
    if dropout_to_cutoff is not None and 0 <= dropout_to_cutoff < 5:
        findings.append({"severity": "critical", "code": "MOTOR_CUTOFF_AFTER_DROPOUT",
                         "message": f"All motors cut out {dropout_to_cutoff:.3f} s after the last mocap frame."})
    if tracking["last_speed_m_s"] is not None and tracking["last_speed_m_s"] > 0.75:
        findings.append({"severity": "critical", "code": "DROPOUT_AT_HIGH_SPEED",
                         "message": f"Last mocap speed was {tracking['last_speed_m_s']:.3f} m/s, not a stationary loss."})
    if pushing_commands and zero_pushing_commands / len(pushing_commands) > 0.95:
        findings.append({"severity": "warning", "code": "ZERO_HORIZONTAL_PUSH_COMMANDS",
                         "message": f"{zero_pushing_commands}/{len(pushing_commands)} horizontal attitude commands during detected pushes were zero."})
    recovery_max = control["max_abs_roll_pitch_dropout_to_cutoff_deg"]
    if any(value is not None and value > 30 for value in recovery_max):
        findings.append({"severity": "critical", "code": "ATTITUDE_ESCALATION_AFTER_DROPOUT",
                         "message": f"After mocap stopped, commanded roll/pitch grew to {fmt(recovery_max[0])}°/{fmt(recovery_max[1])}° before motor cutoff."})
    if after_cutoff_commands:
        findings.append({"severity": "warning", "code": "COMMANDS_AFTER_MOTOR_CUTOFF",
                         "message": f"The RPi logged {len(after_cutoff_commands)} more setpoints after all motors were off."})
    if control["max_abs_pitch_command_deg"] is not None and control["max_abs_pitch_command_deg"] > 10:
        findings.append({"severity": "warning", "code": "LARGE_ATTITUDE_COMMAND",
                         "message": "Recovery emitted horizontal attitude commands above 10 degrees."})

    wrench, wrench_findings = analyze_wrench_records(records, wrench_config)
    findings.extend(wrench_findings)

    return {
        "source": str(path),
        "integrity": integrity,
        "record_count": len(records),
        "record_types": dict(types),
        "git_version": next((record.get("data", {}).get("version") for record in records
                             if record.get("type") == "git"), None),
        "translation_config": config,
        "wrench_config": wrench_config,
        "wrench": wrench,
        "wait_time": wait_time,
        "localization_noise": noise,
        "tracking": tracking,
        "motor_cutoff_time": cutoff,
        "dropout_to_motor_cutoff_s": dropout_to_cutoff,
        "control": control,
        "interaction_sessions": sessions,
        "findings": findings,
    }


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_report(reports: list[dict[str, Any]]) -> str:
    lines = ["# Interaction flight log analysis", ""]
    for report in reports:
        lines += [f"## {Path(report['source']).name}", ""]
        for finding in report["findings"]:
            lines.append(f"- **{finding['severity'].upper()} · {finding['code']}** — {finding['message']}")
        noise = report["localization_noise"]
        tracking = report["tracking"]
        lines += [
            "", "### Localization", "",
            f"Quiet speed p50/p95/p99/max: {fmt(noise['speed_p50_m_s'])} / "
            f"{fmt(noise['speed_p95_m_s'])} / {fmt(noise['speed_p99_m_s'])} / "
            f"{fmt(noise['speed_max_m_s'])} m/s; trigger threshold: {fmt(noise['threshold_m_s'])} m/s.",
            f"Quiet position span X/Y/Z: {noise['position_span_m']} m.",
            f"Mocap: {tracking['frame_count']} frames at {fmt(tracking['rate_hz'], 1)} Hz; "
            f"last speed {fmt(tracking['last_speed_m_s'])} m/s; telemetry continued "
            f"{fmt(tracking['telemetry_after_last_frame_s'])} s after mocap.",
            "", "### Interactions", "",
            "| # | start after wait (s) | duration (s) | events | peak speed (m/s) | likely noise |",
            "|---:|---:|---:|---:|---:|:---:|",
        ]
        for index, session in enumerate(report["interaction_sessions"], 1):
            lines.append(
                f"| {index} | {fmt(session.get('start_after_wait_s'))} | "
                f"{fmt(session['duration_s'])} | {session['push_event_count']} | "
                f"{fmt(session['peak_event_speed_m_s'])} | "
                f"{'yes' if session['likely_noise_trigger'] else 'no'} |"
            )
        lines += [
            "", "### Control and log integrity", "",
            f"Motor cutoff after last mocap: {fmt(report['dropout_to_motor_cutoff_s'])} s.",
            f"Z-distance commands: {report['control']['zdistance_command_count']} "
            f"({report['control']['zero_roll_pitch_command_count']} with zero roll/pitch); "
            f"maximum roll/pitch command: {fmt(report['control']['max_abs_roll_command_deg'])}° / "
            f"{fmt(report['control']['max_abs_pitch_command_deg'])}°.",
            f"During pushes, {report['control']['pushing_zero_roll_pitch_count']}/"
            f"{report['control']['pushing_zdistance_command_count']} Z-distance commands had zero roll/pitch. "
            f"Dropout-to-cutoff maximum: {report['control']['max_abs_roll_pitch_dropout_to_cutoff_deg']}; "
            f"commands after cutoff: {report['control']['commands_after_motor_cutoff']}.",
            f"Log valid JSON: {report['integrity']['valid_json']}; "
            f"closed array: {report['integrity']['closed_array']}; "
            f"discarded partial lines: {report['integrity']['discarded_lines']}.",
            "",
        ]
        wrench = report.get("wrench")
        if wrench is not None:
            lines += [
                "### External wrench observer",
                "",
                f"Mode: {'shadow' if wrench['shadow_mode'] else 'active'}; "
                f"state source: {wrench['state_source']}; "
                f"samples/calibrated/quiet: {wrench['sample_count']} / "
                f"{wrench['calibrated_sample_count']} / {wrench['quiet_accepted_sample_count']}; "
                f"measurement rejection: {fmt(wrench['measurement_rejection_fraction'])}.",
                f"Quiet absolute force p95 X/Y/Z: {wrench['quiet_abs_force_p95_N']} N; "
                f"p99: {wrench['quiet_abs_force_p99_N']} N.",
                f"Quiet absolute yaw torque p95: {wrench['quiet_abs_torque_p95_Nm'][2]} N·m; "
                f"p99: {wrench['quiet_abs_torque_p99_Nm'][2]} N·m.",
                f"Active samples — translation: {wrench['translation_contact_sample_count']}; "
                f"yaw: {wrench['yaw_contact_sample_count']}; "
                f"calibration excitation: {wrench['calibration_excitation_sample_count']}.",
                f"Yaw interaction detection: "
                f"{'enabled' if wrench['yaw_detection_enabled'] else 'disabled'}.",
                f"Maximum state/motor-data age and motor/state skew: "
                f"{fmt(wrench['max_frame_age_s'])} / "
                f"{fmt(wrench['max_motor_data_age_s'])} / "
                f"{fmt(wrench['max_motor_pose_skew_s'])} s; shadow command changes: "
                f"{wrench['shadow_command_change_count']}.",
                f"Causal yaw dynamics fit: {wrench['yaw_dynamics_fit']}.",
                f"Yaw motor-scale fit: "
                f"{None if wrench['angular_motor_scale_fit'] is None else {'estimated_yaw_accel_scale': wrench['angular_motor_scale_fit']['estimated_angular_accel_scale'][2], 'r_squared': wrench['angular_motor_scale_fit']['r_squared'][2], 'mixer_input_std': wrench['angular_motor_scale_fit']['mixer_input_std'][2]}}.",
                "",
            ]
    return "\n".join(lines)


def json_logs(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    controller_logs = sorted(
        path for path in input_path.glob("lb*.json")
        if "_tracker_" not in path.name
    )
    if controller_logs:
        return controller_logs
    return sorted(
        path for path in input_path.glob("*.json")
        if not path.name.startswith("analysis_report") and path.name != "metadata.json"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="session directory or controller JSON log")
    parser.add_argument("--output-dir", type=Path, help="default: INPUT/analysis")
    args = parser.parse_args()

    paths = json_logs(args.input)
    if not paths:
        parser.error(f"no controller JSON logs found in {args.input}")
    output_dir = args.output_dir or ((args.input if args.input.is_dir() else args.input.parent) / "analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for path in paths:
        records, integrity = load_records(path)
        reports.append(analyze_records(path, records, integrity))

    (output_dir / "analysis_report.json").write_text(
        json.dumps({"logs": reports}, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "analysis_report.md").write_text(markdown_report(reports), encoding="utf-8")
    print(markdown_report(reports))
    print(f"Reports written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
