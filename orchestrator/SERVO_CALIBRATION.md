# Servo calibration accuracy

Enable with `--illumination --calibrate-servos`. Calibration uses the first
SFL servo row at the target handshake. The default tolerance is 0.75 degrees.
Two consecutive independent camera measurements must fall inside tolerance;
each camera measurement requires five consistent fresh frames. Explicit
manifest values override defaults; update older `tolerance_deg: 2.0` entries.

All rods remain under observation while their drone is being calibrated.
A previously converged rod resumes correction if it drifts outside tolerance,
and a missing measurement clears its confirmation count. Summaries always
show the latest measurement, including missing values.

After all drones finish initial calibration, a mandatory verification sweep
remeasures each drone and corrects any new error. Each pass allows up to ten
measurement attempts per drone. The entire phase still has a 60-second budget,
servo travel/offset bounds, stall detection, battery checks, rollback and the
configured abort/continue policy. A failed initial pass does not spend another
pass retrying a stalled or limited servo. With the default abort policy, any
unverified or inaccurate rod prevents illumination START.

Relevant manifest settings:

```yaml
servo_calibration:
  enabled: true
  camera_id: CAM_Front
  calibration_file: /home/fls/fls-cf-offboard-controller/config/rgb_camera.json
  tolerance_deg: 0.75
  max_iterations: 10
  consensus_frames: 5
  max_spread_deg: 1.0
  confirmation_measurements: 2
  verification_passes: 1
  timeout_s: 60.0
  failure_policy: abort
```

Observations in `logs/<tag>/servo_calibration.json` identify their pass and
initial/verification phase. The final rod values are the latest measured
angles, not the first values accepted during initial calibration. These are
camera-based measurements; tighter acceptance does not remove camera-model
bias or guarantee sub-degree physical accuracy. Calibration can take longer
or fail when the requested tolerance cannot be confirmed.

## Saved results

After calibration and cleanup, the orchestrator atomically updates the local
`swarm_manifest.yaml`. Verified offsets are stored in each participating
drone's `servo_offsets` and become the starting offsets on the next launch.
Failed, unverified, or unacknowledged results never replace previous offsets.
`servo_calibration_result` stores the run tag, verification status, latest
measured angles/errors, and saved offsets, including failure information.

Comments and unrelated settings are preserved. If servo offsets were edited
during calibration, the save is refused rather than overwriting the edit.
Save failures are logged and included as `manifest_save_error` in the run's
calibration JSON; they do not discard the calibration recording or results.
