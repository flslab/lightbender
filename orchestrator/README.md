# Swarm Orchestrator

The Swarm Orchestrator manages the LightBender drone swarm. It handles executing SFL files, managing network communications, interacting with camera and radio nodes, downloading execution logs, and automatically uploading data after experiments.

## Setup

### 1. Install Dependencies
Ensure you have the required Python dependencies installed.
```bash
pip install -r requirements_orchestrator.txt
```

### 2. Configure the Swarm Manifest
The orchestrator relies on `swarm_manifest.yaml` to configure the network, drones, and mission files.

A sample template is provided as `swarm_manifest_sample.yaml`. To get started:
```bash
cp swarm_manifest_sample.yaml swarm_manifest.yaml
```

**Adjusting the manifest:**
Open `swarm_manifest.yaml` and update the following settings to match your environment:
- **`controller`**: Set your local machine's `ip`. You can define `mission_path` (e.g., `"SFL"`) and list out multiple `mission_files` to run sequentially.
- **`drones`**: Define the drones participating in the swarm. For each drone, make sure the `ip`, `uri`, hardware `type` (H or V), and `servo_offsets` match your actual hardware configurations.
- **`camera_node` / `radio_node`**: Provide the corresponding IP addresses and usernames for the remote nodes if you are using them for recording or CrazyRadio communication.

### 3. Uploader Setup
The orchestrator automatically uploads logs and experiment data (like recorded videos and drone logs) to Google Drive via the `uploader` module once a mission concludes.

Before running an experiment, you need to configure the uploader credentials:
1. A `.env.example` file is provided in the `uploader` directory. Copy it to create your `.env` file:
   ```bash
   cp uploader/.env.example uploader/.env
   ```
2. Edit `uploader/.env` and fill in:
   - `GOOGLE_CLIENT_SECRET`: Path to your downloaded OAuth client secret JSON file.
   - `GDRIVE_FOLDER_ID`: The ID of the shared Google Drive folder where data should be uploaded to.
   If you need to generate a new OAuth client secret JSON file, follow the instructions here: https://developers.google.com/workspace/guides/create-credentials

## Usage

Run the main script using Python. The orchestrator has a number of available execution modes.

```bash
python orchestrator.py [OPTIONS]
```

### Important Runtime Flags:
- **`--illumination`**: Run an illumination application mission.
- **`--interaction`**: Run an interaction application mission.
- **`--active-septic-brake-test`**: One-flag, single-flight hardware entry
  point for the active seventh-order release brake. It selects interaction,
  potentiometer sensing, and rigid-body XYZ-only Vicon, then requires the
  mission's current quality-gated planar braking fit before the Pi can arm.
  It does not run calibration, relax a gate, or enable the diagnostic firmware
  release event as a control source. Do not add `--skip-confirm` or `--radio`.
  A plain XYZ `--calibrate` result is not sufficient.
- **`--active-septic-brake-calibration`**: Separate, single-flight entry for
  the fixed-pulse planar fit required by the test. It reuses an existing XYZ
  fit and skips the XYZ excitation stage; stationary observer bias collection
  remains. It uses rigid-body XYZ-only Vicon and the mission's configured ±Y
  sweep, with adaptive braking off. The current battery-limited dedicated
  sweep has six pulses (three per direction); fit-quality gates are unchanged.
  Recovery holds the measured end position, then moves a bounded target back
  to center. If it is not back within the scheduled 5 seconds, it keeps
  returning under the same speed/displacement/tilt limits; the next pulse
  requires 2 seconds of stable position, speed, and tilt. An additional
  12-second wait timeout aborts rather than extending flight indefinitely.
  It does not start the interaction test after landing. A failed fit is not
  saved as usable flight authority.
- **`--vicon-rigidbody-position-only`**: With `--interaction` or `--calibrate`,
  use each selected drone's `obj_name` (or `label`) from `swarm_manifest.yaml`
  as a Vicon rigid body. The controller logs its orientation but sends only
  XYZ (`extpos`) to the flight controller. This skips pointcloud Dispatcher
  matching and does not enable the separate contact-attitude Run 2 experiment.
- **SFL potentiometer selection**: `--interaction` reads the selected SFL's
  `Interaction.config` before launching the drone. For `behavior: level_coast`,
  `detection_method: potentiometer` automatically starts the sensor;
  `model` and `vel` do not start it by default. Set
  `Interaction.config.record_potentiometer: true` (default `false`) to start it
  with **any detector** and save every valid UART sample to the flight JSON as
  `potentiometer_raw`, including `compression_mm`, `force_n`, raw ADC and device/
  host timestamps. `potentiometer_recording` metadata saves the calibration
  curve and spring coefficient (`force_n = compression_mm * k`, default
  `k = 0.16 N/mm`). This estimates spring force; recording does not change the
  selected contact/release detector. No `sensing` field or manual `--sense`
  flag is needed for level-coast; legacy values of either cannot override this
  selection. Deploy the corresponding offboard logging code before using the
  new option. Contact-validation capture shares the same raw-sample callback,
  so enabling both does not duplicate samples.
  Separate sensor-backed hardware experiments still require their sensor.
  Use `detection_method: potentiometer # potentiometer | model | vel` in the
  SFL and remove `level_coast.detector`. The onboard momentum observer and
  synchronized logs remain enabled for every level-coast detection method.
  Standard interactions also
  enable it for `virtual_object.contact_detection.source: potentiometer` or
  `virtual_object.release_behavior.mode: potentiometer_coast`.
  For standard interactions, optional potentiometer comparison data can still
  be enabled with `Interaction.config.sensing: true`;
  this does not change the detector. `sensing: false` disables optional
  acquisition and overrides the legacy CLI flag, but conflicts with a
  sensor-dependent detector/release/experiment or `record_potentiometer: true`
  and fails before launch. `record_potentiometer` also works for standard
  interactions without requiring the legacy `sensing` field.
  Calibration, braking tests, baseline, and hover do not inherit these
  interaction sensing settings. For standard interactions the old `--sense`
  still enables optional acquisition if `sensing` is omitted. The orchestrator
  internally supplies
  the controller's existing `--sense`; sensor startup/freshness gates remain.
  Launch the current SFL with `python orchestrator.py --interaction` from the
  orchestrator directory, retaining any other transport/recording options.
- **`--contact-attitude-run {1,2,3}`**: With `--interaction`, select
  the validated three-flight estimator comparison. Run 1 uses pointcloud plus
  an exact onboard-EKF mirror, run 2 uses rigid-body XYZ only, and run 3 uses
  rigid-body full extpose. The shadow remains diagnostic-only in all runs.
  Runs 2 and 3 use the manifest's fixed drone-to-rigid-body identity and skip
  the unlabeled-marker Dispatcher; the configured rigid body must still be
  visible in the Vicon DataStream before the controller can arm.
  Runs 2 and 3 require a Crazyflie firmware image built with
  `crazyflie-firmware/configs/contact_release_shadow.conf`; pre-arm fails closed if its
  producer-latched IMU/state timestamps are unavailable or untrusted.
- **`--calibrate`**: Run the contact-free XYZ wrench-model calibration. This
  saves per-drone parameters to `Interaction/wrench_calibration.json`; use it
  separately from `--interaction`. Plain calibration does not run planar
  attitude/braking trials or online prediction fitting.
- **`--calibrate-estimator-imu`**: Run separate motors-off static IMU
  calibration interactively on one Pi. Default `gravity_norm` allows cage
  support with unknown exact orientation. This branches before swarm
  setup: no Dispatcher, Vicon forwarding, camera, flight-controller reboot,
  flight handshake, or takeoff. Original `--calibrate` behavior is unchanged.
  Run from this directory:

  ```bash
  python orchestrator.py --calibrate-estimator-imu --drone-id lb11
  ```

  If the selected mission references exactly one aircraft, `--drone-id` may be
  omitted. Otherwise it is required. The selected Pi/SSH user and offboard
  environment come from `swarm_manifest.yaml`; a named drone does not require
  an SFL mission. The default device connection is `usb://0`, matching ordinary
  launches. Add `--radio` to use that aircraft's manifest radio URI instead.

  Firmware build/tag, fixture identity, and reference description default to
  `unknown` without questions. Optionally supply `--imu-firmware-id`,
  `--imu-fixture-id`, and `--imu-reference-note` when known.
  Remove props. The Pi guides 18 different training poses + 6 NEW validation
  poses; press Enter at each pose. Exact angles are not required. Use different
  stable cage supports/tilts covering all directions. Poses within 8 degrees
  of an earlier pose prompt a retry. The result fits axis scale and bias;
  sensor/body rotation and cross-axis terms remain uncalibrated. Add
  `--imu-calibration-method six_face` only with an exact body-aligned fixture.
  There is no typed props-off confirmation. Prompts remain interactive over SSH;
  do not run this mode in the background. `--imu-duration-s` and `--imu-settle-s`
  override the 4-second recording and 2-second settling defaults.

  Data, fit reports, accepted calibration (if any), and terminal transcript are
  downloaded to `logs/estimator_imu_<drone>_<session>/`. Failed calibration also
  retrieves the available raw data. `launch_result.json` records missing files
  and transfer errors; originals remain on the Pi in
  `Interaction/estimator_calibrations/<drone>/<session>/`. A successful result
  requires all outputs, a successful fit and a matching dataset fingerprint.

  Both IMU entries synchronize the dedicated runtime sources from the local sibling
  offboard checkout before launch. Pi originals are backed up under
  `.imu_runtime_backups/`; syntax and SHA-256 checks run before replacement.
  A running controller blocks synchronization. No Git reset or firmware change
  occurs, and unrelated files/calibration artifacts are retained.
  It saves a checked candidate; it does **not** load corrections into estimator 3
  or identify Vicon delay. The offboard guide
  `Interaction/ESTIMATOR_IMU_CALIBRATION.md` describes fixture accuracy and limits.

  Cage IMU calibration and automatic offline estimator-3 validation:

  ```bash
  python orchestrator.py --calibrate-estimator-imu --drone-id lb11 --imu-auto-flight
  ```

  No flight YAML or interaction mission is needed. The selected aircraft's
  existing manifest `init_pos` selects its marker. The fixed trajectory hovers
  0.6 m above the measured start and moves 20 cm in ±X/±Y. Vicon host and battery
  threshold come from the Pi's existing `Interaction.config`. The local abort
  fence is ±0.6 m XY and from 0.05 m below to 1 m above the measured start.
  Keep this space clear. Single-marker pointcloud tracking needs no rigid body.
  Full gyro axis/scale fitting is optional: add `--imu-calibrate-gyro` for
  12 known fixed-axis ±90° fixture turns (six training + six validation).
  Without it, static gyro bias is still fitted. Keep props off until fitting
  finishes. Refit props, clear the area and use the
  local Dispatcher point selection and standard launch confirmation. The
  standard controller handles localization, PID setup, takeoff and landing;
  the airborne hover/±X/±Y task samples IMU, Vicon
  position and the default estimator. **Estimator 3 stays disabled onboard.**

  To fly without recollecting static poses, use
  `python orchestrator.py --validate-estimator-imu --drone-id lb11`.
  It selects this aircraft's latest successful local fixture fit automatically;
  rejected fits and flight-only copies are skipped. An explicit session directory
  remains optional after the flag. Results are saved in a new session. Keep the
  aircraft still on the floor while gyro zero refreshes before READY. The standard
  validation entry can reuse the accelerometer fit after reboot; the original fit
  is preserved and a fresh clock reference is recorded with the new gyro zero.
  Both validation-only and calibration-plus-auto-flight use the standard
  automatic reboot through the configured radio node before the Pi controller
  starts. `--skip-reboot` skips it only when explicitly requested.
  This entry runs the normal Dispatcher point-selection UI, READY/START and local
  launch confirmation. The controller runs detached on the Pi, with its terminal
  output in `drone_lb11.log`. Captures are fetched through the normal download queue
  into `logs/<mission_tag>/imu_validation/`, then replayed offline. The usual automatic
  firmware reboot is skipped to retain the static gyro bias session. No new YAML
  is required; the built-in IMU task is served as a temporary mission.

  After landing/download/fingerprint checks, estimator 3 runs locally through
  the frozen real firmware C kernel with raw and corrected IMU inputs. The
  default estimator is the relative flight reference, not independent truth.
  A local sibling offboard checkout and C compiler are needed; no Pi compiler
  is needed. Reports and CSV are in `flight/replay/`. The standard controller
  path records ordinary IMU logs at 100 Hz and does not enable the RAM recorder.
  This path needs no firmware change.
  Log-rate replay and reported resets do not establish bit-exact onboard
  behavior or active estimator-3 control accuracy. The gyro stage needs known
  mechanical axes and angle stops; freehand/default-estimator-derived angles
  are not calibration truth. See the offboard calibration guide for all limits.

- **`--no-adaptive-braking-calibration`**: Keep every calibration brake pulse
  at its fixed configured duration. Use this for a directly comparable
  baseline or when the online prediction dependencies are unavailable.
- **`--adaptive-braking-calibration`**: Explicitly add the model-guided planar
  attitude/braking stage.
- **`--targeted-braking-calibration`**: With `--calibrate`, run the explicit
  high-speed data protocol: 20 degrees, 0.32 s acceleration, 0.16/0.20/0.24 s
  braking, two repetitions in each ±Y direction (12 trials total). This mode
  disables adaptive pulse replacement so the requested durations stay fixed.

- **`--braking-test`**: Run only the attitude-braking repeat trials described
  below. It skips XYZ excitation and preserves the saved calibration. Do not
  combine it with `--interaction`, `--calibrate`, or `--sense`.
- **`--braking-test-direction {both,positive-y,negative-y}`**: Select trial
  direction; requires `--braking-test`. Defaults to both, alternating -Y first.
- **`--braking-test-repetitions {1,2,3}`**: Number of trials per selected
  direction; requires `--braking-test`. Defaults to three per direction.
- **`--mpc`**: Run the dedicated Learning-MPC bootstrap collection. It starts
  a contact-free automatic world ±Y acceleration protocol, configures a 100 Hz
  Crazyflie state stream and a separate 50 Hz decision grid, and keeps all LMPC
  online command authority disabled. No potentiometer is used. Do not add
  `--sense`, another flight mode, or a ground/droneless shortcut.

Adaptive prediction validation is saved independently from the legacy
fixed-pulse planar-fit quality gate. If the adaptive pulse schedule fails that
legacy gate, the controller may preserve an already valid planar fit while
saving the newly validated prediction model. A first calibration still needs a
valid fixed baseline, obtainable with `--active-septic-brake-calibration`.
Adaptive candidates must also predict an endpoint within 0.05 m/s of zero
projected velocity and 3 degrees of level attitude; both are hard gates.

Run the targeted protocol only after confirming the same clear calibration
volume and localization health used for the existing 20-degree trials:

```bash
python3 orchestrator.py --calibrate --targeted-braking-calibration --skip-record
```

This is a real flight command. It does not enable the experimental residual
observer or normal interaction control; it only collects the deterministic
high-speed braking-duration sweep and runs the existing online fit/report path.
- **`--morphing`**: Run the mission with morphing algorithms enabled.
- **`--radio`**: Connect to drones over CrazyRadio (commands forwarded via the radio node).
- **`--ground`**: Perform a ground test without making the drones take off (useful for testing LED interactions or servos).
- **`--dark`**: Optimize camera parameters for recording in darkness.

### Control & Override Flags:
- **`--off`**: Cleanly power off / shut down all Raspberry Pis on the drones.
- **`--kill`**: Terminate the active Python controller processes on all drones.
- **`--skip-record`**: Run the mission without triggering the camera node.
- **`--record`**: Run the camera script strictly to record (no drone operation).
- **`--skip-confirm`**: Bypass manual confirmation prompts during the initialization sequence.

### Example Experiment

Running a standard illumination mission:
```bash
python orchestrator.py --illumination
```
When executed, the orchestrator will:
1. Parse the configurations and mission files from `swarm_manifest.yaml`.
2. Connect to the drones and issue restart commands.
3. Automatically launch the remote scripts and await "READY" payloads.
4. Prompt you (if not skipped) to launch the swarm.
5. Manage flight state logic and await "LANDED" confirmations.
6. Gather logs locally and trigger the `uploader` to save them seamlessly to Google Drive.

## Standalone attitude-braking repeat test

Use a controller checkout that supports `--braking-test`. In
`swarm_manifest.yaml`, keep `controller.mission_path: ../Interaction/SFL` and
`controller.mission_file: translation_inertia.yaml`. From the LightBender repo:

```bash
cd orchestrator
python3 orchestrator.py --braking-test --skip-record
```

This is a **real flight command**, with the normal launch confirmation. It
automatically forwards `--log --smooth-controller-rate 100 --cf-log-period 10`.
No potentiometer/Arduino input is needed; omit `--sense`. Omit `--skip-record`
only when video recording is also wanted and the camera node is available.

The controller uses the calibration nominal hover position and performs six
contact-free trials, alternating **-Y, +Y, -Y, +Y, -Y, +Y**. Each uses a 20-degree
acceleration pulse for 0.24 s, level for 0.20 s, opposite 20-degree braking for
0.24 s, then 0.65 s of level observation before position recovery. Trial
readiness and existing flight boundary/localization protections remain active.
Confirm the same clear test volume used for calibration before launching.

To replace one invalid **+Y** sample with a single trial of the identical pulse
protocol, run:

```bash
python3 orchestrator.py --braking-test --braking-test-direction positive-y --braking-test-repetitions 1 --skip-record
```

This is also a real flight command. It changes only the selected direction and
trial count, not the pulse timing, angle, readiness checks, or safety limits.
`positive-y` means acceleration along the existing experiment's +Y direction;
the braking pulse is opposite. Use `negative-y` for a single -Y trial. Omitting
both selectors preserves the original six-trial sequence. Fix localization
dropouts before collecting a replacement sample; the selectors do not bypass
localization validity checks.

Unlike `--calibrate`, this repeat test does not run the XYZ excitation, fit a
replacement flight calibration, or overwrite `Interaction/wrench_calibration.json`.
Raw trial data and the exact overridden protocol are logged for independent
offline validation. The mission YAML itself is not rewritten. Downloaded logs
remain under `orchestrator/logs/translation_inertia_<timestamp>/`; use the test
mode/protocol recorded inside the flight log to distinguish them from normal
interaction and full-calibration runs.

## Offline Learning-MPC bootstrap collection

Use a controller checkout with `--mpc` support and keep
`translation_inertia.yaml` selected in `swarm_manifest.yaml`. From this
directory, the requested one-flag entry point is:

```bash
python3 orchestrator.py --mpc
```

This is a **real flight command**. Add `--skip-record` only if camera recording
is not wanted. Before arming, the drone checks that its saved planar braking
fit is current, both +Y and -Y prediction models passed independent held-out
validation, nominal body Y is aligned with world Y, and the calibration hover
point has at least 0.70 m XY boundary margin (0.60 m maneuver budget plus a
0.10 m reserve). It freezes each direction's exact model, delay, state layout,
and fingerprint for later offline replay; the legacy planar controller still
owns braking. A failed check aborts before takeoff.

The drone settles at the calibration nominal position, sends at least 0.10 s
of level z-distance commands, then applies a calibrated 8-degree attitude along
world +Y or -Y. Fresh measured velocity—not elapsed pulse time—starts legacy
attitude-coast braking within 0.02 m/s of the 0.25, 0.45, and 0.65 m/s targets.
The protocol collects two terminal-success samples per direction at each speed
and completes both signs at a lower speed before advancing to the next tier.
No push, release, potentiometer, or `--sense` input is needed.

Before every attempt, speed must be at most 0.03 m/s, XY and Z position errors
must each be at most 0.05 m, and the ready state must remain valid for 0.30 s
(5 s timeout). A state gap above 0.10 s, acceleration longer than 0.85 s,
maneuver displacement above 0.60 m, remaining boundary margin below 0.10 m,
tilt above 12 degrees, roll/pitch rate above 100 deg/s, or wrong-way speed above
0.05 m/s fails closed. The automatic prelude then either gives a still-trusted
moving state to legacy attitude-coast braking or sends level and enters the
normal landing path; it never continues a stale tilt command.

The private overlay suppresses duplicate-state command sends and hands position
control the measured terminal position only after the full measured terminal
gate and command-delay queue settle. It then returns to the nominal position
and repeats the readiness dwell before another attempt. It never fits or saves
the old calibration and never lets LMPC own a flight command.

Completion in the flight log means only raw speed-cell coverage. Download the
complete JSON, run the offline decision-time resampler, then the strict replay
validator before creating a safe-set artifact. Failed/reversed/stale/skewed or
off-axis attempts remain in the raw log but do not count.

## Mock Blender Flow

To emulate an illumination launch without booting the real swarm, run:

```bash
python mock_orchestrator.py
```

This mock client connects to the Blender add-on's swarm monitor socket on `127.0.0.1:5598`, sends the same newline-delimited JSON status messages as the real orchestrator, and pauses before each message group so you can confirm the next step from the CLI.

Helpful flags:
- `--drone-id lb1 --drone-id lb2`: limit the mock run to specific drones.
- `--manifest path/to/swarm_manifest.yaml`: use a different manifest.
- `--message-delay 0.0`: send each group immediately with no spacing between messages.
- `--auto-cleanup-on-stop`: if Blender sends `stop`, automatically follow with `logs_fetched` and `all_stopped`.

## Analyze an interaction flight

The session analyzer accepts either an orchestrator session directory or one
drone JSON log. It recovers complete records from logs left unterminated by a
crash, measures the quiet localization-noise margin, segments interaction
triggers, and checks for mocap dropout, motor cutoff, and extreme recovery
commands. For model-based interaction logs it also reports observer rejection,
quiet force/torque residuals, contact-channel activity, telemetry freshness,
shadow-mode command invariants, and a diagnostic angular motor-scale fit.

```bash
python3 orchestrator/analyze_interaction_logs.py \
  orchestrator/logs/translation_2026-07-24_14-53-43
```

It prints a diagnosis and writes machine-readable `analysis_report.json` plus
`analysis_report.md` under the session's `analysis/` directory. Use
`--output-dir PATH` to put the reports elsewhere.

IMU calibration metadata (`--imu-firmware-id`, `--imu-fixture-id`, `--imu-reference-note`)
is optional and defaults to `unknown` without prompting. Props-off, pose sampling
and physical flight-readiness confirmations remain part of the measurement protocol.
