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
- **`common.localizer_work_dir`**: Optional path to the marker-localization checkout on each drone; it defaults to `/home/fls/fls-marker-localization`. When that Git checkout exists, drone boot pulls it with a fast-forward-only update and runs its incremental high-rate release build before launching the controller.
- **`drones`**: Define the drones participating in the swarm. For each drone, make sure the `ip`, `uri`, hardware `type` (H or V), and `servo_offsets` match your actual hardware configurations.
- **`camera_node` / `radio_node`**: Provide the corresponding IP addresses and usernames for the remote nodes if you are using them for recording or CrazyRadio communication.
- **`marker_grid_node`**: Define the dedicated Raspberry Pi Zero W, its marker-grid checkout/virtual environment, grid JSON, UDP port, and GPIO levels. Set optional `hypergrid_tiles` to a list of `[i, j]` coordinates to light only those tiles' HyperGrid LEDs; omit it to light all HyperGrid tiles (an empty list lights none). The orchestrator validates and launches this node before it launches any drone. A drone that owns a MyGrid tile declares `marker_tile: [i, j]`.

The marker grid is deliberately not a drone peer. A drone requests a state
change over its existing ZMQ connection by sending
`{"id":"lb1","status":"MARKER_GRID_MODE","mode":"off"}`. The orchestrator
maps the drone ID to its manifest-owned tile and forwards an idempotent UDP
command to the grid. Modes are `blink`, `static`, and `off`; requests without a
drone target are reserved for orchestrator-wide lifecycle changes.
During cleanup, the orchestrator terminates the remote marker-grid controller;
its signal handler clears every LED channel before the process exits.

To test the physical marker grid without launching a mission, run
`python orchestrator.py --test-marker-grid`. The orchestrator turns on the
HyperGrid and all MyGrids, then prompts for Enter before turning each MyGrid
off and back on in snake order. Tile coordinates are `(i, j) = (x, y)`: tiles
with the same `i`, such as `(-1, -1)` and `(-1, 0)`, share an x row and are
adjacent along y. The controller is stopped when the test ends.

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
- **`--calibrate-estimator-imu`**: Run the separate motors-off six-face IMU
  fixture calibration interactively on one Pi. This branches before swarm
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

  Before connecting, the terminal prompts for the **actual flashed firmware**
  build/tag, fixture identity, and independent reference description. You can
  also supply `--imu-firmware-id`, `--imu-fixture-id`, and `--imu-reference-note`.
  The Pi then asks for props-off confirmation and guides six training and six
  independently reseated validation poses. Prompts remain interactive over SSH;
  do not run this mode in the background. `--imu-duration-s` and `--imu-settle-s`
  override the 4-second recording and 2-second settling defaults.

  Data, fit reports, accepted calibration (if any), and terminal transcript are
  downloaded to `logs/estimator_imu_<drone>_<session>/`. Failed calibration also
  retrieves the available raw data. `launch_result.json` records missing files
  and transfer errors; originals remain on the Pi in
  `Interaction/estimator_calibrations/<drone>/<session>/`. A successful result
  requires all outputs, accepted fit and a matching dataset fingerprint.

  Update both this orchestrator checkout and the Pi's offboard checkout before
  using the entry. This mode does not automatically pull or alter remote code.
  It saves a checked candidate; it does **not** load corrections into estimator 3
  or identify Vicon delay. The offboard guide
  `Interaction/ESTIMATOR_IMU_CALIBRATION.md` describes fixture accuracy and limits.
  Do not combine this flag with `--calibrate`, `--interaction`, other flight or
  sensing modes, or `--skip-confirm`.
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
