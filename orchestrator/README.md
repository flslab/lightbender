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
