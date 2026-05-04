import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from .vicon_scanner import ViconScanner, mock_scan, sort_and_match, ViconConnectionError, InsufficientMarkersError

logger = logging.getLogger(__name__)


def _run_ui_subprocess(assignments, outliers, mission_data):
    """Run Tk in a plain Python subprocess and receive the result via JSON files."""
    ui_script = Path(__file__).resolve().with_name("ui.py")

    with tempfile.TemporaryDirectory(prefix="dispatcher_ui_") as tmp_dir:
        tmp_path = Path(tmp_dir)
        input_path = tmp_path / "input.json"
        output_path = tmp_path / "output.json"

        payload = {
            "assignments": assignments,
            "outliers": outliers,
            "mission": mission_data,
        }
        input_path.write_text(json.dumps(payload), encoding="utf-8")

        cmd = [sys.executable, str(ui_script), str(input_path), str(output_path)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            stdout, stderr = proc.communicate()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise

        if stdout.strip():
            logger.debug(stdout.strip())
        if stderr.strip():
            logger.warning(stderr.strip())

        if proc.returncode != 0:
            logger.warning(f"Dispatcher UI process exited with code {proc.returncode}.")
            return None, None

        if not output_path.exists():
            logger.warning("Dispatcher UI exited without writing a result.")
            return None, None

        try:
            result = json.loads(output_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            logger.warning(f"Dispatcher UI wrote invalid JSON: {e}")
            return None, None

        if not result.get("confirmed"):
            return None, None

        return result.get("assignments"), result.get("extra_params") or {}

def run_dispatch(manifest_drones, mission_data, vicon_host="192.168.1.39", mock=False):
    """
    Scans Vicon (or mocks), matches sorted points to sorted manifest positions,
    shows the UI for confirmation/editing, and returns the updated drone manifest array 
    where 'init_pos' equals their physical Vicon coordinate.
    """
    n_drones = len(manifest_drones)
    
    if mock:
        logger.info(f"Using mock scanner for {n_drones} drones.")
        vicon_points = mock_scan(manifest_drones)
    else:
        logger.info(f"Connecting to Vicon scanner at {vicon_host} required {n_drones} markers...")
        scanner = ViconScanner(vicon_host)
        try:
            # We use 5 frames so that it averages and confirms stability
            vicon_points = scanner.scan(n_frames=5, n_drones=n_drones)
        except (ViconConnectionError, InsufficientMarkersError) as e:
            logger.error(str(e))
            logger.error("Falling back to MOCK scanner due to Vicon failure.")
            vicon_points = mock_scan(manifest_drones)

    # Generate initial sorted assignments
    assignments, outliers = sort_and_match(vicon_points, manifest_drones)
    
    # Run UI in an isolated subprocess so Tcl/Tk owns a clean top-level process.
    logger.info("Opening Dispatcher UI for confirmation...")
    final_assignments, extra_params = _run_ui_subprocess(assignments, outliers, mission_data)

    if not final_assignments:
        logger.warning("Dispatcher UI aborted or skipped. No Vicon coordinates will be updated.")
        return manifest_drones
    
    # Update manifest drones with their confirmed real coordinates
    # We will build a dictionary to preserve the exact properties of the original drones
    drones_by_id = {d['id']: d for d in manifest_drones}
    updated_drones = []
    
    for a in final_assignments:
        d_id = a['id']
        drone = drones_by_id[d_id]
        # Replace init_pos with real matched coordinate
        drone['init_pos'] = a['vicon_pos']
        if extra_params:
            if extra_params.get('viewpoint'):
                vp = extra_params['viewpoint']
                offset_x, offset_y, offset_z = -0.025, 0.0, 0.005
                drone['viewpoint'] = [vp[0] + offset_x, vp[1] + offset_y, vp[2] + offset_z]
            if extra_params.get('anchor'):
                drone['anchor'] = extra_params['anchor']
        updated_drones.append(drone)
        
    return updated_drones
