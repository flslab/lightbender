"""Independent fixture IMU calibration with opt-in bounded flight capture."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import subprocess
import uuid

import yaml


OPTIONS = ('drone_id', 'imu_firmware_id', 'imu_fixture_id', 'imu_reference_note', 'imu_calibration_method',
           'imu_duration_s', 'imu_settle_s', 'imu_flight_config')
ARTIFACTS = ('dataset.json', 'packets.jsonl', 'fit/report.json', 'fit/report.md',
             'fit/calibration.json')
FIT_ARTIFACTS = ('dataset.json', 'packets.jsonl', 'fit/report.json', 'fit/report.md',
                 'fit/candidate.json')
FLIGHT_ARTIFACTS = ('dataset.json', 'packets.jsonl', 'fit/report.json', 'fit/report.md',
                    'fit/candidate.json', 'flight/config.json', 'flight/report.json',
                    'flight/report.md', 'flight/packets.jsonl')


def validate_estimator_calibration_options(args):
    enabled = getattr(args, 'calibrate_estimator_imu', False)
    if not enabled:
        if (any(getattr(args, field, None) is not None for field in OPTIONS)
                or getattr(args, 'imu_auto_flight', False) or getattr(args, 'imu_calibrate_gyro', False)):
            raise ValueError('--drone-id and --imu-* options require --calibrate-estimator-imu')
        return
    if getattr(args, 'imu_flight_config', None) is not None and not getattr(args, 'imu_auto_flight', False):
        raise ValueError('--imu-flight-config requires --imu-auto-flight')
    conflicts = (
        'interaction', 'calibrate', 'braking_test', 'mpc', 'baseline', 'hover',
        'active_septic_brake_test', 'active_septic_brake_calibration',
        'illumination', 'intractable_illumination', 'morphing', 'sense',
        'ground', 'droneless', 'record', 'loadcell', 'blender', 'skip_confirm',
        'targeted_braking_calibration', 'vicon_rigidbody_position_only',
        'vicon_source_metadata', 'test_marker_grid',
    )
    if (any(getattr(args, field, False) for field in conflicts)
            or any(getattr(args, field, None) is not None for field in (
                'adaptive_braking_calibration', 'braking_test_direction',
                'braking_test_repetitions', 'contact_attitude_run', 'kill', 'off'))):
        raise ValueError('--calibrate-estimator-imu is a standalone calibration workflow; '
                         'do not combine it with flight, sensor, shutdown or skip-confirm options')
    for field, low in (('imu_duration_s', 3), ('imu_settle_s', 1)):
        value = getattr(args, field, None)
        if value is not None and (not math.isfinite(value) or not low <= value <= 30):
            raise ValueError(f'--{field.replace("_", "-")} must be between {low} and 30 seconds')
    drone_id = getattr(args, 'drone_id', None)
    if drone_id is not None and not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', drone_id):
        raise ValueError('--drone-id must be a simple manifest ID, not a path')


def select_drone(manifest, args, base_dir):
    drones = manifest.get('drones', [])
    requested = getattr(args, 'drone_id', None)
    if requested is None and len(drones) > 1:
        # Read identity only; never create SwarmOrchestrator or invoke Dispatcher.
        controller = manifest.get('controller', {})
        files = controller.get('mission_files') or [controller.get('mission_file')]
        identities = set()
        for filename in files:
            if not filename:
                continue
            mission_path = Path(controller.get('mission_path', '.')) / filename
            if not mission_path.is_absolute():
                mission_path = Path(base_dir) / mission_path
            with mission_path.open() as stream:
                mission = yaml.safe_load(stream)
            identities.update((mission or {}).get('drones', {}))
        if len(identities) == 1:
            requested = next(iter(identities))
    selected = [d for d in drones if requested is None or d['id'] == requested]
    if len(selected) != 1:
        raise ValueError('IMU calibration requires one aircraft; select --drone-id from the manifest')
    drone = selected[0]
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', str(drone['id'])):
        raise ValueError('unsafe manifest drone ID')
    for field in ('ip', 'user'):
        if not isinstance(drone.get(field), str) or not drone[field].strip():
            raise ValueError(f'drone {drone["id"]} needs manifest {field}')
    return drone


def calibration_command(manifest, drone, args, session, provenance):
    common = manifest['common']
    work = PurePosixPath(common['work_dir'])
    python = PurePosixPath(common['venv_path']) / 'bin' / 'python'
    if not work.is_absolute() or not python.is_absolute():
        raise ValueError('manifest work_dir and venv_path must be absolute remote paths')
    uri = 'usb://0'
    if getattr(args, 'radio', False):
        uri = drone.get('uri', '')
        if not isinstance(uri, str) or not uri.startswith('radio://'):
            raise ValueError('--radio requires the selected aircraft radio:// URI in the manifest')
    relative = PurePosixPath('Interaction/estimator_calibrations') / drone['id'] / session
    tokens = [str(python), '-u', '-m', 'Interaction.calibrate_estimator_imu', 'collect',
              '--uri', uri, '--drone-id', drone['id'],
              '--firmware-id', provenance['firmware_id'],
              '--fixture-id', provenance['fixture_id'],
              '--reference-note', provenance['reference_note'],
              '--output', str(relative),
              '--calibration-method', getattr(args, 'imu_calibration_method', None) or 'gravity_norm',
              '--duration-s', str(getattr(args, 'imu_duration_s', None) or 4.),
              '--settle-s', str(getattr(args, 'imu_settle_s', None) or 2.)]
    if getattr(args, 'imu_fit_only', False):
        tokens += ['--fit-only']
    elif getattr(args, 'imu_auto_flight', False):
        tokens += ['--auto-flight']
        if getattr(args,'imu_flight_config',None):
            tokens += ['--flight-config',f'/tmp/fls_estimator_flight_{session}.yaml']
        else:
            tokens += ['--flight-initial-position',*map(str,validation_seed(drone))]
    if getattr(args, 'imu_calibrate_gyro', False):
        tokens += ['--gyro-calibration']
    if getattr(args, 'validate_estimator_imu', None):
        seed = validation_seed(drone)
        tokens = [str(python), '-u', 'controller.py', '--calibrate', '--log', '--vicon',
                  '--vicon-mode', 'pointcloud', '--drone-id', drone['id'],
                  '--init-pos', *map(str,seed), '--takeoff-altitude',str(seed[2]+.6),
                  '--smooth-controller-rate','100', '--cf-log-period','10',
                  '--imu-validation-session',str(relative),
                  '--imu-validation-output',str(relative/'flight'),
                  '--tag','imu_validation_'+session]
        if uri.startswith('radio://'):
            tokens += ['--radio',uri]
        if getattr(args, 'imu_flight_config', None):
            tokens += ['--imu-validation-config', f'/tmp/fls_estimator_flight_{session}.yaml']
    return f'cd {shlex.quote(str(work))} && {shlex.join(tokens)}', work / relative


class Tee:
    def __init__(self, console, transcript):
        self.console, self.transcript = console, transcript

    def write(self, text):
        self.console.write(text)
        self.transcript.write(text)
        self.flush()

    def flush(self):
        self.console.flush()
        self.transcript.flush()


def validation_seed(drone):
    position=drone.get('init_pos')
    if (not isinstance(position,(list,tuple)) or len(position)!=3
            or any(type(value) not in (int,float) or not math.isfinite(value) for value in position)):
        raise ValueError('built-in validation flight needs the selected aircraft init_pos in the existing manifest')
    return list(position)


def latest_imu_calibration(base_dir, drone_id):
    candidates = []
    for source in (Path(base_dir)/'logs').glob(f'estimator_imu_{drone_id}_*'):
        try:
            # Validation-only copies are not new static calibration sessions.
            launch_path = source/'launch_result.json'
            if launch_path.exists() and json.loads(launch_path.read_text()).get('source_calibration'):
                continue
            data = (source/'dataset.json').read_bytes()
            fit = json.loads((source/'fit/candidate.json').read_text())
            if (fit.get('fit_passed') is True and fit.get('drone_id') == drone_id
                    and fit.get('dataset_sha256') == hashlib.sha256(data).hexdigest()):
                candidates.append(source)
        except (OSError, ValueError):
            continue
    if not candidates:
        raise ValueError(f'no successful saved IMU candidate found for {drone_id}')
    return max(candidates, key=lambda path: path.name)


def run_estimator_calibration(args, manifest_file, base_dir, *, connection_factory=None,
                              prompt=None, session=None, replay_runner=None):
    """Forward the live terminal; retrieve results even when fitting fails."""
    validate_estimator_calibration_options(args)
    with Path(manifest_file).open() as stream:
        manifest = yaml.safe_load(stream)
    drone = select_drone(manifest, args, base_dir)
    saved_inputs = {}
    saved_session = getattr(args, 'validate_estimator_imu', None)
    if saved_session:
        source = (latest_imu_calibration(base_dir, drone['id']) if saved_session == 'latest'
                  else Path(saved_session).expanduser().resolve())
        for name in ('dataset.json', 'packets.jsonl', 'fit/report.json', 'fit/report.md', 'fit/candidate.json'):
            saved_inputs[name] = (source / name).read_bytes()
        saved_fit = json.loads(saved_inputs['fit/candidate.json'])
        if (saved_fit.get('fit_passed') is not True or saved_fit.get('drone_id') != drone['id']
                or saved_fit.get('dataset_sha256') != hashlib.sha256(saved_inputs['dataset.json']).hexdigest()):
            raise ValueError('saved session needs a successful matching static IMU candidate for this aircraft')
    provenance = {}
    # Optional descriptive metadata must not block calibration startup.
    # Unknown firmware is never inferred from an offboard Git revision.
    for key in ('firmware_id','fixture_id','reference_note'):
        value = getattr(args, 'imu_' + key, None)
        provenance[key] = value.strip() if isinstance(value,str) and value.strip() else 'unknown'
    if saved_session:
        provenance['firmware_id'] = saved_fit['firmware_id']
    session = session or (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
                          + '_' + uuid.uuid4().hex[:8])
    if not re.fullmatch(r'[A-Za-z0-9_-]+', session):
        raise ValueError('invalid calibration session identity')
    command, remote_dir = calibration_command(manifest, drone, args, session, provenance)
    fit_only = getattr(args, 'imu_fit_only', False)
    auto_flight = getattr(args, 'imu_auto_flight', False) and not fit_only
    config_bytes = None
    if auto_flight:
        if getattr(args,'imu_flight_config',None):
            config_path = Path(args.imu_flight_config).expanduser()
            if not config_path.is_absolute():
                config_path = Path(base_dir) / config_path
            config_bytes = config_path.read_bytes()
            config = yaml.safe_load(config_bytes)
        else:
            config={'builtin_profile':'imu_validation_v1','initial_position_m':validation_seed(drone)}
        if not isinstance(config, dict):
            raise ValueError('flight config must be a mapping; pointcloud or rigidbody position tracking is supported')
    artifacts = FIT_ARTIFACTS if fit_only else (FLIGHT_ARTIFACTS if auto_flight else ARTIFACTS)
    ram_requested = auto_flight and 'ram_trace' in config
    if ram_requested:
        artifacts += ('flight/ram_trace/trace.bin', 'flight/ram_trace/report.json',
                      'flight/ram_trace/packets.jsonl')
    artifacts=list(artifacts)
    flight_temporary_path=f'/tmp/fls_estimator_flight_{session}.yaml'
    local_dir = Path(base_dir) / 'logs' / f'estimator_imu_{drone["id"]}_{session}'
    local_dir.mkdir(parents=True, exist_ok=False)
    result = {'mode': 'calibrate_estimator_imu', 'drone_id': drone['id'],
              'provenance': provenance, 'remote_dir': str(remote_dir),
              'exit_code': None, 'downloaded': [], 'missing': [], 'download_errors': {},
              'firmware_applied': False, 'flight_started': False}
    result['automatic_flight_requested'] = auto_flight
    if saved_session:
        result['source_calibration'] = str(source)
        print(f'IMU validation flight: {drone["id"]} at {drone["ip"]}. Reusing saved accelerometer fit; gyro zero will refresh on the floor.')
    else:
        method = getattr(args, 'imu_calibration_method', None) or 'gravity_norm'
        print(f'IMU static calibration ({method}): {drone["id"]} at {drone["ip"]}. Remove props.')
    print(f'Artifacts: {local_dir}', flush=True)
    if connection_factory is None:
        from fabric import Connection
        connection_factory = Connection
    try:
        with connection_factory(host=drone['ip'], user=drone['user'], connect_timeout=5) as connection:
            if fit_only:
                from imu_runtime_sync import synchronize
                result['runtime_sync'] = synchronize(connection,manifest['common']['work_dir'],
                    str(PurePosixPath(manifest['common']['venv_path'])/'bin/python'),base_dir)
            with (local_dir / 'terminal.txt').open('w') as transcript:
                try:
                    if saved_inputs:
                        connection.run('mkdir -p ' + shlex.quote(str(remote_dir / 'fit')), hide=True)
                        for name, contents in saved_inputs.items():
                            connection.put(io.BytesIO(contents), remote=str(remote_dir / name))
                    if config_bytes is not None:
                        connection.put(io.BytesIO(config_bytes), remote=flight_temporary_path)
                    remote_result = connection.run(command, pty=True, hide=False, warn=True,
                                                   in_stream=sys.stdin,
                                                   out_stream=Tee(sys.stdout, transcript))
                    result['exit_code'] = remote_result.exited
                except (KeyboardInterrupt, Exception) as error:
                    result['error'] = str(error) or type(error).__name__
                    result['exit_code'] = 130 if isinstance(error, KeyboardInterrupt) else 1
                finally:
                    # This dedicated session never joins the flight upload queue.
                    for name in artifacts:
                        local = local_dir / name
                        local.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            connection.get(str(remote_dir / name), local=str(local))
                            result['downloaded'].append(name)
                            if name=='flight/report.json':
                                try:
                                    report=json.loads(local.read_text())
                                except ValueError:
                                    report={}
                                if isinstance(report,dict) and report.get('ram_trace'):
                                    for extra in ('flight/ram_trace/trace.bin','flight/ram_trace/report.json',
                                                  'flight/ram_trace/packets.jsonl'):
                                        if extra not in artifacts:
                                            artifacts.append(extra)
                        except FileNotFoundError:
                            result['missing'].append(name)
                        except Exception as error:
                            result['download_errors'][name] = str(error)
                    if config_bytes is not None:
                        try:
                            connection.sftp().remove(flight_temporary_path)
                        except Exception as error:
                            result['temporary_config_cleanup_error'] = str(error)
    except (KeyboardInterrupt, Exception) as error:
        result['error'] = str(error) or type(error).__name__
        result['exit_code'] = 130 if isinstance(error, KeyboardInterrupt) else 1
    code = result['exit_code'] if type(result['exit_code']) is int else 1
    if code == 0:
        if result['missing'] or result['download_errors']:
            code = 1
            result['error'] = 'calibration returned success but some artifacts were not retrieved'
        else:
            try:
                fit_name = 'fit/candidate.json' if auto_flight or fit_only else 'fit/calibration.json'
                fit = json.loads((local_dir / fit_name).read_text())
                if (fit.get('schema') != 'estimator_processed_imu_calibration_v1'
                        or fit.get('fit_passed' if auto_flight or fit_only else 'accepted') is not True or fit.get('drone_id') != drone['id']
                        or fit.get('firmware_id') != provenance['firmware_id']):
                    raise ValueError('downloaded calibration not accepted or belongs to another aircraft')
                checksum = hashlib.sha256((local_dir / 'dataset.json').read_bytes()).hexdigest()
                if fit.get('dataset_sha256') != checksum:
                    raise ValueError('downloaded dataset does not match the calibration fingerprint')
                if auto_flight:
                    flight = json.loads((local_dir / 'flight/report.json').read_text())
                    if (flight.get('capture_completed') is not True or flight.get('candidate_applied') is not False
                            or flight.get('drone_id') != drone['id']
                            or flight.get('firmware_id') != provenance['firmware_id']
                            or flight.get('candidate_sha256') != hashlib.sha256((local_dir / fit_name).read_bytes()).hexdigest()
                            or flight.get('packets_sha256') != hashlib.sha256((local_dir / 'flight/packets.jsonl').read_bytes()).hexdigest()):
                        raise ValueError('flight capture failed or does not match this fixture candidate')
                    if ram_requested or flight.get('ram_trace'):
                        trace_path=local_dir / 'flight/ram_trace/trace.bin'
                        if flight.get('ram_trace', {}).get('sha256') != hashlib.sha256(trace_path.read_bytes()).hexdigest():
                            raise ValueError('RAM trace fingerprint does not match the flight report')
                    result['flight_capture_completed'] = True
                    if replay_runner is None:
                        replay_runner = run_offline_replay
                    replay_runner(local_dir, base_dir)
                    result['offline_replay_completed'] = True
            except (ValueError, OSError, subprocess.SubprocessError) as error:
                code = 1
                result['error'] = str(error)
    if auto_flight and (local_dir / 'flight/report.json').exists():
        try:
            result['flight_started'] = json.loads((local_dir / 'flight/report.json').read_text()).get('flight_started', False)
        except ValueError:
            pass
    result['launcher_exit_code'] = code
    (local_dir / 'launch_result.json').write_text(json.dumps(result, indent=2) + '\n')
    label = 'Flight capture from saved calibration' if saved_session else ('Six-face fit + flight capture' if auto_flight else 'Calibration')
    print(f'{label} {"COMPLETED" if code == 0 else "STOPPED/REJECTED"}. Results: {local_dir}')
    if result.get('error'):
        print(result['error'], file=sys.stderr)
    if result['download_errors']:
        print('Some downloads failed; original session remains on the Pi:', remote_dir, file=sys.stderr)
    print('Saved only; no firmware calibration has been applied.')
    return code


def run_offline_replay(local_dir, base_dir):
    """Compile/run on the analysis host after download, never on the flying Pi."""
    repo = Path(base_dir).resolve().parents[1] / 'fls-cf-offboard-controller'
    if not (repo / 'Interaction/replay_estimator3.py').exists():
        raise ValueError('local offboard checkout with Interaction/replay_estimator3.py is required for offline replay')
    command = [sys.executable, '-B', '-m', 'Interaction.replay_estimator3',
               str(local_dir / 'flight/packets.jsonl'), '--candidate', str(local_dir / 'fit/candidate.json'),
               '--output', str(local_dir / 'flight/replay')]
    replay = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=180)
    if replay.returncode:
        raise ValueError('flight captured but offline replay failed: '+replay.stderr[-3000:])
    print(replay.stdout, end='')
    trace_packets=local_dir / 'flight/ram_trace/packets.jsonl'
    if trace_packets.exists():
        # Re-decode the fingerprint-checked binary locally; do not trust a
        # separately transferred derived JSON stream to match that binary.
        decoded=local_dir / 'flight/ram_trace/decoded'
        decode_command=[sys.executable,'-B','-m','Interaction.estimator_ram_trace','decode',
                        str(local_dir/'flight/ram_trace/trace.bin'),'--output',str(decoded)]
        decode=subprocess.run(decode_command,cwd=repo,text=True,capture_output=True,timeout=60)
        if decode.returncode:
            raise ValueError('RAM binary downloaded but decoding failed: '+decode.stderr[-3000:])
        command[4] = str(decoded/'packets.jsonl')
        command[-1] = str(local_dir / 'flight/ram_trace/replay')
        replay = subprocess.run(command, cwd=repo, text=True, capture_output=True, timeout=180)
        if replay.returncode:
            raise ValueError('RAM trace downloaded but offline replay failed: '+replay.stderr[-3000:])
        print(replay.stdout, end='')
