"""Saved IMU calibration as a normal Dispatcher/SwarmOrchestrator mission."""
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import tempfile

import yaml

from estimator_calibration_launch import latest_imu_calibration, select_drone, validation_seed, run_offline_replay

INPUTS = ('dataset.json','packets.jsonl','fit/candidate.json','fit/report.json','fit/report.md')
OUTPUTS = ('flight/config.json','flight/report.json','flight/report.md','flight/packets.jsonl')


class ImuValidationMission:
    def __init__(self, args, manifest, base_dir):
        self.base_dir = Path(base_dir)
        drone = select_drone(manifest,args,base_dir)
        self.drone_id = drone['id']
        source = getattr(args,'validate_estimator_imu')
        self.source = (latest_imu_calibration(base_dir,self.drone_id) if source=='latest'
                       else Path(source).expanduser().resolve())
        self.inputs = {name:(self.source/name).read_bytes() for name in INPUTS}
        fit = json.loads(self.inputs['fit/candidate.json'])
        if (fit.get('fit_passed') is not True or fit.get('drone_id') != self.drone_id
                or fit.get('dataset_sha256') != hashlib.sha256(self.inputs['dataset.json']).hexdigest()):
            raise ValueError('IMU validation needs a successful matching candidate for this aircraft')
        if getattr(args,'imu_flight_config',None):
            raise ValueError('Dispatcher IMU validation uses its built-in sampling mission; omit --imu-flight-config')
        seed = validation_seed(drone)
        self.mission = {'name':'estimator_imu_validation','takeoff_speed':.5,
            'Interaction':{'action':'translation','config':{}},
            'drones':{self.drone_id:{'target':[seed[0],seed[1],seed[2]+.6], 'interaction':True}}}
        self.directory = tempfile.TemporaryDirectory(prefix='imu_validation_mission_')
        self.mission_path = Path(self.directory.name)/'imu_validation.yaml'
        self.mission_path.write_text(yaml.safe_dump(self.mission))
        # Only this in-memory launch changes; no authored YAML is modified.
        manifest['drones'] = [deepcopy(drone)]
        manifest['controller']['mission_path'] = self.directory.name
        manifest['controller']['mission_files'] = [self.mission_path.name]
        manifest['controller']['mission_file'] = self.mission_path.name
        manifest.setdefault('mission',{})['require_handshake'] = False
        manifest['mission']['delta_t'] = 0

    def remote_dir(self, tag):
        return PurePosixPath('Interaction/estimator_calibrations')/self.drone_id/('validation_'+tag)

    def controller_args(self, tag):
        relative = self.remote_dir(tag)
        return shlex.join(['--imu-validation-session',str(relative),
                           '--imu-validation-output',str(relative/'flight')])

    def stage(self, connection, work, tag, *, python='python3', runtime_sync=None):
        if runtime_sync is None:
            from imu_runtime_sync import synchronize
            runtime_sync = synchronize
        runtime = runtime_sync(connection,work,python,self.base_dir)
        probe = 'from pathlib import Path; import sys; sys.exit(0 if "IMU_VALIDATION_SWARM_API = 2" in Path("controller.py").read_text() and Path("Interaction/imu_logging.py").is_file() else 3)'
        check = connection.run('cd '+shlex.quote(str(work))+' && '+shlex.quote(python)
            +' -B -c '+shlex.quote(probe),hide=True,warn=True)
        if check.exited != 0:
            raise ValueError('Pi offboard has not been updated for orchestrated IMU validation; sync the new controller.py first')
        remote = PurePosixPath(work)/self.remote_dir(tag)
        connection.run('mkdir -p '+shlex.quote(str(remote/'fit')),hide=True)
        local = self.base_dir/'logs'/tag/'imu_validation'
        local.mkdir(parents=True,exist_ok=True)
        for name, raw in self.inputs.items():
            path = local/name
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(raw)
            connection.put(io.BytesIO(raw),remote=str(remote/name))
        (local/'launch_result.json').write_text(json.dumps({'source_calibration':str(self.source),
            'remote_dir':str(remote),'flight_lifecycle':'standard_orchestrator','firmware_applied':False,
            'runtime_sync':runtime},indent=2))
        return {'remote_dir':str(remote),'local_dir':str(local),'base_dir':str(self.base_dir)}


def retrieve_validation(item, drone, download, logger, *, replay=run_offline_replay):
    """Part of the normal persistent pending-download queue; preserve rejected captures."""
    settings = item['imu_validation']
    remote, local = PurePosixPath(settings['remote_dir']), Path(settings['local_dir'])
    success = True
    for name in OUTPUTS:
        path = local/name
        path.parent.mkdir(parents=True,exist_ok=True)
        success = download(drone,str(remote/name),str(path),'IMU validation '+name,required=True) and success
    if not success:
        return False
    try:
        report = json.loads((local/'flight/report.json').read_text())
        if (report.get('candidate_sha256') != hashlib.sha256((local/'fit/candidate.json').read_bytes()).hexdigest()
                or report.get('packets_sha256') != hashlib.sha256((local/'flight/packets.jsonl').read_bytes()).hexdigest()
                or report.get('drone_id') != drone['id'] or report.get('candidate_applied') is not False):
            raise ValueError('IMU validation capture fingerprint/aircraft mismatch')
        if not report.get('capture_completed'):
            logger.warning('IMU validation did not complete; capture retained in %s',local)
            return True
        marker = local/'replay_completed.json'
        if not marker.exists():
            replay(local,Path(settings['base_dir']))
            marker.write_text(json.dumps({'completed':True}))
        logger.info('IMU validation capture and offline replay saved: %s',local)
        return True
    except (OSError,ValueError) as error:
        logger.error('IMU validation retrieval/replay failed: %s',error)
        return False
