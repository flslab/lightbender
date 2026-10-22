"""Backed-up, fingerprint-checked delivery for the dedicated IMU workflow.

No git reset/pull, dependency installation, firmware flash or device access.
Only these runtime sources are replaced, and an active controller blocks it.
"""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import uuid

FILES=('controller.py','Interaction/calibrate_estimator_imu.py',
    'Interaction/commander_handoff.py','Interaction/calibration_contact_logging.py',
    'Interaction/estimator_gravity_calibration.py',
    'Interaction/estimator_hover_initialization.py',
    'Interaction/estimator_hover_live.py','Interaction/estimator_xy_loading.py',
    'Interaction/estimator_imu_calibration.py','Interaction/estimator_gyro_calibration.py',
    'Interaction/estimator_validation_controller.py','Interaction/estimator_validation_flight.py',
    'Interaction/imu_logging.py','Interaction/firmware_parameter_confirmation.py')

PREPARE='''
import hashlib,json,sys
from pathlib import Path
work,stage=map(Path,sys.argv[1:3]); files=json.loads(sys.argv[3])
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try: parts=proc.read_bytes().split(b'\\0')
    except (OSError,PermissionError): continue
    if any(Path(p.decode(errors='replace')).name=='controller.py' for p in parts):
        raise SystemExit('Stop the running controller before updating IMU runtime')
before={}
for name in files:
    path=work/name
    if path.is_symlink():raise SystemExit('Refusing symlink runtime destination: '+name)
    before[name]=hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
stage.mkdir(parents=True,exist_ok=False)
(stage/'before.json').write_text(json.dumps(before))
for name in files:(stage/name).parent.mkdir(parents=True,exist_ok=True)
'''

INSTALL='''
import hashlib,json,os,shutil,sys
from pathlib import Path
work,stage=map(Path,sys.argv[1:3]); expected=json.loads(sys.argv[3])
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try: parts=proc.read_bytes().split(b'\\0')
    except (OSError,PermissionError): continue
    if any(Path(p.decode(errors='replace')).name=='controller.py' for p in parts):
        raise SystemExit('Controller started during IMU runtime synchronization; stop it first')
before=json.loads((stage/'before.json').read_text()); changed=[]
for name,digest in expected.items():
    staged=stage/name; target=work/name; raw=staged.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:raise SystemExit('Upload checksum mismatch: '+name)
    compile(raw,str(target),'exec')
    current=hashlib.sha256(target.read_bytes()).hexdigest() if target.exists() else None
    if current!=before[name]:raise SystemExit('Runtime changed during synchronization: '+name)
    if current!=digest:changed.append(name)
backup=work/'.imu_runtime_backups'/stage.name
backup.mkdir(parents=True,exist_ok=False)
manifest={'before':before,'after':expected,'changed':changed,'firmware_changed':False}
(backup/'manifest.json').write_text(json.dumps(manifest,indent=2))
installed=[]
try:
    for name in changed:
        target=work/name
        if target.exists():
            old=backup/name;old.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,old)
    for name in changed:
        target=work/name;target.parent.mkdir(parents=True,exist_ok=True)
        os.replace(stage/name,target);installed.append(name)
    for name,digest in expected.items():
        if hashlib.sha256((work/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Readback mismatch: '+name)
except BaseException:
    for name in reversed(installed):
        target=work/name
        if before[name] is None:target.unlink(missing_ok=True)
        else:shutil.copy2(backup/name,target)
    raise
shutil.rmtree(stage)
print(json.dumps({'changed':changed,'backup':str(backup),'after':expected}))
'''


def synchronize(connection,work,python,base_dir):
    repo=Path(base_dir).resolve().parents[1]/'fls-cf-offboard-controller'
    payload={name:(repo/name).read_bytes() for name in FILES}
    expected={name:hashlib.sha256(raw).hexdigest() for name,raw in payload.items()}
    work=PurePosixPath(work);stage=work/('.imu_runtime_'+uuid.uuid4().hex)
    def command(script,argument):
        return shlex.join([str(python),'-B','-c',script,str(work),str(stage),json.dumps(argument)])
    prepared=connection.run(command(PREPARE,list(FILES)),hide=True,warn=True)
    if prepared.exited:
        raise ValueError('IMU runtime preflight failed: '+getattr(prepared,'stderr','').strip())
    try:
        for name,raw in payload.items():connection.put(io.BytesIO(raw),remote=str(stage/name))
        installed=connection.run(command(INSTALL,expected),hide=True,warn=True)
        if installed.exited:
            raise ValueError('IMU runtime validation/install failed: '+getattr(installed,'stderr','').strip())
        result=json.loads(installed.stdout.strip().splitlines()[-1])
        if result.get('after')!=expected:raise ValueError('IMU runtime readback did not match local files')
        print(f'[IMU] Pi runtime verified; {len(result["changed"])} files updated. Backup: {result["backup"]}',flush=True)
        return result
    finally:
        # Only the unique staging directory is removed, including on rejection.
        cleanup='import shutil,sys; shutil.rmtree(sys.argv[1],ignore_errors=True)'
        connection.run(shlex.join([str(python),'-B','-c',cleanup,str(stage)]),hide=True,warn=True)
