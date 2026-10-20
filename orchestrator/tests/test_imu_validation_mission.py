"""Normal swarm/UI IMU mission routing, staging and recovery; no hardware."""
import ast
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

sys.path.insert(0,str(Path(__file__).parents[1]))
from imu_validation_mission import ImuValidationMission, retrieve_validation, INPUTS, OUTPUTS
from test_estimator_calibration_launch import manifest, cli_parser
from test_mpc_calibration_launch import command_builder


class MissionTests(unittest.TestCase):
    def setUp(self):
        sync=patch('imu_runtime_sync.synchronize',return_value={'checked_by_separate_sync_tests':True})
        sync.start();self.addCleanup(sync.stop)

    def test_combined_calibration_rejoins_ui_only_after_successful_static_fit(self):
        tree=ast.parse((Path(__file__).parents[1]/'orchestrator.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
        branch=next(n for n in main.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute)
            and n.test.attr=='calibrate_estimator_imu')
        for code in (0,1):
            args=cli_parser().parse_args(['--calibrate-estimator-imu','--drone-id','lb11','--imu-auto-flight'])
            run=Mock(return_value=code)
            scope={'args':args,'run_estimator_calibration':run,'MANIFEST_FILE':'manifest',
                'BASE_DIR':'base','parser':Mock()}
            compiled=compile(ast.Module(body=branch.body,type_ignores=[]),'orchestrator.py','exec')
            if code:
                with self.assertRaises(SystemExit):exec(compiled,scope)
                self.assertTrue(args.calibrate_estimator_imu)
            else:
                exec(compiled,scope)
                self.assertFalse(args.calibrate_estimator_imu)
                self.assertEqual(args.validate_estimator_imu,'latest')
                self.assertTrue(args.calibrate)
                self.assertFalse(args.skip_reboot)
            self.assertTrue(args.imu_fit_only)
            run.assert_called_once()

    def test_combined_calibration_preserves_explicit_skip_reboot(self):
        tree=ast.parse((Path(__file__).parents[1]/'orchestrator.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
        branch=next(n for n in main.body if isinstance(n,ast.If) and isinstance(n.test,ast.Attribute)
                    and n.test.attr=='calibrate_estimator_imu')
        args=cli_parser().parse_args(['--calibrate-estimator-imu','--drone-id','lb11',
                                     '--imu-auto-flight','--skip-reboot'])
        scope={'args':args,'run_estimator_calibration':Mock(return_value=0),'MANIFEST_FILE':'manifest',
               'BASE_DIR':'base','parser':Mock()}
        exec(compile(ast.Module(body=branch.body,type_ignores=[]),'main','exec'),scope)
        self.assertEqual(args.validate_estimator_imu,'latest')
        self.assertTrue(args.skip_reboot)

    def plan(self, root):
        source = root/'logs/estimator_imu_lb11_001'
        (source/'fit').mkdir(parents=True)
        for name in INPUTS:
            (source/name).write_text('{}')
        (source/'fit/candidate.json').write_text(json.dumps(dict(fit_passed=True,drone_id='lb11',
            firmware_id='test',dataset_sha256=hashlib.sha256(b'{}').hexdigest())))
        args=cli_parser().parse_args(['--validate-estimator-imu','--drone-id','lb11'])
        args.calibrate=True
        data=manifest()
        return ImuValidationMission(args,data,root),args,data

    def test_normal_command_is_detached_orchestrated_and_uses_ui_selected_point(self):
        with TemporaryDirectory() as tmp:
            plan,args,data=self.plan(Path(tmp))
            self.addCleanup(plan.directory.cleanup)
            self.assertEqual(len(data['drones']),1)
            self.assertFalse(data['mission']['require_handshake'])
            self.assertEqual(yaml.safe_load(plan.mission_path.read_text())['name'],'estimator_imu_validation')
            methods=command_builder();methods['TIME_SYNC_CMD']=''
            owner=SimpleNamespace(args=args,mission=plan.mission,manifest=data,common_cfg=data['common'],
                tag='trial',imu_validation_plan=plan)
            drone={**data['drones'][0],'init_pos':[.31,-.74,.25],'servo_count':0,'led_count':0}
            command=methods['_get_drone_cmd_interaction'](owner,drone)
            tokens=shlex.split(command)
            for flag in ('nohup','controller.py','--orchestrated','--calibrate','--imu-validation-session'):
                self.assertIn(flag,tokens)
            self.assertNotIn('git pull',command)
            self.assertNotIn('Interaction.estimator_validation_flight',command)
            self.assertIn('> drone_lb11.log 2>&1 < /dev/null &',command)
            pos=tokens.index('--init-pos')
            self.assertEqual(list(map(float,tokens[pos+1:pos+4])),[.31,-.74,.25])
            owner.imu_validation_plan=None
            normal=methods['_get_drone_cmd_interaction'](owner,drone)
            self.assertIn('git pull &&',normal)
            self.assertNotIn('--imu-validation-session',normal)
            self.assertIn('--orchestrated --calibrate',normal)

    def test_stage_and_normal_download_queue_verify_data_then_replay(self):
        with TemporaryDirectory() as tmp:
            root=Path(tmp);plan,args,data=self.plan(root)
            self.addCleanup(plan.directory.cleanup)
            conn=Mock();uploaded={}
            conn.run.return_value.exited=0
            conn.put.side_effect=lambda stream,remote:uploaded.update({remote:stream.read()})
            settings=plan.stage(conn,'/home/fls/controller','trial')
            self.assertEqual(len(uploaded),5)
            local=Path(settings['local_dir'])
            self.assertEqual((local/'dataset.json').read_bytes(),plan.inputs['dataset.json'])
            payload={name:'{}' for name in OUTPUTS}
            payload['flight/report.json']=json.dumps(dict(capture_completed=True,candidate_applied=False,drone_id='lb11',
                candidate_sha256=hashlib.sha256(plan.inputs['fit/candidate.json']).hexdigest(),
                packets_sha256=hashlib.sha256(b'{}').hexdigest()))
            def download(drone,remote,path,label,required):
                name=next(name for name in OUTPUTS if remote.endswith(name))
                Path(path).write_text(payload[name]);return True
            replay=Mock();logger=Mock();item={'imu_validation':settings}
            self.assertTrue(retrieve_validation(item,{'id':'lb11'},download,logger,replay=replay))
            replay.assert_called_once_with(local,root)
            self.assertTrue(retrieve_validation(item,{'id':'lb11'},download,logger,replay=replay))
            replay.assert_called_once()
            payload['flight/packets.jsonl']='changed'
            self.assertFalse(retrieve_validation(item,{'id':'lb11'},download,logger,replay=replay))

    def test_old_pi_code_is_rejected_without_upload_or_flight_launch(self):
        with TemporaryDirectory() as tmp:
            plan,args,data=self.plan(Path(tmp));self.addCleanup(plan.directory.cleanup)
            conn=Mock();conn.run.return_value.exited=3
            with self.assertRaisesRegex(ValueError,'has not been updated'):
                plan.stage(conn,'/home/fls/controller','trial')
            conn.put.assert_not_called()

    def test_main_routes_validation_into_swarm_not_interactive_ssh(self):
        source=(Path(__file__).parents[1]/'orchestrator.py').read_text()
        tree=ast.parse(source)
        main=next(n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare))
        normalization=next(n for n in main.body if isinstance(n,ast.If)
            and isinstance(n.test,ast.Attribute) and n.test.attr=='validate_estimator_imu')
        for skip in (False,True):
            with self.subTest(explicit_skip=skip):
                namespace={'args':SimpleNamespace(validate_estimator_imu='latest',calibrate_estimator_imu=False,
                    imu_calibrate_gyro=False,calibrate=False,skip_reboot=skip),'parser':Mock()}
                exec(compile(ast.Module(body=[normalization],type_ignores=[]),'main','exec'),namespace)
                self.assertTrue(namespace['args'].calibrate)
                self.assertEqual(namespace['args'].skip_reboot,skip)
                self.assertFalse(namespace['args'].calibrate_estimator_imu)

    def test_normal_reboot_routes_selected_aircraft_through_radio_pi(self):
        tree=ast.parse((Path(__file__).parents[1]/'orchestrator.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SwarmOrchestrator')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='reboot_flight_controllers')
        connection=Mock();pause=Mock()
        namespace={'Connection':connection,'time':SimpleNamespace(sleep=pause)}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'reboot','exec'),namespace)
        owner=SimpleNamespace(drones=[{'id':'lb11','uri':'radio://0/100/2M/E7E7E7E706'}],
            args=SimpleNamespace(radio=False,droneless=False),radio_node={'ip':'192.168.1.90','user':'fls'},
            common_cfg={'venv_path':'/home/fls/env','work_dir':'/home/fls/fls-cf-offboard-controller'},logger=Mock())
        namespace['reboot_flight_controllers'](owner,remote=True)
        connection.assert_called_once_with(host='192.168.1.90',user='fls',connect_timeout=5)
        command=connection.return_value.run.call_args.args[0]
        self.assertIn('python3 restart.py radio://0/100/2M/E7E7E7E706',command)
        pause.assert_called_once_with(5)


if __name__=='__main__':unittest.main()
