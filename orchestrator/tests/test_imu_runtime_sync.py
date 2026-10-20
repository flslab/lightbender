"""Real staged file delivery/backup/rejection, local filesystem transport only."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]))
from imu_runtime_sync import synchronize,FILES


class LocalConnection:
    def run(self,command,**kwargs):
        result=subprocess.run(shlex.split(command),text=True,capture_output=True)
        result.exited=result.returncode
        return result
    def put(self,stream,remote):Path(remote).write_bytes(stream.read())


class RuntimeSyncTests(unittest.TestCase):
    def fixtures(self,root):
        base=root/'lightbender/orchestrator';base.mkdir(parents=True)
        repo=root/'fls-cf-offboard-controller';work=root/'pi';work.mkdir()
        for name in FILES:
            source=repo/name;source.parent.mkdir(parents=True,exist_ok=True);source.write_text('value = 2\n')
        (work/'controller.py').write_text('value = 1\n')
        return base,repo,work

    def test_checks_upload_readback_and_keeps_remote_original_with_manifest(self):
        with TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            base,repo,work=self.fixtures(Path(tmp));connection=LocalConnection()
            result=synchronize(connection,work,sys.executable,base)
            self.assertEqual(set(result['changed']),set(FILES))
            self.assertEqual((Path(result['backup'])/'controller.py').read_text(),'value = 1\n')
            self.assertEqual((work/'controller.py').read_text(),'value = 2\n')
            self.assertFalse(json.loads((Path(result['backup'])/'manifest.json').read_text())['firmware_changed'])
            again=synchronize(connection,work,sys.executable,base)
            self.assertEqual(again['changed'],[])
            self.assertFalse([p for p in work.glob('.imu_runtime_*') if p.name!='.imu_runtime_backups'])

    def test_bad_upload_or_local_syntax_rejects_before_replacing_any_file(self):
        for fault in ('checksum','syntax','concurrent_edit'):
            with self.subTest(fault=fault),TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
                base,repo,work=self.fixtures(Path(tmp));connection=LocalConnection()
                if fault=='syntax':(repo/FILES[-1]).write_text('this is not valid Python!\n')
                elif fault=='checksum':connection.put=lambda stream,remote:Path(remote).write_bytes(b'tampered')
                else:
                    original=connection.put
                    def put(stream,remote):
                        original(stream,remote);(work/'controller.py').write_text('user concurrent edit\n')
                    connection.put=put
                with self.assertRaisesRegex(ValueError,'validation/install failed'):
                    synchronize(connection,work,sys.executable,base)
                self.assertEqual((work/'controller.py').read_text(),
                    'user concurrent edit\n' if fault=='concurrent_edit' else 'value = 1\n')
                self.assertFalse((work/FILES[-1]).exists())
                self.assertFalse([p for p in work.glob('.imu_runtime_*') if p.name!='.imu_runtime_backups'])


if __name__=='__main__':unittest.main()
