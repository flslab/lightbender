import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1]/'authoring/camera_facing.py'
spec = importlib.util.spec_from_file_location('camera_facing', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def point(yaw=150, kind='H'):
    return dict(id='lb4', type=kind, x=0, y=0, z=1, yaw=yaw,
                angle_1=135 if kind=='H' else 225, angle_2=225 if kind=='H' else 405,
                length_1=.12, length_2=.1, max_length_limit=.156,
                light_module_offset=[.06, .01, -.02])


class FacingTest(unittest.TestCase):
    def verify_projection(self, p, C):
        original=copy.deepcopy(p)
        out=m.solve_layout(dict(camera=C, points=[p]))['points'][0]
        self.assertEqual(out['status'], 'corrected', out)
        self.assertEqual(p, original)
        H,T,_=m.endpoints(p); h,t,n=m.endpoints(out)
        self.assertGreater(n @ (np.array(C)-h), 0)
        self.assertLessEqual(out['length_1'], .156+1e-9)
        self.assertLessEqual(out['length_2'], .156+1e-9)
        # Compare projected endpoint coordinates in a pinhole camera looking -X.
        def project(P):
            P=P-np.array(C)
            return P[1:]/(-P[0])
        np.testing.assert_allclose(project(H),project(h),atol=1e-10)
        for slot,source in enumerate(out['source_rods']):
            np.testing.assert_allclose(project(T[source-1]),project(t[slot]),atol=1e-10)
            # Whole continuous segments lie on the same projected line; their
            # parameterizations need not match (perspective is nonlinear).
            a,b=project(H),project(T[source-1]); direction=b-a
            for alpha in np.linspace(0,1,21):
                q=project(h+alpha*(t[slot]-h))
                self.assertLess(abs(np.linalg.det(np.stack([direction,q-a]))),1e-10)
        return out

    def test_exact_projection_and_mount(self):
        for yaw in (100,130,180,240,270):
            with self.subTest(yaw=yaw): self.verify_projection(point(yaw),[2.25,0,.94])

    def test_reversal_preserves_world_geometry(self):
        p=point(180)
        out=self.verify_projection(p,[2.25,0,1])
        self.assertEqual(out['strategy'],'reverse_normal')
        H,T,_=m.endpoints(p);h,t,_=m.endpoints(out)
        np.testing.assert_allclose(H,h,atol=1e-10)
        for j,source in enumerate(out['source_rods']):
            np.testing.assert_allclose(T[source-1],t[j],atol=1e-10)

    def test_v_and_tilt(self):
        p=point(160,'V');p.update(roll=12,pitch=-8)
        self.verify_projection(p,[2.25,0,1])

    def test_unchanged_and_unsupported(self):
        p=point(0)
        out=m.solve_layout(dict(camera=[2.25,0,1],points=[p]))['points'][0]
        self.assertEqual(out['status'],'unchanged')
        self.assertEqual(out['x'],p['x'])
        p['type']='RING'
        self.assertEqual(m.solve_point(p,[2.25,0,1])['status'],'unsupported')

    def test_infeasible_is_explicit(self):
        p=point(180)
        out=m.solve_point(p,[0,0,3])
        self.assertEqual(out['status'],'infeasible')
        self.assertEqual(out['yaw'],p['yaw'])

    def test_cli_and_invalid_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'input.json';output=Path(directory)/'output.json'
            source.write_text(json.dumps(dict(camera=[2.25,0,1],points=[point()])))
            subprocess.run([sys.executable,str(SCRIPT),'--input_file',str(source),'--output_file',str(output)],check=True)
            self.assertEqual(json.loads(output.read_text())['points'][0]['status'],'corrected')
        with self.assertRaises(ValueError):
            m.solve_layout(dict(camera=[2,0,1],points=[point(),point()]))


if __name__=='__main__': unittest.main()
