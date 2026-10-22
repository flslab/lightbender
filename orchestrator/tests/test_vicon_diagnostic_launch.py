import ast
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parents[1]))
from interaction_launch import interaction_vicon_diagnostic_args


class ViconDiagnosticLaunchTests(unittest.TestCase):
    def test_opt_in_only(self):
        self.assertEqual(interaction_vicon_diagnostic_args(SimpleNamespace(), {}), '')
        self.assertEqual(interaction_vicon_diagnostic_args(SimpleNamespace(),
                         {'vicon_source_metadata': False}), '')
        for args, config in [(SimpleNamespace(vicon_source_metadata=True), {}),
                             (SimpleNamespace(), {'vicon_source_metadata': True})]:
            self.assertEqual(interaction_vicon_diagnostic_args(args, config),
                             ' --vicon-source-metadata')

    def test_launcher_parses(self):
        ast.parse((Path(__file__).parents[1] / 'orchestrator.py').read_text())
