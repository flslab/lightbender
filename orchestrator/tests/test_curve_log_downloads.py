"""Offline checks only: no connections and no launch/flight commands."""
import ast
import concurrent.futures
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def methods():
    tree=ast.parse((Path(__file__).parents[1]/'orchestrator.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='SwarmOrchestrator')
    selected=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_process_pending_downloads','_download_file')]
    ns={'os':os,'concurrent':concurrent,'Connection':Mock()}
    exec(compile(ast.Module(body=selected,type_ignores=[]),'orchestrator.py','exec'),ns)
    return ns


class CurveDownloadTests(unittest.TestCase):
    def test_required_missing_is_retained_optional_missing_keeps_legacy_behavior(self):
        ns=methods();ns['Connection'].return_value.get.side_effect=FileNotFoundError()
        ctrl=SimpleNamespace(logger=Mock(),emergency=False)
        device={'ip':'not-used','user':'not-used'}
        self.assertFalse(ns['_download_file'](ctrl,device,'remote','local','states',required=True))
        self.assertTrue(ns['_download_file'](ctrl,device,'remote','local','legacy'))

    def test_both_sidecars_attempted_and_failure_retains_run(self):
        ns=methods()
        item={'drone':{'id':'lb11'},'tag':'test','curve_log':True}
        ctrl=SimpleNamespace(logger=Mock(),pending_downloads=[item],
            common_cfg={'work_dir':'/remote'},_download_file=Mock(side_effect=[True,False,True]))
        with patch.object(os,'makedirs'):
            ns['_process_pending_downloads'](ctrl)
        self.assertEqual(ctrl.pending_downloads,[item])
        calls=ctrl._download_file.call_args_list
        self.assertEqual([c.args[1] for c in calls],['/remote/logs/test.json',
            '/remote/logs/test.curves.jsonl','/remote/logs/test.states.jsonl'])
        self.assertTrue(all(c.kwargs.get('required') for c in calls[1:]))
        self.assertEqual(calls[1].args[2],'./logs/test/lb11_test.curves.jsonl')

    def test_old_pending_runs_do_not_request_new_files(self):
        ns=methods();ctrl=SimpleNamespace(logger=Mock(),pending_downloads=[{'drone':{'id':'lb11'},'tag':'old'}],
            common_cfg={'work_dir':'/remote'},_download_file=Mock(return_value=True))
        with patch.object(os,'makedirs'):ns['_process_pending_downloads'](ctrl)
        self.assertEqual(ctrl.pending_downloads,[]);ctrl._download_file.assert_called_once()
