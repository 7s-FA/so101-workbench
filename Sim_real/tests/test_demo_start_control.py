"""Exercise the production nested command handler without starting Isaac."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import unittest


class DemoStartControlTests(unittest.TestCase):
    def context(self,autostart=False):
        tree=ast.parse((Path(__file__).parents[1]/'scripts/factory_sim.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        init=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='demo_time')
        handler=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='demo_command')
        for i,n in enumerate(handler.body):
            if isinstance(n,ast.Nonlocal):handler.body[i]=ast.Global(names=n.names)
        env={'args':SimpleNamespace(verify_from=0.,demo_autostart=autostart,demo_start='preloaded',demo_speed=8.),
             'demo_plans':{'load':('full','full routes'),'preloaded':('preloaded','quick routes')},
             'runtime':SimpleNamespace(select_plan=Mock()),'playback_clock':SimpleNamespace(reset=Mock()),'cycle_reports':[1],'capture_stages':{1},
             'time':SimpleNamespace(monotonic=lambda:10.),'plan':'preloaded','route_report':'quick routes'}
        module=ast.Module(body=[*main.body[init:main.body.index(handler)],handler],type_ignores=[])
        exec(compile(ast.fix_missing_locations(module),'factory_sim commands','exec'),env)
        return env

    def test_launch_and_mode_selection_wait_for_explicit_start(self):
        e=self.context();command=e['demo_command']
        self.assertTrue(e['demo_paused']);self.assertFalse(e['demo_started'])
        command({'command':'start_mode','mode':'load'})
        self.assertEqual(e['plan'],'full');self.assertEqual(e['args'].demo_speed,8.)
        e['runtime'].select_plan.assert_called_with('full')
        self.assertEqual(e['demo_time'],0.);self.assertTrue(e['demo_paused'])
        command('resume');self.assertTrue(e['demo_started']);self.assertFalse(e['demo_paused'])
        e['demo_time']=123.;command({'command':'start_mode','mode':'preloaded'})
        self.assertEqual(e['demo_time'],0.);self.assertFalse(e['demo_started']);self.assertTrue(e['demo_paused'])
        self.assertFalse(e['cycle_reports'] or e['capture_stages'])

    def test_pause_resume_fault_and_restart(self):
        e=self.context(True);command=e['demo_command']
        self.assertFalse(e['demo_paused']);self.assertTrue(e['demo_started'])
        command('pause');self.assertTrue(e['demo_paused'])
        e['demo_fault']='hold';command('resume');self.assertTrue(e['demo_paused'])
        command('restart');self.assertIsNone(e['demo_fault']);self.assertTrue(e['demo_paused'])
        command('resume');self.assertFalse(e['demo_paused'])
