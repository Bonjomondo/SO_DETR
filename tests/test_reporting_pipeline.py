"""Exercise entry point orchestration without starting GPU training."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import yaml

import train_sodetr_visdrone as entry
from sodetr_reports import read_json, write_json


class ReportingPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data.yaml'
        self.data.write_text('names: [car]\n')
        self.run = self.root / 'runs' / 'Q1'
        self.run.mkdir(parents=True)
        self.model = MagicMock()
        self.model.trainer = SimpleNamespace(save_dir=self.run, args=SimpleNamespace(
            data=str(self.data), imgsz=320, batch=2, workers=0, device='cpu'))
        self.model.train.side_effect = self.train
        self.argv = ['train_sodetr_visdrone.py', '--data', str(self.data), '--device', 'cpu',
                     '--reports-dir', str(self.root / 'reports'), '--experiment-description', '质量监督消融']

    def train(self, **kwargs):
        (self.run / 'args.yaml').write_text(yaml.safe_dump({k: v for k, v in kwargs.items() if k != 'trainer'}))
        write_json(self.run / 'training_state.json', {'status': 'completed', 'model': 'r18.yaml', 'best_epoch': 3})

    def evaluate(self, **kwargs):
        write_json(self.run / 'formal_coco/best/coco_metrics.json', {'metrics': {'AP': .31}})

    def invoke(self, extra=(), owner=True):
        env = {'RANK': '-1', 'SODETR_REPORT_OWNER_PID': str(os.getpid()) if owner else 'other-process'}
        with patch.dict(os.environ, env), patch('sys.argv', self.argv + list(extra)), \
             patch.dict('sys.modules', {'ultralytics': SimpleNamespace(RTDETR=lambda _: self.model),
                                       'sodetr_reporting_trainer': SimpleNamespace(ReportingRTDETRTrainer=object)}), \
             patch.object(entry, 'prepare_formal_coco_eval', return_value=None if '--no-formal-coco-eval' in extra else self.data), \
             patch.object(entry, 'run_formal_coco_eval', side_effect=self.evaluate) as evaluate:
            entry.main()
            return evaluate

    def test_success_evaluates_restored_settings_and_exports(self):
        evaluate = self.invoke()
        self.assertEqual(evaluate.call_count, 1)
        self.assertEqual(evaluate.call_args.kwargs['imgsz'], 320)
        self.assertEqual(read_json(self.run / 'evaluation_state.json')['status'], 'completed')
        self.assertEqual(read_json(self.run / 'experiment.json')['description'], '质量监督消融')
        data = read_json(self.root / 'reports/experiments.json')
        self.assertEqual(data['experiments'][0]['metrics']['AP'], .31)

    def test_evaluation_failure_still_exports_status_and_raises(self):
        self.evaluate = MagicMock(side_effect=RuntimeError('GPU unavailable'))
        with self.assertRaisesRegex(RuntimeError, 'GPU unavailable'):
            self.invoke()
        data = read_json(self.root / 'reports/experiments.json')
        self.assertEqual(data['experiments'][0]['evaluation_status'], 'failed')
        self.assertEqual(data['experiments'][0]['metrics'], {})

    def test_disabled_evaluation_and_ddp_worker(self):
        evaluate = self.invoke(extra=['--no-formal-coco-eval'])
        self.assertEqual(evaluate.call_count, 0)
        self.assertEqual(read_json(self.run / 'evaluation_state.json')['status'], 'skipped')
        evaluate = self.invoke(owner=False)
        self.assertEqual(evaluate.call_count, 0)

    def test_training_failure_still_exports(self):
        def fail(**kwargs):
            self.train(**kwargs)
            write_json(self.run / 'training_state.json', {'status': 'failed', 'stop_reason': 'RuntimeError'})
            raise RuntimeError('training failed')
        self.model.train.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, 'training failed'):
            self.invoke()
        self.assertEqual(read_json(self.root / 'reports/experiments.json')['experiments'][0]['training_status'], 'failed')


if __name__ == '__main__':
    unittest.main()
