"""Telemetry must preserve training RNG and checkpoint selection semantics."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import sodetr_reporting_trainer as reporting
from sodetr_reports import read_json


class ReportingTrainerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.trainer = object.__new__(reporting.ReportingRTDETRTrainer)
        t = self.trainer
        t.save_dir = Path(self.tmp.name)
        t.resume = False
        t.device = torch.device('cpu')
        t.start_epoch = 0
        t.model = torch.nn.Linear(2, 2)
        t.model.yaml = {'yaml_file': 'r18.yaml'}
        t.args = SimpleNamespace(imgsz=640, model='r18.yaml')
        t.epoch, t.epochs, t.stop = 2, 10, True
        t.fitness = t.best_fitness = .3

    def test_rng_best_epoch_and_early_stop(self):
        t = self.trainer
        rng = torch.random.get_rng_state().clone()
        def profile(*args):
            torch.rand(5)
            return 12.0
        with patch.object(reporting, 'RANK', -1), patch.object(reporting, 'get_flops', side_effect=profile):
            t._report_start(t)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            t._report_epoch(t)
            t.epoch = 5
            t.fitness = .2
            t._report_epoch(t)
            t._report_end(t)
        state = read_json(t.save_dir / 'training_state.json')
        self.assertEqual(state['best_epoch'], 3)
        self.assertEqual(state['epochs_completed'], 6)
        self.assertEqual(state['stop_reason'], 'early_stopping')
        self.assertEqual(state['status'], 'completed')
        self.assertGreaterEqual(state['training_seconds'], 0)

    def test_setup_failure_is_persisted(self):
        t = self.trainer
        with patch.object(reporting, 'RANK', -1), patch.object(reporting.RTDETRTrainer, '_do_train', side_effect=RuntimeError('setup failed')):
            with self.assertRaisesRegex(RuntimeError, 'setup failed'):
                t._do_train()
        self.assertEqual(read_json(t.save_dir / 'training_state.json')['status'], 'failed')

    def test_other_ddp_ranks_do_not_write(self):
        t = self.trainer
        with patch.object(reporting, 'RANK', 1):
            t._report_start(t)
            t._report_epoch(t)
            t._report_end(t)
        self.assertFalse((t.save_dir / 'training_state.json').exists())


if __name__ == '__main__':
    unittest.main()
