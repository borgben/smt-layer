import csv
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.marco_sweep import run_sweep


class SweepTests(unittest.TestCase):
    def test_shared_checkpoint_randomness_and_csv(self):
        starts, draws, limits = [], [], []

        class Model(torch.nn.Module):
            def __init__(self, size, marco_union_backward=False, marco_max_muses=8):
                super().__init__()
                self.extractor = torch.nn.Linear(1, 1)
                self.decoder = torch.nn.Linear(1, 1)
                self.k = marco_max_muses if marco_union_backward else None
                self.sat = SimpleNamespace(marco_stats=dict(
                    calls=0, muses_found=0, mus_size=0, union_size=0,
                    enumeration_seconds=0., fallbacks=0))

            def cuda(self):
                return self

        def pretrain(model, *args, **kwargs):
            with torch.no_grad():
                model.extractor.weight.fill_(42.)

        def train(epoch, model, *args, **kwargs):
            if epoch == 1:
                starts.append({k: v.clone() for k, v in model.state_dict().items()})
                limits.append(model.k)
            draws.append((random.random(), np.random.rand(), torch.rand(1).item()))
            stats = model.sat.marco_stats
            if model.k is None:
                return 0.75, 0.2, 0.
            stats['calls'] += 1
            stats['muses_found'] += model.k
            stats['mus_size'] += 2 * model.k
            stats['union_size'] += model.k + 1
            stats['enumeration_seconds'] += 0.25
            return 0.75, 0.2, 0.

        with tempfile.TemporaryDirectory() as tmp, patch(
            'scripts.marco_sweep.MNISTSudokuSolver', Model
        ), patch('scripts.marco_sweep.get_dataloader', return_value=([0, 1], [0])) as loaders, patch(
            'scripts.marco_sweep.pretrain', side_effect=pretrain
        ) as pretrained, patch('scripts.marco_sweep.train_epoch', side_effect=train), patch(
            'scripts.marco_sweep.test_epoch', return_value=(0.5, 0.3)
        ), patch('torch.cuda.synchronize'), patch('scripts.marco_sweep.click.echo'):
            output = Path(tmp) / 'results.csv'
            run_sweep(output, seed=7, pretrain_epochs=1, epochs=2)
            self.assertTrue(output.with_suffix('.pretrained.pt').exists())
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(pretrained.call_count, 1)
            self.assertEqual(loaders.call_count, 2)
            self.assertEqual(limits, [None, *range(1, 9)])
            for state in starts:
                self.assertEqual(state['extractor.weight'].item(), 42.)
                for key in state:
                    torch.testing.assert_close(state[key], starts[0][key], rtol=0, atol=0)
            for k in range(9):
                self.assertEqual(draws[2*k:2*k+2], draws[:2])
            self.assertEqual(len(rows), 9)
            self.assertEqual(rows[0]['mode'], 'core')
            self.assertEqual(rows[0]['K'], '')
            for field in ('marco_calls', 'muses_found', 'mean_mus_size',
                          'mean_union_size', 'marco_enumeration_seconds', 'fallbacks'):
                self.assertEqual(float(rows[0][field]), 0.)
            for k, row in enumerate(rows[1:], 1):
                self.assertEqual(row['mode'], 'marco_union')
                self.assertEqual(int(row['K']), k)
                self.assertEqual(int(row['muses_found']), 2*k)
                self.assertEqual(float(row['mean_mus_size']), 2.)
                self.assertEqual(float(row['mean_union_size']), k+1)
                self.assertEqual(float(row['marco_enumeration_seconds']), 0.5)
                self.assertEqual(float(row['train_loss']), 0.2)
                self.assertEqual(float(row['test_loss']), 0.3)


if __name__ == '__main__':
    unittest.main()
