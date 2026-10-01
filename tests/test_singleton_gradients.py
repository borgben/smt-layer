import importlib
import unittest
from unittest.mock import patch

import torch
import z3
from smtlayer import SMTLayer

implementation = importlib.import_module('smtlayer.smtlayer')


class SingletonGradientTests(unittest.TestCase):
    def run_gradient(self, mode='core', joint=False, sample_rate=None, scaling=None):
        a, output, b, fixed = z3.Bools('sg_a sg_output sg_b sg_fixed')
        relation = z3.And(a, z3.Not(b)) if joint else z3.Or(a, z3.Not(b))
        layer = SMTLayer(3, 1, [output == relation],
                         variables=[a, output, b, fixed], fixed_inputs=[fixed],
                         conflict_gradient=mode,
                         measure_singleton_conflicts=sample_rate is not None,
                         singleton_conflict_sample_rate=sample_rate or 0.)
        x = torch.tensor([[0.7, 0., -0.9, 0.3]], requires_grad=True)
        mask = torch.tensor([[1., 0., 1., 1.]])
        y = layer(x, mask=mask, grad_scaling=scaling)
        y.sum().backward()
        self.assertEqual(layer._solver.num_scopes(), 0)
        self.assertEqual(len(layer._solver.assertions()), 1)
        return layer, x.grad, y.detach()

    def test_all_singletons_correct_coordinates_and_magnitudes(self):
        baseline, grad_a, y_a = self.run_gradient()
        experimental, grad_b, y_b = self.run_gradient('all_singletons')
        self.assertTrue(torch.equal(y_a, y_b))
        self.assertEqual(torch.count_nonzero(grad_a).item(), 1)
        expected = torch.zeros_like(grad_b)
        expected[0, 0] = torch.sigmoid(torch.tensor(0.7)) / 4
        expected[0, 2] = (torch.sigmoid(torch.tensor(-0.9)) - 1) / 4
        torch.testing.assert_close(grad_b, expected)
        self.assertEqual(experimental.singleton_gradient_stats['singleton_updates'], 1)
        self.assertEqual(experimental.singleton_gradient_stats['solver_calls'], 3)
        self.assertEqual(baseline.singleton_gradient_stats['queries'], 0)

    def test_no_singletons_falls_back_to_joint_core(self):
        _, original, _ = self.run_gradient(joint=True)
        layer, alternative, _ = self.run_gradient('all_singletons', joint=True)
        torch.testing.assert_close(original, alternative)
        self.assertEqual(torch.count_nonzero(alternative).item(), 2)
        self.assertEqual(layer.singleton_gradient_stats['core_fallbacks'], 1)

    def test_unknown_falls_back_to_original_core(self):
        _, original, _ = self.run_gradient(joint=True)
        with patch.object(implementation, 'probe_singletons', return_value=(
                [z3.Bool('var0')], dict(solver_calls=3, singleton_count=None))):
            layer, alternative, _ = self.run_gradient('all_singletons', joint=True)
        torch.testing.assert_close(original, alternative)
        self.assertEqual(layer.singleton_gradient_stats['core_fallbacks'], 1)

    def test_diagnostic_sampling_does_not_sample_training_updates(self):
        layer, grad, _ = self.run_gradient('all_singletons', sample_rate=0.)
        self.assertEqual(torch.count_nonzero(grad).item(), 2)
        self.assertEqual(layer.singleton_conflicts.summary()['tested'], 0)
        self.assertEqual(layer.singleton_gradient_stats['queries'], 1)

    def test_diagnostic_reuses_training_probes(self):
        with patch.object(implementation, 'probe_singletons', wraps=implementation.probe_singletons) as probe:
            layer, _, _ = self.run_gradient('all_singletons', sample_rate=1.)
        self.assertEqual(probe.call_count, 1)
        self.assertEqual(layer.singleton_conflicts.last['singleton_count'], 2)
        self.assertEqual(layer.singleton_conflicts.last['core_size'], 1)

    def test_existing_gradient_scaling_is_applied(self):
        _, original, _ = self.run_gradient('all_singletons')
        _, scaled, _ = self.run_gradient('all_singletons', scaling=2.)
        weights = torch.softmax(-2 * torch.tensor([0.7, 0.9]), dim=0)
        torch.testing.assert_close(scaled[0, [0, 2]], original[0, [0, 2]] * weights)

    def test_baseline_does_not_scan(self):
        with patch.object(implementation, 'probe_singletons', side_effect=AssertionError('unexpected probe')):
            self.run_gradient()

    def test_invalid_mode_and_maxsat_rejected(self):
        a = z3.Bool('a')
        with self.assertRaises(ValueError):
            SMTLayer(1, 0, [], variables=[a], conflict_gradient='invalid')
        with self.assertRaises(ValueError):
            SMTLayer(1, 0, [], variables=[a], solverop='maxsat', conflict_gradient='all_singletons')
