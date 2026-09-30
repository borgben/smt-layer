"""Run with unittest discover -s tests; the isolated helper needs only Z3."""
import importlib.util
from pathlib import Path
import random
import unittest

import z3

spec = importlib.util.spec_from_file_location(
    'diagnostics', Path(__file__).resolve().parents[1] / 'smtlayer' / 'diagnostics.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Diagnostic = module.AlternativeConflictDiagnostic


class DiagnosticTests(unittest.TestCase):
    def run_case(self, theory, assumptions, hard=(), rate=1):
        solver = z3.Solver()
        solver.set('core.minimize', True)
        solver.add(*theory)
        background = solver.sexpr()
        tracked = [(z3.FreshBool('track'), a) for a in assumptions]
        solver.push()
        solver.add(*hard)
        for label, literal in tracked:
            solver.assert_and_track(literal, label)
        self.assertEqual(solver.check(), z3.unsat)
        core = list(solver.unsat_core())
        solver.pop()
        state = random.getstate()
        diagnostic = Diagnostic(rate)
        diagnostic.observe(solver, hard, tracked, core)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(solver.sexpr(), background)
        self.assertEqual(solver.num_scopes(), 0)
        return diagnostic

    def test_unique_core(self):
        a, b, c = z3.Bools('a b c')
        d = self.run_case([z3.Not(z3.And(a, b))], [a, b, c])
        self.assertFalse(d.last['alternative'])
        self.assertEqual(d.last['probes'], 2)
        self.assertEqual(d.summary()['tested_every_member'], 1)
        self.assertEqual(d.summary()['fraction_with_alternative'], 0)

    def test_disjoint_conflicts(self):
        a, b, c, d = z3.Bools('a b c d')
        diagnostic = self.run_case([z3.Not(z3.And(a, b)), z3.Not(z3.And(c, d))],
                                   [a, b, c, d])
        self.assertTrue(diagnostic.last['alternative'])
        self.assertEqual(diagnostic.last['probes'], 1)
        self.assertEqual(diagnostic.summary()['fraction_with_alternative'], 1)

    def test_overlapping_conflicts(self):
        a, b, c = z3.Bools('a b c')
        d = self.run_case([z3.Not(z3.And(a, b)), z3.Not(z3.And(a, c))], [a, b, c])
        self.assertTrue(d.last['alternative'])

    def test_hard_context_and_negative_literal(self):
        a, b = z3.Bools('a b')
        d = self.run_case([z3.Implies(b, a)], [z3.Not(a)], hard=[b])
        self.assertFalse(d.last['alternative'])
        self.assertEqual(d.last['core_size'], 1)

    def test_disabled_sampling(self):
        a = z3.Bool('a')
        d = self.run_case([z3.Not(a)], [a], rate=0)
        self.assertEqual(d.unsat_seen, 1)
        self.assertIsNone(d.last)
        self.assertIsNone(d.summary()['fraction_with_alternative'])

    def test_empty_core(self):
        a = z3.Bool('a')
        d = self.run_case([z3.BoolVal(False)], [a])
        self.assertTrue(d.last['inconclusive'])
        self.assertEqual(d.last['probes'], 0)
        self.assertIsNone(d.summary()['fraction_with_alternative'])

    def test_unknown_is_inconclusive(self):
        from unittest.mock import patch
        a, label = z3.Bools('a label')
        solver = z3.Solver()
        diagnostic = Diagnostic(1)
        with patch.object(solver, 'check', return_value=z3.unknown):
            diagnostic.observe(solver, [], [(label, a)], [label])
        self.assertEqual(diagnostic.summary()['unknown_probes'], 1)
        self.assertEqual(diagnostic.summary()['inconclusive'], 1)
        self.assertIsNone(diagnostic.summary()['fraction_with_alternative'])
        self.assertEqual(solver.num_scopes(), 0)

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            Diagnostic(1.1)


class GradientIntegrationTests(unittest.TestCase):
    def test_diagnostic_preserves_original_gradient(self):
        try:
            import torch
            from smtlayer import SMTLayer
        except ImportError:
            self.skipTest('PyTorch is required for the integration test')
        a, b, output = z3.Bools('input_a input_b output')
        results = []
        # With output=false, either positive input is a conflict. The probe
        # necessarily finds an alternative but must not replace the saved core.
        for enabled in (False, True):
            layer = SMTLayer(2, 1, [output == z3.Or(a, b)],
                             variables=[a, b, output],
                             default_mask=torch.tensor([1., 1., 0.]),
                             measure_alternative_conflicts=enabled,
                             alternative_conflict_sample_rate=1.)
            x = torch.tensor([[0.7, 0.9, 0.]], requires_grad=True)
            y = layer(x)
            y.sum().backward()
            results.append((y.detach(), x.grad.clone()))
            self.assertEqual(layer._solver.num_scopes(), 0)
            if enabled:
                self.assertEqual(layer.alternative_conflicts.summary()['with_alternative'], 1)
        self.assertTrue(torch.equal(results[0][0], results[1][0]))
        self.assertTrue(torch.equal(results[0][1], results[1][1]))


if __name__ == '__main__':
    unittest.main()
