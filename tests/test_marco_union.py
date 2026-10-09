import unittest
from unittest.mock import patch

import torch
import z3

from smtlayer import SMTLayer
from smtlayer.marco_union import CheckedSolver, enumerate_muses


class MarcoUnionTests(unittest.TestCase):
    def setUp(self):
        self.a, self.b, self.c, self.y = z3.Bools('a b c y')

    def gradient(self, rules, enabled=True, limit=8, corrected=True, fixed=None):
        # Interleave output and input variables to test original-logit indexing.
        layer = SMTLayer(3, 1, rules, variables=[self.a, self.y, self.b, self.c],
                         default_mask=torch.tensor([1., 0., 1., 1.]),
                         fixed_inputs=fixed, marco_union_backward=enabled,
                         marco_max_muses=limit)
        x = torch.tensor([[1., 0.7, 2., 3.]], requires_grad=True)
        out = layer(x)
        out.backward(torch.ones_like(out) if corrected else torch.zeros_like(out))
        return x.grad, layer, out

    def test_singletons_and_hard_background(self):
        background = [z3.Not(self.a), z3.Not(self.b)]
        muses = list(enumerate_muses(background, [self.a, self.b, self.c], 8))
        self.assertEqual({frozenset(m) for m in muses}, {frozenset({0}), frozenset({1})})
        # Each result is UNSAT and every proper one-element deletion is SAT.
        for mus in muses:
            solver = z3.Solver()
            solver.add(background)
            assignments = [self.a, self.b, self.c]
            self.assertEqual(solver.check(*[assignments[i] for i in mus]), z3.unsat)
            for removed in mus:
                self.assertEqual(solver.check(*[assignments[i] for i in mus - {removed}]), z3.sat)
        rules = [z3.Implies(self.a, self.y), z3.Implies(self.b, self.y)]
        grad, layer, _ = self.gradient(rules)
        expected = torch.zeros_like(grad)
        expected[0, [0, 2]] = torch.sigmoid(torch.tensor([1., 2.])) / 4
        torch.testing.assert_close(grad, expected)
        self.assertEqual(layer.marco_stats['muses_found'], 2)
        self.assertEqual(layer.marco_stats['union_size'], 2)
        self.assertEqual(layer.marco_stats['fallbacks'], 0)

    def test_overlapping_muses(self):
        background = [z3.Not(z3.And(self.a, self.b)), z3.Not(z3.And(self.b, self.c))]
        muses = list(enumerate_muses(background, [self.a, self.b, self.c], 8))
        self.assertEqual({frozenset(m) for m in muses},
                         {frozenset({0, 1}), frozenset({1, 2})})
        rules = [z3.Implies(z3.And(self.a, self.b), self.y),
                 z3.Implies(z3.And(self.b, self.c), self.y)]
        grad, layer, _ = self.gradient(rules)
        expected = torch.zeros_like(grad)
        expected[0, [0, 2, 3]] = torch.sigmoid(torch.tensor([1., 2., 3.])) / 4
        torch.testing.assert_close(grad, expected)
        self.assertEqual(layer.marco_stats['union_size'], 3)

    def test_single_mus_gradient_equivalence(self):
        rules = [z3.Implies(z3.And(self.a, self.b), self.y)]
        original, _, old_out = self.gradient(rules, enabled=False)
        union, layer, new_out = self.gradient(rules)
        torch.testing.assert_close(original, union, rtol=0, atol=0)
        torch.testing.assert_close(old_out, new_out, rtol=0, atol=0)
        self.assertEqual(layer.marco_stats['muses_found'], 1)

    def test_disabled_never_calls_marco(self):
        with patch('smtlayer.smtlayer.enumerate_muses', side_effect=AssertionError('called')):
            grad, layer, _ = self.gradient([z3.Implies(self.a, self.y)], enabled=False)
        expected = torch.zeros_like(grad)
        expected[0, 0] = torch.sigmoid(torch.tensor(1.)) / 4
        torch.testing.assert_close(grad, expected)
        self.assertEqual(layer.marco_stats['calls'], 0)

    def test_no_correction_unchanged(self):
        rules = [z3.Implies(self.a, self.y)]
        original, _, _ = self.gradient(rules, enabled=False, corrected=False)
        union, layer, _ = self.gradient(rules, corrected=False)
        torch.testing.assert_close(original, union, rtol=0, atol=0)
        self.assertEqual(layer.marco_stats['calls'], 0)

    def test_limit(self):
        muses = list(enumerate_muses([z3.Not(self.a), z3.Not(self.b)],
                                    [self.a, self.b], 1))
        self.assertEqual(len(muses), 1)

    def test_fallback_on_error_and_no_mus(self):
        rules = [z3.Implies(self.a, self.y)]
        original, _, _ = self.gradient(rules, enabled=False)
        for kwargs in ({'side_effect': RuntimeError('unknown')}, {'return_value': iter(())}):
            with patch('smtlayer.smtlayer.enumerate_muses', **kwargs), self.assertWarns(RuntimeWarning):
                grad, layer, _ = self.gradient(rules)
            torch.testing.assert_close(original, grad, rtol=0, atol=0)
            self.assertEqual(layer.marco_stats['fallbacks'], 1)

    def test_unknown_is_error(self):
        with patch.object(z3.Solver, 'check', return_value=z3.unknown):
            with self.assertRaisesRegex(RuntimeError, 'unknown'):
                CheckedSolver().check()

    def test_unsatisfiable_background_is_error(self):
        with self.assertRaisesRegex(RuntimeError, 'background is UNSAT'):
            list(enumerate_muses([z3.BoolVal(False)], [self.a], 8))

    def test_fixed_inputs_are_hard(self):
        rules = [z3.Implies(z3.And(self.a, self.b), self.y)]
        grad, layer, _ = self.gradient(rules, fixed=[self.a])
        self.assertEqual(grad[0, 0].item(), 0)
        self.assertGreater(grad[0, 2].item(), 0)
        self.assertEqual(layer.marco_stats['union_size'], 1)


if __name__ == '__main__':
    unittest.main()
