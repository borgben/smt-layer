import itertools
import unittest
from unittest.mock import patch

import torch
import z3
from smtlayer import SMTLayer
from smtlayer.conflict_structure import (
    ConflictLiteral, ConflictProfiler, StructuralConflicts,
    canonicalize_conflict, describe_conflict, minimize_core)


class StructuralTests(unittest.TestCase):
    def literal(self, row, col, bit=0, value=True, size=9):
        return ConflictLiteral.decode(4 * (row * size + col) + bit, value, size)

    def test_decoding_and_box_layout(self):
        n = ConflictLiteral.decode(4 * (5 * 9 + 7) + 2, False)
        self.assertEqual((n.row, n.col, n.bit, n.cell_index, n.box_id), (5, 7, 2, 52, 5))
        self.assertEqual(self.literal(3, 4, size=6).box_id, 3)
        self.assertNotEqual(self.literal(0, 0, size=3).box_id, self.literal(0, 1, size=3).box_id)
        with self.assertRaises(ValueError):
            ConflictLiteral.decode(324, True)

    def test_equivalence_non_equivalence_and_permutations(self):
        a = [self.literal(0, 0), self.literal(0, 1, 2, False)]
        b = [self.literal(3, 3), self.literal(3, 4, 2, False)]
        self.assertEqual(canonicalize_conflict(a), canonicalize_conflict(b))
        c = [self.literal(0, 0), self.literal(4, 0, 2, False)]
        self.assertNotEqual(canonicalize_conflict(a), canonicalize_conflict(c))
        triple = a + [self.literal(1, 1, 3)]
        for perm in itertools.permutations(triple):
            self.assertEqual(canonicalize_conflict(triple), canonicalize_conflict(perm))
        self.assertTrue(describe_conflict(a)['same_row'])
        self.assertEqual(describe_conflict(triple)['same_box_pairs'], 3)

    def run_search(self, theory_fn, n=4, hard=(), budget=100, order='frequency'):
        variables = [z3.Bool(f'struct_a{i}') for i in range(n)]
        solver = z3.Solver()
        solver.add(*theory_fn(variables))
        tracked = [(z3.Bool(f'var{i}'), a) for i, a in enumerate(variables)]
        structural = StructuralConflicts(search=True, budget=budget, warmup_epochs=0, order=order)
        structural.begin_epoch(1)
        structural.observe(solver, hard, tracked, [])
        self.assertEqual(solver.num_scopes(), 0)
        return structural, solver, tracked

    def test_pruning_pair_eligibility_and_soundness(self):
        theory = lambda a: [z3.Not(a[0]), z3.Not(z3.And(a[1], a[2])),
                            z3.Not(z3.And(a[1], a[3]))]
        obj, solver, tracked = self.run_search(theory)
        sets = [{n.raw_index for n in c} for c in obj.last]
        self.assertEqual(sets, [{0}, {1, 2}, {1, 3}])
        self.assertEqual(obj.summary()['candidates_generated_k2'], 3)
        self.assertEqual(obj.summary()['candidates_generated_k3'], 0)
        for conflict in obj.last:
            ids = [n.raw_index for n in conflict]
            self.assertEqual(solver.check(*(tracked[j][1] for j in ids)), z3.unsat)
            for sub in itertools.combinations(ids, len(ids) - 1):
                self.assertEqual(solver.check(*(tracked[j][1] for j in sub)), z3.sat)

    def test_triple_and_subset_verification(self):
        obj, _, _ = self.run_search(lambda a: [z3.Not(z3.And(*a))], n=3)
        self.assertEqual([len(c) for c in obj.last], [3])
        self.assertGreater(obj.summary()['cache_hits'], 0)

    def test_context_safety(self):
        a, output = z3.Bools('struct_context_a struct_context_output')
        solver = z3.Solver()
        solver.add(a == output)
        tracked = [(z3.Bool('var0'), a)]
        obj = StructuralConflicts(search=True, warmup_epochs=0)
        obj.begin_epoch(1)
        obj.observe(solver, [z3.Not(output)], tracked, [])
        self.assertEqual(len(obj.last), 1)
        obj.begin_epoch(2)
        obj.observe(solver, [output], tracked, [])
        self.assertEqual(obj.last, [])
        self.assertEqual(obj.summary()['solver_calls'], 2)
        self.assertEqual(obj.summary()['cache_hits'], 0)

    def test_equal_budget_and_warmup(self):
        for order in ('frequency', 'random'):
            obj, _, _ = self.run_search(lambda a: [z3.Not(z3.And(*a))], budget=3, order=order)
            self.assertEqual(obj.summary()['solver_calls'], 3)
            self.assertEqual(obj.summary()['budget_exhausted_count'], 1)
        solver = z3.Solver()
        a = z3.Bool('struct_warmup')
        solver.add(z3.Not(a))
        obj = StructuralConflicts(search=True)
        obj.begin_epoch(1)
        obj.observe(solver, [], [(z3.Bool('var0'), a)], [z3.Bool('var0')])
        self.assertEqual(obj.summary()['solver_calls'], 0)
        self.assertEqual(obj.profiler.conflict_total, 1)

    def test_minimization_background_and_unknown(self):
        a, b = z3.Bools('struct_min_a struct_min_b')
        labels = z3.Bools('var0 var1')
        solver = z3.Solver()
        solver.add(z3.Not(a))
        tracked = list(zip(labels, [a, b]))
        self.assertEqual(minimize_core(solver, [], tracked, labels), [labels[0]])
        self.assertEqual(solver.num_scopes(), 0)
        with patch.object(solver, 'check', return_value=z3.unknown):
            self.assertIsNone(minimize_core(solver, [], tracked, labels))
        obj = StructuralConflicts(search=True, warmup_epochs=0)
        obj.begin_epoch(1)
        obj.observe(solver, [a], tracked, [])
        self.assertEqual(obj.last, [])
        self.assertEqual(obj.summary()['background_unsat'], 1)

    def test_profiler_large_cores_and_probabilities(self):
        p = ConflictProfiler()
        nodes = [self.literal(0, j) for j in range(4)]
        p.observe_conflict(nodes)
        p.observe_conflict(nodes[:2])
        p.observe_conflict(nodes[:2])
        self.assertEqual(p.conflict_total, 3)
        self.assertEqual(p.size_probability(2), 2 / 3)
        self.assertEqual(p.template_probability(canonicalize_conflict(nodes[:2])), 1)
        self.assertEqual(p.summary()['top5_coverage_by_k'][2], 1)

    def test_integration_preserves_gradients(self):
        def run(options):
            a, output, b = z3.Bools('struct_grad_a struct_grad_output struct_grad_b')
            layer = SMTLayer(2, 1, [output == z3.Or(a, z3.Not(b))],
                             variables=[a, output, b], conflict_struct_options=options)
            x = torch.tensor([[0.7, 0., -0.9]], requires_grad=True)
            layer(x, mask=torch.tensor([[1., 0., 1.]])).sum().backward()
            return x.grad, layer
        expected, _ = run(None)
        for options in ({}, {'minimal': True}, {'search': True, 'warmup_epochs': 0}):
            actual, layer = run(options)
            torch.testing.assert_close(expected, actual)
            self.assertEqual(layer._solver.num_scopes(), 0)
            self.assertEqual(layer.structural_conflicts.profiler.conflict_total >= 1, True)

    def test_prior_controls_order_but_never_proves_unsat(self):
        a, b = z3.Bools('struct_rank_a struct_rank_b')
        solver = z3.Solver()
        solver.add(z3.Not(b))
        tracked = [(z3.Bool('var0'), a), (z3.Bool('var1'), b)]
        obj = StructuralConflicts(search=True, warmup_epochs=0, budget=2)
        obj.begin_epoch(1)
        for _ in range(10):
            obj.profiler.observe_conflict([ConflictLiteral.decode(1, True)])
        obj.observe(solver, [], tracked, [])
        self.assertEqual([c[0].raw_index for c in obj.last], [1])
        # The same high-frequency shape is SAT under the next background.
        solver = z3.Solver()
        solver.add(b)
        obj.observe(solver, [], tracked, [])
        self.assertEqual(obj.last, [])

    def test_training_solver_is_never_probed(self):
        a = z3.Bool('struct_isolated')
        solver = z3.Solver()
        solver.add(z3.Not(a))
        obj = StructuralConflicts(search=True, minimal=True, warmup_epochs=0)
        obj.begin_epoch(1)
        with patch.object(solver, 'check', side_effect=AssertionError('training solver probed')):
            obj.observe(solver, [], [(z3.Bool('var0'), a)], [z3.Bool('var0')])
        self.assertEqual(len(obj.last), 1)
