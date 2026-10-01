import unittest
from unittest.mock import patch

import z3
import test_alternative_conflicts as existing_tests

Singleton = existing_tests.module.SingletonConflictDiagnostic


class SingletonTests(unittest.TestCase):
    def run_case(self, theory, assumptions, hard=(), rate=1):
        return existing_tests.DiagnosticTests.run_case(self, theory, assumptions, hard, rate, Singleton)

    def test_joint_conflict_has_no_singletons(self):
        a, b = z3.Bools('a b')
        d = self.run_case([z3.Not(z3.And(a, b))], [a, b])
        self.assertEqual(d.last['singleton_count'], 0)
        self.assertEqual(d.last['core_size'], 2)
        self.assertEqual(d.last['solver_calls'], 3)
        self.assertEqual(d.summary()['fractions']['0'], 1)

    def test_multiple_singletons_with_negative_literal(self):
        a, b, c = z3.Bools('a b c')
        d = self.run_case([a, z3.Not(b)], [z3.Not(a), b, c])
        self.assertEqual(d.last['singleton_count'], 2)
        self.assertEqual(d.last['core_size'], 1)
        self.assertEqual(d.last['solver_calls'], 4)
        self.assertEqual(d.summary()['fraction_count_exceeds_core_size'], 1)

    def test_hard_context(self):
        a, b = z3.Bools('a b')
        d = self.run_case([z3.Implies(b, a)], [z3.Not(a)], hard=[b])
        self.assertEqual(d.last['singleton_count'], 1)

    def test_inconsistent_background_excluded(self):
        a = z3.Bool('a')
        d = self.run_case([], [a], hard=[z3.BoolVal(False)])
        self.assertIsNone(d.last['singleton_count'])
        self.assertEqual(d.last['solver_calls'], 1)
        self.assertEqual(d.summary()['background_unsat'], 1)
        self.assertEqual(d.summary()['inconclusive'], 1)
        self.assertEqual(d.summary()['histogram'], {})

    def test_zero_sampling(self):
        a = z3.Bool('a')
        d = self.run_case([a], [z3.Not(a)], rate=0)
        self.assertEqual(d.unsat_seen, 1)
        self.assertEqual(d.summary()['solver_calls'], 0)
        self.assertEqual(d.summary()['tested'], 0)

    def test_unknown_excluded_from_exact_counts(self):
        a, label = z3.Bools('a label')
        solver = z3.Solver()
        d = Singleton(1)
        with patch.object(solver, 'check', side_effect=[z3.sat, z3.unknown]):
            d.observe(solver, [], [(label, a)], [label])
        self.assertEqual(d.last['unknown_probes'], 1)
        self.assertIsNone(d.last['singleton_count'])
        self.assertEqual(d.summary()['exact_examples'], 0)
        self.assertEqual(solver.num_scopes(), 0)

    def test_distribution_and_core_size_comparison(self):
        combined = Singleton(1)
        for count in (0, 1, 2, 4, 5, 9, 10):
            bits = z3.Bools(' '.join('b' + str(i) for i in range(max(2, count))))
            theory = [z3.Not(bit) for bit in bits[:count]] if count else [z3.Not(z3.And(*bits))]
            d = self.run_case(theory, bits)
            combined.records.extend(d.records)
        report = combined.summary()
        self.assertEqual(report['tested'], 7)
        self.assertEqual(report['histogram'], {n: 1 for n in (0, 1, 2, 4, 5, 9, 10)})
        self.assertEqual(report['median_singleton_count'], 4)
        self.assertAlmostEqual(report['mean_singleton_count'], 31 / 7)
        self.assertEqual(report['fractions'], {'0': 1/7, '1': 1/7, '2-4': 2/7, '5-9': 2/7, '10+': 1/7})
        self.assertEqual(report['by_core_size'][2]['mean_singleton_count'], 0)
        self.assertEqual(report['by_core_size'][1]['exact_examples'], 6)

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            Singleton(-0.1)
