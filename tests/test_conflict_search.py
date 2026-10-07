import unittest
from unittest.mock import patch

import torch
import z3
from smtlayer import SMTLayer, ConflictSearch
from smtlayer.conflict_search import minimum_hitting_set


def identity(indices):
    return tuple(indices)


class HittingSetTests(unittest.TestCase):
    def test_minimum_weight_set_meets_every_conflict(self):
        conflicts = [{1, 4}, {4, 7}, {2, 7, 9}]
        weights = {1: 1., 2: 1., 4: 1., 7: 1., 9: 0.1}
        chosen = minimum_hitting_set(conflicts, weights)
        self.assertEqual(chosen, {4, 9})

    def test_greedy_choice_is_improved(self):
        # Greedy takes the shared cheap-per-conflict element first.
        conflicts = [{0, 1}, {0, 2}, {0, 3}]
        weights = {0: 2.5, 1: 0.5, 2: 0.5, 3: 0.5}
        self.assertEqual(minimum_hitting_set(conflicts, weights), {1, 2, 3})


class ConflictSearchTests(unittest.TestCase):
    def setUp(self):
        self.vars = z3.Bools('cs_a cs_b cs_c cs_d cs_e')
        self.labels = [z3.Bool('var{}'.format(i)) for i in range(5)]

    def query(self, search, theory, n, weights=None):
        solver = z3.Solver()
        solver.set(':core.minimize', True)
        solver.add(*theory)
        tracked = list(zip(self.labels[:n], self.vars[:n]))
        solver.push()
        for label, literal in tracked:
            solver.assert_and_track(literal, label)
        self.assertEqual(solver.check(), z3.unsat)
        core = list(solver.unsat_core())
        solver.pop()
        selected = search.select(solver, [], tracked, core, weights or [1.] * n)
        self.assertEqual(solver.num_scopes(), 0)
        self.assertEqual(len(solver.assertions()), len(theory))
        return None if selected is None else {str(label) for label in selected}

    def test_warmup_keeps_core_and_learns_profile(self):
        a, b = self.vars[:2]
        search = ConflictSearch(template_fn=identity, warmup=1)
        self.assertIsNone(self.query(search, [z3.Xor(a, b)], 2))
        self.assertEqual(search.profile.counts[(0, 1)], 1)
        self.assertEqual(search.stats['solver_calls'], 0)

    def test_pair_conflict_revises_least_confident_literal(self):
        a, b = self.vars[:2]
        search = ConflictSearch(warmup=0)
        selected = self.query(search, [z3.Xor(a, b)], 3, weights=[0.9, 0.2, 0.5])
        self.assertEqual(selected, {'var1'})
        self.assertEqual(search.found_by_size[2], 1)

    def test_singletons_are_always_revised(self):
        a, b, c = self.vars[:3]
        search = ConflictSearch(warmup=0)
        selected = self.query(search, [z3.Not(a), z3.Xor(b, c)], 3, weights=[5., 0.3, 0.4])
        self.assertEqual(selected, {'var0', 'var1'})
        self.assertEqual(search.found_by_size[1], 1)

    def test_found_conflicts_update_the_profile_after_warmup(self):
        a, b, c = self.vars[:3]
        search = ConflictSearch(template_fn=identity, warmup=0)
        self.query(search, [z3.Not(a), z3.Xor(b, c)], 3)
        self.assertEqual(dict(search.profile.counts), {(0,): 1, (1, 2): 1})

    def test_prior_decides_which_candidate_gets_the_budget(self):
        a, b, c, d = self.vars[:4]
        search = ConflictSearch(template_fn=identity, warmup=0, budget=1)
        for _ in range(5):
            search.profile.observe((2, 3))
        selected = self.query(search, [z3.Xor(a, b), z3.Xor(c, d)], 4,
                              weights=[1., 1., 0.1, 1.])
        self.assertEqual(search.stats['candidate_checks'], 1)
        self.assertEqual(search.found_by_size[2], 1)
        # After the warmup the core is not one of the conflicts.
        self.assertEqual(selected, {'var2'})
        self.assertEqual(search.stats['core_fallbacks'], 0)

    def test_core_is_the_fallback_when_search_finds_nothing(self):
        a, b, c = self.vars[:3]
        search = ConflictSearch(template_fn=identity, warmup=0, max_size=2)
        selected = self.query(search, [z3.Not(z3.And(a, b, c))], 3, weights=[0.9, 0.2, 0.5])
        self.assertEqual(selected, {'var1'})
        self.assertEqual(search.stats['core_fallbacks'], 1)
        self.assertEqual(search.profile.counts[(0, 1, 2)], 1)

    def test_skipped_core_is_computed_only_as_fallback(self):
        a, b, c = self.vars[:3]
        tracked = list(zip(self.labels[:3], [a, b, c]))
        solver = z3.Solver()
        solver.set(':core.minimize', True)
        solver.add(z3.Xor(a, b))
        search = ConflictSearch(warmup=0)
        selected = search.select(solver, [], tracked, None, [0.9, 0.2, 0.5])
        self.assertEqual({str(label) for label in selected}, {'var1'})
        self.assertEqual(search.stats['core_fallbacks'], 0)

        solver = z3.Solver()
        solver.set(':core.minimize', True)
        solver.add(z3.Not(z3.And(a, b, c)))
        search = ConflictSearch(warmup=0, max_size=2)
        selected = search.select(solver, [], tracked, None, [0.9, 0.2, 0.5])
        self.assertEqual({str(label) for label in selected}, {'var1'})
        self.assertEqual(search.stats['core_fallbacks'], 1)
        self.assertEqual(solver.num_scopes(), 0)

    def test_top_k_restricts_search_to_learned_templates(self):
        a, b, c, d = self.vars[:4]
        search = ConflictSearch(template_fn=identity, warmup=0, top_k=1)
        for _ in range(5):
            search.profile.observe((2, 3))
        self.query(search, [z3.Xor(a, b), z3.Xor(c, d)], 4)
        self.assertEqual(search.stats['candidate_checks'], 1)

    def test_random_order_with_full_budget_finds_all_pairs(self):
        a, b, c, d = self.vars[:4]
        search = ConflictSearch(warmup=0, order='random')
        selected = self.query(search, [z3.Xor(a, b), z3.Xor(c, d)], 4)
        self.assertEqual(search.found_by_size[2], 2)
        self.assertEqual(len(selected), 2)

    def test_triples_skip_supersets_of_known_pairs(self):
        a, b, c, d, e = self.vars
        search = ConflictSearch(warmup=0, max_size=3, budget=100)
        self.query(search, [z3.Xor(a, b), z3.Not(z3.And(c, d, e))], 5)
        self.assertEqual(dict(search.found_by_size), {2: 1, 3: 1})

    def test_repeated_query_is_answered_from_the_cache(self):
        a, b, c = self.vars[:3]
        search = ConflictSearch(warmup=0)
        theory = [z3.Not(a), z3.Xor(b, c)]
        first = self.query(search, theory, 3, weights=[5., 0.3, 0.4])
        calls = search.stats['solver_calls']
        self.assertGreater(calls, 0)
        self.assertEqual(search.stats['cache_hits'], 0)
        second = self.query(search, theory, 3, weights=[5., 0.3, 0.4])
        self.assertEqual(first, second)
        self.assertEqual(search.stats['solver_calls'], calls)
        self.assertGreater(search.stats['cache_hits'], 0)

    def test_cache_is_not_shared_between_hard_contexts(self):
        a, b = self.vars[:2]
        search = ConflictSearch(warmup=0)
        solver = z3.Solver()
        tracked = list(zip(self.labels[:2], [a, b]))
        self.assertEqual({str(l) for l in search.select(solver, [z3.Not(a)], tracked, [self.labels[0]], [1., 1.])}, {'var0'})
        self.assertEqual({str(l) for l in search.select(solver, [z3.Not(b)], tracked, [self.labels[1]], [1., 1.])}, {'var1'})
        self.assertEqual(search.stats['cache_hits'], 0)
        self.assertEqual(len(search.cache), 2)

    def test_refinement_extends_revision_to_a_correction_set(self):
        a, b, c, d = self.vars[:4]
        theory = [z3.Xor(a, b), z3.Xor(c, d)]
        plain = self.query(ConflictSearch(warmup=0, budget=0), theory, 4)
        refined = self.query(ConflictSearch(warmup=0, budget=0, refinements=2), theory, 4)
        self.assertEqual(len(plain), 1)
        self.assertEqual(len(refined), 2)


class TargetedGradientTests(unittest.TestCase):
    def run_gradient(self, mode, search=None):
        a, output, b = z3.Bools('tg_a tg_output tg_b')
        layer = SMTLayer(2, 1, [output == z3.And(a, z3.Not(b))], variables=[a, output, b],
                         conflict_gradient=mode, conflict_search=search)
        x = torch.tensor([[0.7, 0., -0.9]], requires_grad=True)
        y = layer(x, mask=torch.tensor([[1., 0., 1.]]))
        y.sum().backward()
        self.assertEqual(layer._solver.num_scopes(), 0)
        return x.grad

    def test_targeted_gradient_revises_only_the_hitting_set(self):
        core = self.run_gradient('core')
        targeted = self.run_gradient('targeted', ConflictSearch(warmup=0))
        self.assertEqual(torch.count_nonzero(core).item(), 2)
        self.assertEqual(torch.count_nonzero(targeted).item(), 1)
        torch.testing.assert_close(targeted[0, 0], core[0, 0])

    def test_warmup_matches_core_gradient(self):
        core = self.run_gradient('core')
        targeted = self.run_gradient('targeted', ConflictSearch(warmup=10))
        torch.testing.assert_close(core, targeted)

    def test_core_is_not_extracted_after_warmup(self):
        search = ConflictSearch(warmup=0)
        with patch.object(z3.Solver, 'assert_and_track',
                          side_effect=AssertionError('core was tracked')):
            targeted = self.run_gradient('targeted', search)
        self.assertEqual(torch.count_nonzero(targeted).item(), 1)

    def test_targeted_requires_a_search(self):
        a = z3.Bool('a')
        with self.assertRaises(ValueError):
            SMTLayer(1, 0, [], variables=[a], conflict_gradient='targeted')


if __name__ == '__main__':
    unittest.main()
