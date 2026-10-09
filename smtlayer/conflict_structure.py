"""Sudoku structural priors guide checks, never certify UNSAT or select gradients."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations, permutations
import heapq
import math
import random

import z3


@dataclass(frozen=True)
class ConflictLiteral:
    raw_index: int
    bool_value: bool
    row: int
    col: int
    bit: int
    cell_index: int
    box_id: int

    @classmethod
    def decode(cls, index, value, size=9):
        if size not in (3, 6, 9) or not 0 <= index < 4 * size * size:
            raise ValueError('expected a Sudoku input index and size 3, 6 or 9')
        cell, bit = divmod(index, 4)
        row, col = divmod(cell, size)
        # Size 3 has no box constraints in the existing encoding.
        box = (-cell - 1 if size == 3 else
               (row // (2 if size == 6 else 3)) * (size // 3) + col // 3)
        return cls(index, bool(value), row, col, bit, cell, box)


def relations(a, b):
    return tuple(name for name, equal in (
        ('SAME_CELL', a.cell_index == b.cell_index),
        ('SAME_ROW', a.row == b.row), ('SAME_COL', a.col == b.col),
        ('SAME_BOX', a.box_id == b.box_id)) if equal)


def canonicalize_conflict(literals):
    """Exact labeled-graph canonicalization for the prototype's low orders."""
    if len(literals) > 3:
        raise ValueError('exact canonicalization supports cardinalities 0 through 3')
    return min((('k' + str(len(p)), tuple((n.bit, int(n.bool_value)) for n in p),
                 tuple(relations(a, b) for a, b in combinations(p, 2)))
                for p in permutations(literals)), default=('k0', (), ()))


def describe_conflict(literals):
    edges = [relations(a, b) for a, b in combinations(literals, 2)]
    result = dict(size=len(literals), bit_pattern=sorted(n.bit for n in literals),
                  predicted_value_pattern=sorted(int(n.bool_value) for n in literals))
    for relation in ('CELL', 'ROW', 'COL', 'BOX'):
        result['same_' + relation.lower() + '_pairs'] = sum('SAME_' + relation in e for e in edges)
    if len(literals) == 2:
        result.update({ 'same_' + r.lower(): 'SAME_' + r in edges[0]
                        for r in ('CELL', 'ROW', 'COL', 'BOX')})
        result['same_bit'] = literals[0].bit == literals[1].bit
    return result


class ConflictProfiler:
    def __init__(self, minimal=False):
        self.minimal = minimal
        self.conflict_size_counts = Counter()
        self.template_counts = Counter()
        self.template_counts_by_size = defaultdict(Counter)
        self.sources = Counter()
        self.minimization_inconclusive = 0
        self.minimization_solver_calls = 0

    @property
    def conflict_total(self):
        return sum(self.conflict_size_counts.values())

    def observe_conflict(self, literals, source='core'):
        self.conflict_size_counts[len(literals)] += 1
        self.sources[source] += 1
        # Avoid factorial work on large raw cores; retain their cardinalities.
        if len(literals) <= 3:
            key = canonicalize_conflict(literals)
            self.template_counts[key] += 1
            self.template_counts_by_size[len(literals)][key] += 1

    def top_templates(self, k, limit=None):
        return sorted(self.template_counts_by_size[k].items(), key=lambda x: (-x[1], x[0]))[:limit]

    def template_probability(self, template):
        total = sum(self.template_counts.values())
        return self.template_counts[template] / total if total else 0.

    def size_probability(self, k):
        return self.conflict_size_counts[k] / self.conflict_total if self.conflict_total else 0.

    def summary(self, limit=20):
        total = sum(self.template_counts.values())
        result = dict(total_conflicts=self.conflict_total, unique_templates=len(self.template_counts),
                      size_counts=dict(self.conflict_size_counts), sources=dict(self.sources),
                      core_structure='inclusion-minimized' if self.minimal else 'raw core structure',
                      metric_scope='cumulative training backward queries',
                      minimization_inconclusive=self.minimization_inconclusive,
                      minimization_solver_calls=self.minimization_solver_calls,
                      template_max_k=3, top_templates={k: [dict(template=t, count=n) for t, n in self.top_templates(k, limit)]
                                                     for k in sorted(self.template_counts_by_size)},
                      entropy=-sum((n / total) * math.log2(n / total) for n in self.template_counts.values()) if total else 0.)
        for top in (1, 5, 10):
            result[f'top{top}_coverage_by_k'] = {k: sum(n for _, n in self.top_templates(k, top)) / count
                                                for k, count in self.conflict_size_counts.items() if k <= 3 and count}
        return result


def minimize_core(solver, hard, tracked, core, stats=None):
    """Deletion minimization; unknown never licenses a deletion. Restore scope."""
    by_id = {label.get_id(): literal for label, literal in tracked}
    remaining = list(core)
    solver.push()
    try:
        solver.add(*hard)
        for label in list(remaining):
            trial = [l for l in remaining if l.get_id() != label.get_id()]
            if stats is not None:
                stats.minimization_solver_calls += 1
            result = solver.check(*(by_id[l.get_id()] for l in trial))
            if result == z3.unknown:
                return None  # Do not describe an unproved core as inclusion-minimal.
            if result == z3.unsat:
                remaining = trial
    finally:
        solver.pop()
    return remaining


class StructuralConflicts:
    def __init__(self, size=9, minimal=False, search=False, max_k=3, budget=32,
                 warmup_epochs=1, top_templates=20, order='frequency', seed=0):
        if max_k not in (1, 2, 3) or budget < 0 or warmup_epochs < 0 or top_templates < 1 or order not in ('frequency', 'random'):
            raise ValueError('invalid structural search configuration')
        self.size, self.search, self.max_k, self.budget = size, search, max_k, budget
        self.warmup_epochs, self.top_templates, self.order = warmup_epochs, top_templates, order
        self.profiler = ConflictProfiler(minimal)
        self.rng = random.Random(seed)
        self.epoch = 0
        self.begin_epoch(0)
        self.last = []

    def begin_epoch(self, epoch):
        self.epoch = epoch
        self.stats = Counter()
        self.ranks = []

    def summary(self):
        names = ['queries_requested', 'solver_calls', 'cache_hits', 'cache_misses', 'sat_hits', 'unsat_hits',
                 'budget_exhausted_count', 'background_unsat', 'unknown_results']
        names += [f'{name}_k{k}' for name in ('candidates_generated', 'candidates_tested', 'conflicts_found') for k in (1, 2, 3)]
        result = {n: self.stats[n] for n in names}
        result.update(cache_hit_rate=self.stats['cache_hits'] / self.stats['queries_requested'] if self.stats['queries_requested'] else 0.,
                      mean_rank_of_found_conflict=sum(self.ranks) / len(self.ranks) if self.ranks else None,
                      order=self.order, budget=self.budget, metric_scope='current epoch',
                      recall_omitted='full conflict set is unknown')
        return result

    def observe(self, solver, hard, tracked, core, mask_signature=()):
        # Isolate search heuristics as well as assertions: instrumentation must
        # not perturb which core the training solver chooses on later batches.
        background = list(solver.assertions())
        solver = z3.Solver(ctx=solver.ctx)
        solver.add(*background)
        lookup = {str(label): (int(str(label)[3:]), literal) for label, literal in tracked}
        def decode(label):
            index, literal = lookup[str(label)]
            return ConflictLiteral.decode(index, not z3.is_not(literal), self.size)
        profiled = minimize_core(solver, hard, tracked, core, self.profiler) if self.profiler.minimal else core
        if profiled is None:
            self.profiler.minimization_inconclusive += 1
        else:
            self.profiler.observe_conflict([decode(l) for l in profiled])
        self.last = []
        if not self.search or self.epoch <= self.warmup_epochs:
            return
        # Fresh per-query cache: even identical examples never share solver state.
        # Signature includes every asserted background formula, hard conditions,
        # and the full mask. Keys also retain literal IDs and predicted values.
        context = (tuple(c.sexpr() for c in solver.assertions()), tuple(c.sexpr() for c in hard), tuple(mask_signature))
        cache = {}
        calls = 0
        def check(candidate):
            nonlocal calls
            self.stats['queries_requested'] += 1
            key = (context, frozenset((str(tracked[j][0]), tracked[j][1].sexpr()) for j in candidate))
            if key in cache:
                self.stats['cache_hits'] += 1
                self.stats['sat_hits' if cache[key] == z3.sat else 'unsat_hits'] += 1
                return cache[key]
            self.stats['cache_misses'] += 1
            if calls >= self.budget:
                return None
            calls += 1
            self.stats['solver_calls'] += 1
            result = solver.check(*(tracked[j][1] for j in candidate))
            if result in (z3.sat, z3.unsat):
                cache[key] = result
            else:
                self.stats['unknown_results'] += 1
            if result == z3.unsat and candidate:
                known_unsat.append(frozenset(candidate))
            return result
        nodes = [decode(l) for l, _ in tracked]
        prior = self.profiler.template_counts.copy()  # Freeze ranking for this query.
        known_unsat = []
        singleton_sat = []
        solver.push()
        try:
            solver.add(*hard)
            background = check(())
            if background != z3.sat:
                self.stats['background_unsat'] += int(background == z3.unsat)
                return
            for k in range(1, self.max_k + 1):
                pool = range(len(tracked)) if k == 1 else singleton_sat
                def candidates():
                    for candidate in combinations(pool, k):
                        if any(s.issubset(candidate) for s in known_unsat):
                            continue
                        self.stats[f'candidates_generated_k{k}'] += 1
                        key = canonicalize_conflict([nodes[j] for j in candidate])
                        score = -prior[key] if self.order == 'frequency' else self.rng.random()
                        yield score, candidate
                # Stream enumeration with bounded storage. All shapes remain eligible;
                # unseen templates tie by index. CPU enumeration is outside SMT budget.
                ranked = heapq.nsmallest(self.budget, candidates())
                for rank, (_, candidate) in enumerate(ranked, 1):
                    if calls >= self.budget:
                        break
                    # All immediate subsets SAT implies all proper subsets SAT.
                    if k > 1 and any(check(sub) != z3.sat for sub in combinations(candidate, k - 1)):
                        continue
                    result = check(candidate)
                    if result is None:
                        break
                    self.stats[f'candidates_tested_k{k}'] += 1
                    if k == 1 and result == z3.sat:
                        singleton_sat.append(candidate[0])
                    if result == z3.unsat:
                        self.last.append(tuple(nodes[j] for j in candidate))
                        self.stats[f'conflicts_found_k{k}'] += 1
                        self.ranks.append(rank)
                if calls >= self.budget:
                    break
        finally:
            solver.pop()
            self.stats['budget_exhausted_count'] += int(calls >= self.budget)
        for conflict in self.last:
            self.profiler.observe_conflict(conflict, source='verified_search')
