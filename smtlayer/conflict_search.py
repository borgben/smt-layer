"""Conflict-prior guided search for minimal conflicts in the backward pass."""
from collections import Counter
from itertools import combinations
import random

import z3

from .diagnostics import probe_singletons


def size_template(indices):
    """Default template: the conflict size alone."""
    return len(indices)


class ConflictProfile:
    """Empirical prior over conflict templates, learned from minimized cores."""

    def __init__(self, template_fn=size_template):
        self.template_fn = template_fn
        self.counts = Counter()

    def observe(self, indices):
        if indices:
            self.counts[self.template_fn(tuple(indices))] += 1

    def score(self, indices):
        return self.counts[self.template_fn(tuple(indices))]

    def top(self, k):
        return {template for template, _ in self.counts.most_common(k)}

    def summary(self):
        total = sum(self.counts.values())
        return dict(observed=total,
                    templates=[dict(template=repr(template), count=count,
                                    probability=count / total)
                               for template, count in self.counts.most_common()])


def minimum_hitting_set(conflicts, weights, node_limit=2000):
    """Minimum-weight set of positions meeting every conflict.

    A greedy solution bounds a branch and bound search. If the node limit is
    reached, the best set found so far is returned. It still hits every conflict.
    """
    conflicts = [frozenset(c) for c in conflicts if c]
    chosen, missed = set(), list(conflicts)
    while missed:
        pick = min({p for c in missed for p in c},
                   key=lambda p: weights[p] / sum(p in c for c in missed))
        chosen.add(pick)
        missed = [c for c in missed if pick not in c]
    best = [sum(weights[p] for p in chosen), chosen]
    nodes = 0

    def branch(picked, cost):
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            return
        open_conflict = next((c for c in conflicts if not c & picked), None)
        if open_conflict is None:
            if cost < best[0]:
                best[:] = [cost, set(picked)]
            return
        for p in sorted(open_conflict, key=lambda p: weights[p]):
            if cost + weights[p] < best[0]:
                branch(picked | {p}, cost + weights[p])

    branch(frozenset(), 0.)
    return best[1]


class ConflictSearch:
    """Level-wise search for small conflicts, ordered by a learned template prior.

    The first `warmup` UNSAT queries use the minimized core and add it to the
    profile. After that the search takes over, the core is only a fallback,
    and the profile is updated with the conflicts the search finds.
    `select` probes singletons, then tests pairs of the literals that
    survived, then triples that contain no conflicting pair.
    Candidates are tested in order of template frequency until `budget` solver
    calls are spent. Every SAT and UNSAT result is cached per hard context, so
    a repeated query costs no solver call.
    The prior only proposes candidates. The solver certifies each conflict.
    The predictions to revise are a minimum-weight hitting set of the conflicts.
    """

    def __init__(self, template_fn=size_template, max_size=2, budget=32, warmup=100,
                 order='prior', top_k=None, refinements=0, max_witnesses=8, seed=0,
                 print_every=0):
        if order not in ('prior', 'random'):
            raise ValueError('order must be prior or random')
        self.profile = ConflictProfile(template_fn)
        self.max_size = max_size
        self.budget = budget
        self.warmup = warmup
        self.order = order
        self.top_k = top_k
        self.refinements = refinements
        self.max_witnesses = max_witnesses
        # Print every n-th searched query; 0 prints nothing.
        self.print_every = print_every
        # Do not consume the RNG used by training or data sampling.
        self._rng = random.Random(seed)
        self.stats = Counter()
        self.found_by_size = Counter()
        self.checks_by_size = Counter()
        # hard context -> SAT/UNSAT result per literal set, and one model.
        self.cache = {}

    def _candidates(self, survivors, witness, size, found, var_index):
        # A candidate satisfied by the background model is SAT, so every
        # candidate needs at least one literal that is false in that model.
        false = [p for p in survivors if not witness >> p & 1]
        pool = sorted({tuple(sorted((anchor,) + rest)) for anchor in false
                       for rest in combinations([p for p in survivors if p != anchor], size - 1)})
        # A superset of a known conflict is UNSAT but cannot be minimal.
        pool = [c for c in pool if not any(known <= set(c) for known in found)]
        if self.order == 'random':
            self._rng.shuffle(pool)
            return pool
        key = lambda c: tuple(var_index[p] for p in c)
        if self.top_k is not None:
            allowed = self.profile.top(self.top_k)
            pool = [c for c in pool if self.profile.template_fn(key(c)) in allowed]
        return sorted(pool, key=lambda c: -self.profile.score(key(c)))

    @property
    def active(self):
        """True once the warmup is over and the search replaces the core."""
        return self.stats['queries'] >= self.warmup

    def select(self, solver, hard, tracked, core, weights):
        """Return the tracking labels to revise, or None to keep the given core.

        `tracked` and `hard` are as in `probe_singletons`. `weights` gives the
        confidence of each tracked literal, in the same order. `core` is the
        original core, or None if the caller skipped it. After the warmup the
        core is only a fallback for when the search finds no conflict, and it
        is computed here if it was not given.
        """
        labels = [label for label, _ in tracked]
        literals = [literal for _, literal in tracked]
        var_index = [int(str(label)[3:]) for label in labels]
        by_label = {label.get_id(): p for p, label in enumerate(labels)}
        by_literal = {literal.get_id(): p for p, literal in enumerate(literals)}

        def core_positions():
            if core is not None:
                return sorted(by_label[label.get_id()] for label in core
                              if label.get_id() in by_label)
            self.stats['solver_calls'] += 1
            solver.push()
            try:
                solver.add(*hard)
                solver.check(*literals)
                return sorted(by_literal[lit.get_id()] for lit in solver.unsat_core())
            finally:
                solver.pop()

        def keep_core():
            return None if core is not None else [labels[p] for p in core_positions()]

        self.stats['queries'] += 1
        if self.stats['queries'] <= self.warmup:
            self.stats['warmup_queries'] += 1
            self.profile.observe(var_index[p] for p in core_positions())
            return keep_core()

        # SAT and UNSAT results are reused, but only under the same hard context.
        entry = self.cache.setdefault(tuple(sorted(c.sexpr() for c in hard)),
                                      dict(results={}, model=None))
        results = entry['results']
        keys = [literal.sexpr() for literal in literals]
        unknown = [p for p in range(len(tracked)) if frozenset([keys[p]]) not in results]
        hits = len(tracked) - len(unknown)
        if unknown:
            singles, record = probe_singletons(solver, hard, [tracked[p] for p in unknown])
            self.stats['solver_calls'] += record['solver_calls']
            if record['singleton_count'] is None:
                self.stats['core_fallbacks'] += 1
                return keep_core()
            self.checks_by_size[1] += len(unknown)
            proven = {label.get_id() for label in singles}
            for p in unknown:
                results[frozenset([keys[p]])] = labels[p].get_id() in proven
        single = {p for p in range(len(tracked)) if results[frozenset([keys[p]])]}
        conflicts = [frozenset([p]) for p in sorted(single)]
        survivors = [p for p in range(len(tracked)) if p not in single]
        found = []
        tried = []
        spent = 0

        def truth(model):
            return sum(1 << p for p in survivors
                       if z3.is_true(model.eval(literals[p], model_completion=True)))

        solver.push()
        try:
            solver.add(*hard)
            if self.max_size >= 2 and len(survivors) >= 2 and self.budget > 0:
                if entry['model'] is None:
                    self.stats['solver_calls'] += 1
                    if solver.check() == z3.sat:
                        entry['model'] = solver.model()
                if entry['model'] is not None:
                    witnesses = [truth(entry['model'])]
                    for size in range(2, self.max_size + 1):
                        for cand in self._candidates(survivors, witnesses[0], size, found, var_index):
                            bits = sum(1 << p for p in cand)
                            if any(bits & ~w == 0 for w in witnesses):
                                self.stats['witness_pruned'] += 1
                                continue
                            key = frozenset(keys[p] for p in cand)
                            if key in results:
                                hits += 1
                                unsat = results[key]
                                tried.append((cand, 'cached unsat' if unsat else 'cached sat'))
                            elif spent >= self.budget:
                                continue
                            else:
                                result = solver.check(*[literals[p] for p in cand])
                                spent += 1
                                tried.append((cand, str(result)))
                                self.checks_by_size[size] += 1
                                unsat = result == z3.unsat
                                if result != z3.unknown:
                                    results[key] = unsat
                                if result == z3.sat and len(witnesses) < self.max_witnesses:
                                    witnesses.append(truth(solver.model()))
                            if unsat:
                                found.append(frozenset(cand))
                                self.found_by_size[size] += 1
            conflicts += found
            fallback = None
            if not conflicts:
                # The search found nothing, so the core is the only conflict.
                self.stats['core_fallbacks'] += 1
                fallback = core_positions()
                conflicts.append(frozenset(fallback))
            # The profile keeps learning from every conflict the search uses.
            for conflict in conflicts:
                self.profile.observe(var_index[p] for p in sorted(conflict))
            revise = minimum_hitting_set(conflicts, weights)
            # Optional check that the revision is a correction set: if the
            # remaining literals are still UNSAT, their core is a new conflict.
            for _ in range(self.refinements):
                result = solver.check(*[lit for p, lit in enumerate(literals) if p not in revise])
                self.stats['solver_calls'] += 1
                extra = frozenset(by_literal[lit.get_id()] for lit in solver.unsat_core()) \
                    if result == z3.unsat else None
                if not extra:
                    break
                self.stats['refinement_conflicts'] += 1
                conflicts.append(extra)
                revise = minimum_hitting_set(conflicts, weights)
        finally:
            solver.pop()

        if single:
            self.found_by_size[1] += len(single)
        self.stats['solver_calls'] += spent
        self.stats['candidate_checks'] += spent
        self.stats['cache_hits'] += hits
        self.stats['searched_queries'] += 1
        self.stats['revised_literals'] += len(revise)
        if self.print_every and self.stats['searched_queries'] % self.print_every == 0:
            names = lambda positions: [var_index[p] for p in sorted(positions)]
            print('\nconflict_query {}: singletons={} (singletons probed={}, cache hits={})'.format(
                self.stats['queries'], names(single), len(unknown), hits))
            print('  tried: ' + (', '.join('{}={}'.format(names(c), r) for c, r in tried) or 'nothing'))
            if fallback is not None:
                print('  search found nothing, fell back to core={}'.format(names(fallback)))
            print('  conflicts: {}'.format([names(c) for c in conflicts]))
            print('  revise={}'.format(names(revise)))
        return [labels[p] for p in sorted(revise)]

    def summary(self):
        names = ('queries', 'warmup_queries', 'searched_queries', 'core_fallbacks',
                 'solver_calls', 'candidate_checks', 'cache_hits', 'witness_pruned',
                 'refinement_conflicts', 'revised_literals')
        return dict({name: self.stats[name] for name in names},
                    order=self.order, cached_contexts=len(self.cache),
                    checks_by_size=dict(sorted(self.checks_by_size.items())),
                    found_by_size=dict(sorted(self.found_by_size.items())),
                    profile=self.profile.summary())
