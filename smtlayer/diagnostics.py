"""Conflict diagnostics and singleton probes shared with the optional gradient mode."""
from collections import Counter
import random
from statistics import mean, median

import z3


class AlternativeConflictDiagnostic:
    def __init__(self, sample_rate=0.1, seed=0):
        if not 0 <= sample_rate <= 1:
            raise ValueError('alternative conflict sample rate must be in [0, 1]')
        self.sample_rate = sample_rate
        # Do not consume the RNG used by training or data sampling.
        self._rng = random.Random(seed)
        self.unsat_seen = 0
        self.by_core_size = {}
        self.last = None
        # Scalar measurements only; no cores are cached or reused.
        self.core_pairs = []

    def observe(self, solver, hard, tracked, core):
        """Called after popping the original example scope.

        `tracked` contains (tracking label, neural literal) pairs. `hard`
        contains corrected outputs and fixed inputs. The permanent theory is
        already in solver. Save the original core before invoking this method.
        """
        self.unsat_seen += 1
        if self._rng.random() >= self.sample_rate:
            return
        labels = {label.get_id(): i for i, (label, _) in enumerate(tracked)}
        core_indices = [labels[label.get_id()] for label in core]
        assumptions = [literal for _, literal in tracked]
        literal_indices = {literal.get_id(): i for i, literal in enumerate(assumptions)}
        probes = unknowns = 0
        alternative = False
        pair = None
        solver.push()
        try:
            solver.add(*hard)
            for index in core_indices:
                result = solver.check(*(assumptions[:index] + assumptions[index + 1:]))
                probes += 1
                if result == z3.unsat:
                    alternative = True
                    # K1 uses tracking labels; this assumption-based check
                    # returns literals. Compare positions in the same A.
                    k1 = set(core_indices)
                    k2 = {literal_indices[literal.get_id()]
                          for literal in solver.unsat_core()}
                    shared = len(k1 & k2)
                    novel = len(k2 - k1)
                    pair = dict(k1_size=len(k1), k2_size=len(k2),
                                intersection_size=shared, new_coordinates=novel,
                                jaccard_overlap=shared / len(k1 | k2),
                                novelty=novel / len(k2) if k2 else None,
                                disjoint=not bool(shared))
                    self.core_pairs.append(pair)
                    break
                if result == z3.unknown:
                    unknowns += 1
        finally:
            solver.pop()
        size = len(core_indices)
        # An empty core means the hard context itself is inconsistent; the
        # single-removal test cannot identify another neural conflict.
        inconclusive = not alternative and (unknowns > 0 or size == 0)
        self.last = dict(core_size=size, probes=probes,
                         alternative=alternative, inconclusive=inconclusive,
                         core_pair=pair)
        counts = self.by_core_size.setdefault(size, Counter())
        counts.update(tested=1, with_alternative=int(alternative),
                      inconclusive=int(inconclusive), probes=probes,
                      tested_every_member=int(size > 0 and probes == size),
                      probes_to_alternative=probes if alternative else 0,
                      unknown_probes=unknowns)

    def summary(self):
        def report(counts):
            counts = {name: counts[name] for name in (
                'tested', 'with_alternative', 'inconclusive', 'probes',
                'tested_every_member', 'probes_to_alternative', 'unknown_probes')}
            tested = counts['tested']
            resolved = tested - counts['inconclusive']
            found = counts['with_alternative']
            return dict(counts,
                        fraction_with_alternative=found / resolved if resolved else None,
                        mean_probes_to_alternative=(counts['probes_to_alternative'] / found
                                                    if found else None))
        total = Counter()
        for counts in self.by_core_size.values():
            total.update(counts)
        # Empty K2 means inconsistent hard constraints, not a second set of
        # neural coordinates. Retain its record but exclude it from averages.
        pairs = [pair for pair in self.core_pairs if pair['k2_size'] > 0]
        pair_stats = dict(pair_count=len(self.core_pairs),
                          nonempty_pair_count=len(pairs),
                          empty_alternative_cores=len(self.core_pairs) - len(pairs),
                          fraction_disjoint=(mean(p['disjoint'] for p in pairs)
                                             if pairs else None))
        for name in ('k1_size', 'k2_size', 'intersection_size', 'new_coordinates',
                     'jaccard_overlap', 'novelty'):
            values = [pair[name] for pair in pairs]
            pair_stats['mean_' + name] = mean(values) if values else None
            pair_stats['median_' + name] = median(values) if values else None
        return dict(unsat_seen=self.unsat_seen, **report(total),
                    core_pairs=pair_stats,
                    by_core_size={size: report(counts)
                                  for size, counts in sorted(self.by_core_size.items())})


def probe_singletons(solver, hard, tracked):
    """Return conflicting tracking labels and probe statistics; restore assertions."""
    selected = []
    record = dict(assumption_count=len(tracked), singleton_count=None,
                  proven_singletons=0, unknown_probes=0, solver_calls=1)
    solver.push()
    try:
        solver.add(*hard)
        background = solver.check()
        record['background_status'] = str(background)
        if background == z3.sat:
            for label, literal in tracked:
                result = solver.check(literal)
                record['solver_calls'] += 1
                if result == z3.unsat:
                    selected.append(label)
                record['unknown_probes'] += int(result == z3.unknown)
            record['proven_singletons'] = len(selected)
            if not record['unknown_probes']:
                record['singleton_count'] = len(selected)
    finally:
        solver.pop()
    return selected, record


class SingletonConflictDiagnostic:
    """Count individually inconsistent neural literals, without selecting gradients."""

    def __init__(self, sample_rate=0.01, seed=0):
        if not 0 <= sample_rate <= 1:
            raise ValueError('singleton conflict sample rate must be in [0, 1]')
        self.sample_rate = sample_rate
        self._rng = random.Random(seed)
        self.unsat_seen = 0
        self.records = []
        self.last = None

    def observe(self, solver, hard, tracked, core, probe=None):
        self.unsat_seen += 1
        if self._rng.random() >= self.sample_rate:
            return
        if probe is None:
            probe = probe_singletons(solver, hard, tracked)
        _, stats = probe
        record = dict(stats, unsat_query=self.unsat_seen, core_size=len(core))
        self.last = record
        self.records.append(record)

    def summary(self):
        exact = [r for r in self.records if r['singleton_count'] is not None]

        def distribution(records):
            counts = [r['singleton_count'] for r in records]
            bins = dict.fromkeys(('0', '1', '2-4', '5-9', '10+'), 0)
            for count in counts:
                bucket = ('0' if count == 0 else '1' if count == 1 else
                          '2-4' if count <= 4 else '5-9' if count <= 9 else '10+')
                bins[bucket] += 1
            return dict(exact_examples=len(counts),
                        mean_singleton_count=mean(counts) if counts else None,
                        median_singleton_count=median(counts) if counts else None,
                        histogram=dict(sorted(Counter(counts).items())),
                        fractions={k: v / len(counts) if counts else None
                                   for k, v in bins.items()},
                        mean_core_size=mean(r['core_size'] for r in records) if records else None,
                        median_core_size=median(r['core_size'] for r in records) if records else None,
                        fraction_count_exceeds_core_size=(
                            mean(r['singleton_count'] > r['core_size'] for r in records)
                            if records else None))

        return dict(unsat_seen=self.unsat_seen, tested=len(self.records),
                    inconclusive=len(self.records) - len(exact),
                    background_unsat=sum(r['background_status'] == 'unsat' for r in self.records),
                    background_unknown=sum(r['background_status'] == 'unknown' for r in self.records),
                    unknown_probes=sum(r['unknown_probes'] for r in self.records),
                    solver_calls=sum(r['solver_calls'] for r in self.records),
                    **distribution(exact),
                    by_core_size={size: distribution([r for r in exact if r['core_size'] == size])
                                  for size in sorted({r['core_size'] for r in exact})})


class SingletonCacheDiagnostic:
    """Measure recurrence only; never reuse solver results or change gradients."""

    def __init__(self):
        self.seen = {}
        self.context_seen = {}
        self.sets = {}
        self.counts = Counter()

    def observe(self, instance_id, hard, tracked, probe):
        selected, stats = probe
        self.counts['queries'] += 1
        if stats['singleton_count'] is None:
            self.counts['inconclusive_queries'] += 1
            return
        self.counts['exact_queries'] += 1
        labels = {label.get_id() for label in selected}
        cores = frozenset((str(label), literal.sexpr())
                          for label, literal in tracked if label.get_id() in labels)
        seen = self.seen.setdefault(instance_id, set())
        context = tuple(sorted(c.sexpr() for c in hard))
        context_seen = self.context_seen.setdefault((instance_id, context), set())
        self.counts['singleton_occurrences'] += len(cores)
        self.counts['singleton_hits'] += len(cores & seen)
        self.counts['same_context_hits'] += len(cores & context_seen)
        if cores:
            self.counts['nonempty_queries'] += 1
            self.counts['queries_with_hits'] += int(bool(cores & seen))
            sets = self.sets.setdefault(instance_id, set())
            self.counts['whole_set_hits'] += int(cores in sets)
            sets.add(cores)
        seen.update(cores)
        context_seen.update(cores)

    def summary(self):
        names = ('queries', 'inconclusive_queries', 'exact_queries',
                 'singleton_occurrences', 'singleton_hits', 'same_context_hits',
                 'nonempty_queries', 'queries_with_hits', 'whole_set_hits')
        result = {name: self.counts[name] for name in names}
        n = result['singleton_occurrences']
        result.update(instances=len(self.seen),
                      unique_singletons=sum(len(s) for s in self.seen.values()),
                      singleton_hit_rate=result['singleton_hits'] / n if n else None,
                      same_context_hit_rate=result['same_context_hits'] / n if n else None)
        return result
