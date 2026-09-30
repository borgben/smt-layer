"""Optional probes of alternative conflicts; never selects a gradient core."""
from collections import Counter
import random

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
        probes = unknowns = 0
        alternative = False
        solver.push()
        try:
            solver.add(*hard)
            for index in core_indices:
                result = solver.check(*(assumptions[:index] + assumptions[index + 1:]))
                probes += 1
                if result == z3.unsat:
                    alternative = True
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
                         alternative=alternative, inconclusive=inconclusive)
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
        return dict(unsat_seen=self.unsat_seen, **report(total),
                    by_core_size={size: report(counts)
                                  for size, counts in sorted(self.by_core_size.items())})
