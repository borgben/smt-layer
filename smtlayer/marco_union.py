"""Small in-process adapter to liffiton/MARCO (optional external dependency)."""

import z3


class CheckedSolver(z3.Solver):
    """MARCO's SMT adapter assumes a two-valued check; reject unknown."""

    def check(self, *assumptions):
        result = super().check(*assumptions)
        if result == z3.unknown:
            raise RuntimeError("MARCO solver returned unknown: " + self.reason_unknown())
        return result


def enumerate_muses(background, assignments, max_muses):
    """Yield zero-based assignment indices; background is never selectable.

    Uses upstream enumeration, shrinking and map solving unchanged. Unlike the
    file API, only assignments get selectors. Upstream result indices are 1-based.
    """
    from src.marco.SMTsolvers import Z3SubsetSolver
    from src.marco.MarcoPolo import MarcoPolo
    from src.marco.mapsolvers import MinisatMapSolver
    from src.marco.utils import Statistics

    class SubsetSolver(Z3SubsetSolver):
        def __init__(self):
            self.constraints = list(assignments)
            self.n = len(self.constraints)
            # Upstream caches are class attributes; isolate each invocation and
            # use fresh selectors so theory symbols cannot collide with them.
            self.varcache = {i: z3.FreshBool('marco') for i in range(1, self.n + 1)}
            self.idcache = {self.get_id(v): i for i, v in self.varcache.items()}
            self.s = CheckedSolver()
            self.s.add(list(background))
            for i, constraint in enumerate(self.constraints, 1):
                self.s.add(z3.Implies(self.c_var(i), constraint))

    if max_muses < 1:
        raise ValueError('marco_max_muses must be positive')
    subset = SubsetSolver()
    if subset.s.check() != z3.sat:
        raise RuntimeError('MARCO hard background is UNSAT')
    if not assignments:
        return
    config = dict(bias='MUSes', maximize=True, verbose=False, comms_ignore=False)
    enumerator = MarcoPolo(0, subset, MinisatMapSolver(subset.n, bias=True),
                           Statistics(), config)
    results = enumerator.enumerate()
    found = 0
    try:
        for kind, _, indices in results:
            if kind == 'U':
                yield {index - 1 for index in indices}
                found += 1
                if found >= max_muses:
                    break
    finally:
        results.close()
