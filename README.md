# SMT Layer

**Code coming soon!**

Implementation of SMTLayer in Pytorch with Z3
This code is the python companion to **Grounding Neural Inference with Satisfiability Modulo Theories**, which is appearing in NeurIPS 2023.

 In this paper we present a set of techniques for integrating Satisfiability Modulo Theories (SMT) solvers into the forward and backward passes of a deep network layer, called SMTLayer. **Notably, the solver needs not be differentiable.** We implement SMTLayer as a Pytorch module. An overview of our work is shown as follows.

<img width="1026" alt="smt_layer" src="./images/teaser_figure.png">

SMTLayers, when used on top of other neural network layers, can be leveraged to solve many tasks requiring logical reasoning. For example, the addition of two digits in an image. Morever, we show how to do visual Sudoku, Lier's Puzzle (above), etc.
<img width="660" alt="example" src="./images/mnist_addition.png">


We implement SMTLayer as a Pytorch module, and our empirical results show that it leads to models that 1) require fewer training samples than conventional models, 2) that are robust to certain types of covariate shift, and 3) that ultimately learn representations that are consistent with symbolic knowledge, and thus naturally interpretable.

<img width="1000" alt="table" src="./images/table.png">


## Prerequisites

You may need the following dependencies installed on your system:

- Python (3.6+)
- PyTorch
- torchvision
- matplotlib
- livelossplot
- z3

Depending on your operating system, you can install z3 using different methods. Here's how to install z3 using `pip`:

```bash
pip install z3-solver
```

Otherwise follow the instructions [here](https://github.com/Z3Prover/z3).


## Data Preparation
Datasets used in the paper are included in `notebook/` execept MNIST, which is publicly available. For example, you can download from Pytorch.
```python
import torchvision
from torchvision import transforms

# Define a transformation
transform = transforms.Compose([transforms.ToTensor()])

# Download and transform the training dataset
mnist_train = torchvision.datasets.MNIST(
    '/data/data', 
    train=True, 
    download=True, 
    transform=transform
)

# Download and transform the test dataset
mnist_test = torchvision.datasets.MNIST(
    '/data/data', 
    train=False, 
    download=True, 
    transform=transform
)
```

Replace '/data/data' with the appropriate path where you want to store the dataset.


## Acknowledgments

If you use this code please cite
```
@inproceedings{
    wang2023grounding,
    title={Grounding Neural Inference with Satisfiability Modulo Theories},
    author={Zifan Wang and Saranya Vijaykumar and Kaiji Lu and Vijay Ganesh and Somesh Jha and Matt Fredrikson},
    booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
    year={2023},
    url={https://openreview.net/forum?id=r8snfquzs3}
}
```

## Alternative-conflict diagnostic

The optional Sudoku diagnostic samples UNSAT **backward queries**, which include
neural input literals, fixed inputs (if configured), and gradient-derived corrected
outputs. It does not measure forward prediction inconsistency alone.

```bash
python -m scripts.mnist_sudoku --measure-alternative-conflicts \
    --alternative-conflict-sample-rate 0.1
```

Minimization stays enabled. After saving the original gradient core `K`, the
probe holds the theory and dynamic hard constraints fixed and checks `A - {k}`
for each tracked neural assumption in `K`, stopping at the first UNSAT result.
The original `assert_and_track` scope must first be popped: omitting a tracking
label from `check()` would not disable its tracked constraint. Probes use explicit
literal assumptions in a temporary scope on the existing solver.

At the first UNSAT probe, the diagnostic retrieves one alternative minimized
core `K2` using the existing solver configuration, with no additional checks.
It compares neural bit coordinates against the original core `K1`. Each pair
records `k1_size`, `k2_size`, `intersection_size`, `new_coordinates` (`|K2 - K1|`),
`jaccard_overlap` (intersection size divided by union size),
`novelty` (`|K2 - K1| / |K2|`), and `disjoint`.

The epoch JSON's `core_pairs` object reports pair counts, mean and median of
these sizes and ratios, and `fraction_disjoint`. Per-pair scalar records are
available in `layer.alternative_conflicts.core_pairs`, with the latest pair in
`.last['core_pair']`. Empty alternative cores are counted separately and excluded
from pair averages because they identify an inconsistent hard context rather
than neural coordinates. No-pair averages are `null`. These statistics describe
the first alternative found in solver core order, not a uniform sample of all
conflicts. Records accumulate for the training run; no cores are cached.

Each epoch prints cumulative `alternative_conflicts` JSON containing:

- `unsat_seen`, sampled `tested`, and `with_alternative`;
- `fraction_with_alternative`, also broken down in `by_core_size`;
- total `probes`, `mean_probes_to_alternative` among successful searches, and
  `tested_every_member` (including success on the final member);
- `inconclusive` and `unknown_probes`.

The fraction uses resolved sampled examples as its denominator. Empty cores
indicate inconsistent hard constraints and are recorded as inconclusive; unknown
probe results are inconclusive unless another probe establishes UNSAT. Core size
is the number of tracked neural literals, not the number of Sudoku cells. The
latest sample is available as `layer.alternative_conflicts.last`; cumulative
metrics are available through `.summary()`. Sampling uses a private seeded RNG.

There is no conflict cache, enumeration beyond the first alternative, or change to gradient
construction. Disabled diagnostics perform no probes. Enabled probes preserve
assertions and use the original core for the current gradient, but extra checks
can affect Z3's internal search state and subsequent model/core choices; identical
training trajectories are not guaranteed when reusing the same solver.

Sanity checks: `python -m unittest discover -s tests -v` (Z3 required; the gradient
integration check additionally requires PyTorch). Full Sudoku training requires
the existing dataset, dependencies, and CUDA setup.

## Singleton-conflict diagnostic

```bash
python -m scripts.mnist_sudoku --measure-singleton-conflicts \
    --singleton-conflict-sample-rate 0.01
```

This independently sampled diagnostic can run alone or alongside the alternative
core diagnostic. For each sampled UNSAT backward query, it reuses the solver in a
temporary scope with the same corrected outputs and any explicitly fixed inputs.
It first checks that this hard background is SAT, then calls `solver.check(a_i)`
for every non-fixed neural input literal independently. No other neural
assumptions are active in that call. Each UNSAT result counts one singleton
conflict. It costs `1 + number_of_assumptions` extra checks per sampled example
with a SAT background. The original core and gradient calculation are unchanged.

Each epoch prints cumulative `singleton_conflicts` JSON with `tested`, exact-count
mean/median, `histogram`, and `fractions` for `0`, `1`, `2-4`, `5-9`, and `10+`.
`by_core_size` reports the same distribution grouped by original minimized-core
size. `fraction_count_exceeds_core_size` directly compares the singleton count
with the original blame-set size. A sample with core size 1 and singleton count
10 has nine additional individually conflicting bits outside that core.

Per-example scalar records are available at `layer.singleton_conflicts.records`
and the latest sampled record at `.last`, including `core_size`,
`singleton_count`, `assumption_count`, and the run-local `unsat_query` counter
(not a dataset ID). `.summary()` returns the epoch report. Inconsistent or
unknown hard backgrounds and unknown individual probes are counted separately
and excluded from exact-count statistics; partial proven counts remain in the
records. Fractions use `exact_examples` as their denominator. These are bit
conflicts relative to the corrected outputs, not necessarily different cells.
The same solver-search-state caveat as the alternative diagnostic applies.

## Training with all singleton conflicts

The default `--conflict-gradient core` retains the original minimized-core
backward update. `--conflict-gradient all_singletons` checks every non-fixed
neural assumption independently on every UNSAT backward query. If the background
is SAT, all checks are conclusive, and at least one singleton conflict exists,
its full set of bit coordinates receives the existing `grad_minus` update.
Otherwise the original minimized core is used. SAT and unchanged-target paths
remain unchanged. The mode applies only to the SMT backward implementation.

Run a paired comparison with the same settings and seed:

```bash
python -m scripts.mnist_sudoku --conflict-gradient core \
    --seed 0 --epochs 5 --no-early-stop | tee baseline-seed0.log
python -m scripts.mnist_sudoku --conflict-gradient all_singletons \
    --seed 0 --epochs 5 --no-early-stop | tee singletons-seed0.log
```

Repeat with seeds 1 and 2. Seeding covers model initialization, pretraining,
shuffling, and Python's batch-splitting choices. CUDA and solver execution can
still be nondeterministic; identical full trajectories are not guaranteed.
`--no-early-stop` keeps the epoch budget equal. Existing CUDA, data, and dependency
requirements still apply; these commands run pretraining as before.

There is no new gradient normalization: the default Sudoku configuration uses
`grad_scaling=None`, so each selected bit gets its usual gradient magnitude.
Existing optimizer gradient clipping still applies. If using the layer API with
`grad_scaling` set, its existing softmax weights apply over the selected set.
The new mode can increase the total gradient magnitude and takes extra solver
time; both are part of this initial comparison.

Each epoch prints `training_metrics` JSON with the mode, existing train/test
accuracy and loss, training seconds for that epoch, and cumulative
`singleton_gradient` counters (`queries`, `solver_calls`, `singleton_updates`,
`core_fallbacks`). The accuracy fields retain the existing task metric; they
are not a newly introduced exact match metric against solution labels. Compare
accuracy after equal epochs and alongside elapsed time before drawing conclusions.

Diagnostic flags remain optional and independently sampled. They never sample
which training examples get singleton updates. When singleton diagnostics are
enabled alongside the new gradient mode, they reuse that example's probe result
rather than checking its assumptions twice. Their `solver_calls` reports the
checks underlying sampled measurements, not additional diagnostic-only checks.
No caching is introduced.
