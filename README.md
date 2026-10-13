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

## Experimental MARCO union backward

The optional `marco_union_backward=True` argument to `SMTLayer` replaces the
single minimized core in `Z3SolverOp.backward` with the union of at most
`marco_max_muses=8` MUSes. Defaults preserve the original core behavior. This
mode requires `solverop='smt'`.

Install the original [liffiton/MARCO](https://github.com/liffiton/MARCO) source
alongside this project (Python 3.9+, `make`, and a C++ compiler are required):

```bash
python -m pip install -e .
git clone https://github.com/liffiton/MARCO.git ../MARCO
git -C ../MARCO checkout 27e056bd5199ab995cf57473ea17c6a2762ff564
make -C ../MARCO/src/pyminisolvers
make -C ../MARCO/src/pyminisolvers test
export PYTHONPATH="$(realpath ../MARCO):$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

MARCO is optional; its source root must be on `PYTHONPATH` when enabled. The
SMT path needs `z3-solver` (already an SMTLayer dependency) and the built MiniSAT
bindings; MUSer2 is unnecessary. The commit above is the version tested here.

The adapter in `smtlayer/marco_union.py` calls upstream `MarcoPolo.enumerate`,
`Z3SubsetSolver` shrinking/growing, and `MinisatMapSolver` directly, in process,
with no parallel workers. We inspected upstream README, `lib_examples.py`, and
these implementations. The top-level file API makes each assertion selectable
and configures processes/signals, so the adapter instead supplies an in-memory
subset solver. Only neural assignment literals receive selectors; theory,
corrected-output literals, and existing fixed inputs remain hard. Upstream
1-based assignment indices are converted to zero-based indices, then mapped
back through the input mask to original logit positions. Fresh selectors and
per-call caches avoid symbol collisions and cross-call solver state.

The corrected output remains `sign(sign(y) - 2*sign(grad_output))`. Neural
assignments use the existing literal conversion (negative logits mean false;
zero retains the existing nonnegative convention). The only gradient change is
`indices = sorted(union)` instead of the single core's indices. Each selected
index still receives the existing `grad_minus`, calculated by
`BCEWithLogitsLoss(x, 1 - relu(sign(x)))`; its mean reduction, optional
`grad_scaling`, and forward MaxSAT flip handling are unchanged. With nonzero
logits and no scaling/flips this is
`(sigmoid(u_i) - (1 - zbar_i)) / x.numel()` on the union, and zero elsewhere.
The no-output-correction branch and forward pass are unchanged.

Enumeration stops after the requested number of MUSes (MSS results do not count).
This caps MUS count, not elapsed time. Solver `unknown` raises an error rather
than being classified as SAT/UNSAT. Enumeration errors emit a warning; any
already discovered MUSes are retained. If no MUS was discovered, the original
single core is used and the fallback counter increments. An `unknown` from the
original backward solver raises, since no valid fallback core is available.

`layer.marco_stats` contains cumulative `calls`, `muses_found`, `union_size`
(sum of per-call unique union sizes), `enumeration_seconds`, and `fallbacks`.
The Sudoku trainer prints this dictionary each epoch through its existing
`tqdm` logging.

Example training (requires the existing Sudoku data and CUDA setup):

```bash
python scripts/mnist_sudoku.py --marco_union_backward --marco_max_muses 8 --epochs 5
```

Run the small CPU tests after installing MARCO as above:

```bash
python -m unittest discover -s tests -v
```

### Seeded Sudoku splits

Add `--seed 0` to reproduce a randomized Sudoku train/test split and initialize
Python, NumPy, and PyTorch training randomness. Different seeds produce different
splits. Images, labels, and masks are shuffled together after preprocessing;
the cached dataset remains unchanged. Pretraining uses the same permutation as
training, keeping pretraining examples out of the test set. Existing split sizes
and the reserved test region starting at `max(9000, nTrain)` are preserved.
Omitting `--seed` retains the original ordered split. Repeated trials use the
same split and advance the seeded training random sequence. GPU operations may
still introduce nondeterminism.

```bash
python -u -m scripts.mnist_sudoku \
  --pct 0.5 --pretrain_epochs 30 --epochs 5 \
  --batch_size 1 --lr 1.0 --clip_norm 0.1 \
  --seed 0 --marco_union_backward --marco_max_muses 8 \
  2>&1 | tee marco-union-seed0.log
```

### Diminishing returns sweep (core baseline, then K = 1 through 8)

After setting up MARCO and `PYTHONPATH` above, run:

```bash
python -u -m scripts.marco_sweep \
  --pct 0.5 --pretrain_epochs 30 --epochs 5 \
  --batch_size 1 --lr 1.0 --clip_norm 0.1 --seed 0 \
  --output marco-sweep.csv \
  2>&1 | tee marco-sweep.log
```

This CUDA experiment pretrains once, saves `marco-sweep.pretrained.pt` (a full
model state dictionary), and reloads that same state for the original single-core baseline first, then
for every K. To reuse an
existing full-model checkpoint, pass `--pretrained_checkpoint PATH`; pretraining
is then skipped. Use a checkpoint prepared with the same dataset split/seed.
Each K resets Python, NumPy, PyTorch/CUDA, and Z3 seeds, uses fresh optimizer and
scheduler state, and runs the same fixed epoch budget without early stopping.
Dataset loaders are reused, preserving the split across K. GPU operations may
still be nondeterministic. K controls the existing union of the first K MUSes;
the surrogate gradient is unchanged.

The CSV is flushed after each K and has nine rows on successful completion. The first row has `mode=core` and
an empty K; the following rows have `mode=marco_union` and K=1 through 8.
The core baseline disables MARCO and has zero MARCO statistics.
Accuracy/loss are from the final epoch: training metrics aggregate the updates
in that epoch, and test metrics evaluate the final epoch model. Losses are
sample-weighted means of the existing masked BCE loss (including its existing
mask normalization). `training_seconds` sums synchronized training-epoch wall
time, including enumeration and optimizer work; it excludes shared pretraining
and test evaluation. `marco_enumeration_seconds` and `muses_found` cover all
training epochs. `mean_mus_size` is total discovered MUS membership divided by
MUS count. `mean_union_size` is summed unique union sizes divided by enumeration
calls, including zero-size fallback calls. No-call/no-MUS means are recorded as
zero. `marco_calls` and `fallbacks` are included to identify incomplete runs.
Hyperparameters, seed, and checkpoint path are recorded in every row.
