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

There is no conflict cache, additional core enumeration, or change to gradient
construction. Disabled diagnostics perform no probes. Enabled probes preserve
assertions and use the original core for the current gradient, but extra checks
can affect Z3's internal search state and subsequent model/core choices; identical
training trajectories are not guaranteed when reusing the same solver.

Sanity checks: `python -m unittest discover -s tests -v` (Z3 required; the gradient
integration check additionally requires PyTorch). Full Sudoku training requires
the existing dataset, dependencies, and CUDA setup.
