# Bayesian Poisson tensor factorization
Source code for the paper: [Bayesian Poisson Tensor Factorization for Inferring Multilateral Relations from Sparse Dyadic Event Counts](http://arxiv.org/abs/1506.03493) by Aaron Schein, John Paisley, David M. Blei, and Hanna Wallach, in KDD 2015.

The MIT License (MIT)

Copyright (c) 2015 Aaron Schein

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## What's included:

* [bptf.py](https://github.com/aschein/bptf/blob/master/src/bptf/bptf.py): The main code file.  Implements batch variational inference for BPTF.
* [tensor_utils.py](https://github.com/aschein/bptf/blob/master/src/bptf/tensor_utils.py): Includes some useful/efficient utils and tensor operations, including a faster re-implementation of tensorly's unfolding_dot_khatri_rao.
* [ICEWS.py](https://github.com/aschein/bptf/blob/master/examples/ICEWS/ICEWS.py): An example application of using BPTF to factorize the ICEWS tensor discussed in the paper.
* [anomaly_detection.py](https://github.com/aschein/bptf/blob/master/examples/anomaly_detection/anomaly_detection.py): An example application of using BPTF for anomaly detection.

## Dependencies:

* tensorly
* scikit-learn
* sparse

## Installation

First, clone the repository:

```bash
git clone https://github.com/aschein/bptf.git
cd bptf
```

Use one [Miniforge](https://github.com/conda-forge/miniforge) installation and a
dedicated `bptf` conda environment. Keep `base` for managing conda, disable its
automatic activation, and use conda-forge with strict channel priority:

```bash
conda config --set auto_activate false
conda config --set channel_priority strict
conda env create --file environment.yml
conda activate bptf
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

`environment.yml` specifies Python 3.13, the scientific dependencies, and the
development tools. Conda installs these packages before pip installs only this
repository in editable mode; changes to its source are immediately available.
Use `python -m pip` inside the activated environment, without `--user` or `sudo`.

For subsequent sessions:

```bash
conda activate bptf
python examples/ICEWS/ICEWS.py  # Requires the example's input data.
conda deactivate
```

The environment lives under Miniforge's `envs/bptf` directory. Keep environment
directories and package caches local; version-control their specifications.
Update `environment.yml` when dependencies change, and prefer rebuilding from
the specification over accumulating ad hoc conda/pip changes.

### Reproduce the tested Apple Silicon environment

`conda-osx-arm64.explicit.txt` records the exact conda package artifacts and their
SHA-256 hashes. It is specific to Apple Silicon macOS; use `environment.yml` on
other platforms. From this repository, with the intended source revision checked
out and no existing environment with the chosen name:

```bash
conda create --name bptf --file conda-osx-arm64.explicit.txt
conda activate bptf
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
```

The explicit file covers conda dependencies; the editable BPTF package comes from
this checkout. After an intentional dependency update and validation, refresh it:

```bash
conda list --name bptf --explicit --sha256 > conda-osx-arm64.explicit.txt
```

## Inferring factors for new cells

For a cells-by-genes count matrix, fit on the training cells, then infer factors
for any number of additional cells with the same genes in the same order:

```python
model = BPTF(data_shape=Y_train.shape, n_components=K)
model.fit(Y_train, max_iter=200, verbose=False)

cell_factors = model.transform(Y_new, mode=0, expectation="geometric")
assert cell_factors.shape == (Y_new.shape[0], K)
```

`transform(data, *, mode=...)` accepts one integer axis and returns one factor
matrix. This replaces the old `transform(modes=[...], data=...)` interface. Only
the selected axis may change size. Other axes must retain their training entities
and ordering; the method checks their sizes but does not track entity labels.

The trained model is unchanged: its factor distributions, hyperparameters, and
training reconstruction are preserved. Inference retains both the arithmetic
and geometric expectations of the fixed factors. Use `expectation="arithmetic"`
to return the inferred Gamma means instead of geometric expectations.

Initialization is deterministic and convergence is checked separately for each
row. Concatenating results from batches gives the same factors as transforming
all rows together, up to floating-point error. `max_iter` defaults to 200 and
`tol` to 1e-4; here `tol` measures relative shape-parameter change per row. An
unfinished projection emits `sklearn.exceptions.ConvergenceWarning`.

Dense NumPy arrays and zero-filled `sparse.COO` tensors are supported. An optional
binary `mask` has the data's shape; entries equal to `missing_val` (default 1)
are excluded. Unstored sparse counts are observed zeros unless masked out.

Run the projection regression tests in the `bptf` environment:

```bash
python -m pytest -q
```

## How to run the code:

### Data format
Input data tensor must be stored as `numpy.ndarray` (for dense tensors) or as `sparse._coo.core.COO` (for sparse tensors).  If you pass a  `numpy.ndarray` in, the code will check if it is sparse, and preprocess it into a `sparse._coo.core.COO`

```python

import numpy as np
from pathlib import Path
from bptf import BPTF, save_bptf, load_bptf

# create a random sparse count tensor 
Y = np.random.poisson(0.5, size=(20, 30, 50))

# create a BPTF model with K = 5
model = BPTF(data_shape=Y.shape, n_components=5)

# fit the model
model.fit(Y,  max_iter=100, verbose=False)

# save the model
save_bptf(model, filepath=Path('model.npz'))

# load the model back in
model = load_bptf(Path('model.npz'))

# get factor matrices (geometric expectations)
G_DK_M = model.G_DK_M

# check the shapes
assert G_DK_M[0].shape == (20, 5)
assert G_DK_M[1].shape == (30, 5)
assert G_DK_M[2].shape == (50, 5)

# the arithmetic expectations of factor matrices are also available
E_DK_M = model.E_DK_M

# check the shapes
assert E_DK_M[0].shape == (20, 5)
assert E_DK_M[1].shape == (30, 5)
assert E_DK_M[2].shape == (50, 5)

# get the variational parameters (shape/rate)
shp_DK_M = model.shp_DK_M
rte_DK_M = model.rte_DK_M

# get the estimated rate hyperparameters
beta_M = model.beta_M
```
The naming convention `_DK_M` tells you the size/shape of the object.  A variable with one underscore---e.g., `E_DK`---is a `numpy.ndarray` that has shape (D, K).  A variable with two underscores---e.g., `E_DK_M`---is a Python `list` of M `numpy.ndarray` objects each with shape (Dm, K), where the first dimension (Dm) may be different across the M arrays but the second dimension K (number of components) is shared.

Either of the two lists `E_DK_M` and `G_DK_M` correspond to the usual list of factor matrices returned by any tensor factorization (CP-decomposition/PARAFAC) model.  They are two different point-estimates (i.e., expectations) of the factor matrices under the variational distribution.  For more information on the two expectations, see Section 7 of the paper.   