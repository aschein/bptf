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