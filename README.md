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

* [bptf.py](https://github.com/aschein/bptf/blob/master/code/bptf.py): The main code file.  Implements batch variational inference for BPTF.
* [utils.py](https://github.com/aschein/bptf/blob/master/code/utils.py): Utility functions.  Includes some important multilinear algebra functions (e.g., PARAFAC, Khatri-Rao product), preprocessing functions, and serialization functions.
* [anomaly_detection.py](https://github.com/aschein/bptf/blob/master/code/anomaly_detection.py): An example application of using BPTF for anomaly detection.

## Dependencies:

* argparse
* numpy
* path
* pickle
* scikit-learn
* scikit-tensor

## How to run the code:

BPTF can be run from the command line as follows:
```
python bptf.py -d=$DATA_DIR/data.dat -o=$OUT_DIR -k=25
```
#### Required arguments:
* `-d` : Path to the pickled data tensor (see below for details on data format)
* `-o` : Directory to dump results
* `-k` : Number of latent components

#### Optional arguments:
* `-m` : Path to a binary mask (e.g., for prediction experiments), default is None
* `-n` : Maximum number of iterations to run inference, default is 200
* `-t` : Tolerance level (inference stops after percent change in the bound is less than this), default is 1e-4
* `-s` : How the smooth random initialization of variational parameters is, default is 100
* `-a` : Alpha, the prior shape parameter for Gamma-distributed latent factors, default is 0.1 (sparsity-inducing)
* `-v` : Verbose (if toggled, inference will printout information every iteration).

### Data format
Input data tensor must be stored as a `numpy.ndarray` or `sktensor.dtensor` (for dense tensors) or a `sktensor.sptensor` (for sparse tensors).  The object can either be pickled with `pickle.dump` into a file with any extension (e.g., `.dat`) or can be put inside a `.npz` file ([Numpy's compressed file format](http://docs.scipy.org/doc/numpy/reference/routines.io.html)) along with another other associated arrays (e.g., metadata).  If using a `.npz` file, the data tensor's key must be either be `data` or `Y`.  A few examples are given below:

#### Using `numpy.ndarray` or `sktensor.dtensor` and `pickle`
```
import pickle
import numpy as np
import sktensor as skt

data = np.ones((10, 8, 3), dtype=int)  # 3-mode count tensor of size 10 x 8 x 3
# data = skt.dtensor(data)             # Optional: cast numpy.ndarray as sktensor.dtensor

with open('data.dat', 'w+') as f:      # can be stored as a .dat using pickle
  pickle.dump(data, f)

with open('data.dat', 'r') as f:       # can be loaded back in using pickle.load
  tmp = pickle.load(f)
  assert np.allclose(tmp, data)
```
#### Using `sktensor.sptensor` and `pickle`
```
import numpy.random as rn

data = rn.poisson(0.2, size=(10, 8, 3))  # 3-mode SPARSE count tensor of size 10 x 8 x 3

subs = data.nonzero()                    # subscripts where the ndarray has non-zero entries   
vals = data[data.nonzero()]              # corresponding values of non-zero entries
sp_data = skt.sptensor(subs,             # create an sktensor.sptensor 
                       vals,
                       shape=data.shape,
                       dtype=data.dtype)

with open('data.dat', 'w+') as f:            # can be stored as a .dat using pickle
  pickle.dump(data, f)

with open('data.dat', 'r') as f:             # can be loaded back in using pickle.load
  tmp = pickle.load(f)
  assert np.allclose(tmp.vals, sp_data.vals)
```
#### Using `numpy.ndarray` and `numpy.savez` for `.npz` files
```
data = np.ones((10, 8, 3), dtype=int)   # 3-mode count tensor of size 10 x 8 x 3
labels = ['foo', 'bar', 'baz']          # list of index labels for the last mode

with open('data.npz', 'w+') as f:       # both arrays can be stored in a Numpy pickle dictionary
  np.savez(f, data=data, labels=labels) # also see numpy.savez_compressed

data_dict = np.load('data.npz')         # can be loaded back in with numpy.load
tmp_data = data_dict['data']
tmp_labels = data_dict['labels']
assert np.allclose(tmp_data, data)
assert np.allclose(tmp_labels, labels) 
```
# Output:
Results will be serialized in a `.npz` file in the given output directory.  By default, if no results files exist in the provided directory, the results file will be called `1_trained_model.npz`.  If results files are already present, the results file will be named by incrementing the latest (e.g., `6_trained_model.npz` if 5 already exist).  In addition, a file containing the dictionary of initial parameters (e.g., `n_components`, `alpha`) is pickled as `1_trained_model.dat`.  The `.npz` file contains the inferred variational parameters, expectations of the latent factors (computed from the variational parameters), and inferred rate parameters.
```
import numpy as np

results = np.load('1_trained_model.npz')

# Variational parameters
gamma_DK_M = results['gamma_DK_M'] # shape variational parameters
delta_DK_M = results['delta_DK_M'] # rate variational parameters

# Expectations of factors
E_DK_M = results['E_DK_M']  # arithmetic expectation
G_DK_M = results['G_DK_M']  # geometric expectation

# Inferred rate parameters
beta_M = results['beta_M']
```
The naming convention `_DK_M` tells you the size/shape of the object.  A variable with one underscore---e.g., `E_DK`---is a `numpy.ndarray` that has shape (D, K).  A variable with two underscores---e.g., `E_DK_M`---is a Python `list` of M `numpy.ndarray` objects each with shape (Dm, K), where the first dimension (Dm) may be different across the M arrays but the second dimension K (number of components) is shared.

Either of the two lists `E_DK_M` and `G_DK_M` correspond to the usual list of factor matrices returned by any tensor factorization (CP-decomposition/PARAFAC) model.  They are two different point-estimates (i.e., expectations) of the factor matrices under the variational distribution.  For more information on the two expectations, see Section 7 of the paper.   
