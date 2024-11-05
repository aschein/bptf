import numpy as np

import sparse
import tensorly.contrib.sparse as tsp

import scipy.special as sp

def is_binary(tensor):
    """Checks whether input is a binary integer tensor."""
    values = tensor if isinstance(tensor, np.ndarray) else tensor.data
    return np.issubdtype(values.dtype, int) and all((values == 1) | (values == 0))


def is_sparse(tensor):
    """Checks whether input ndarray tensor is sparse.

    Implements a heuristic definition of sparsity.

    A tensor is considered sparse if:

    M = number of modes
    S = number of entries
    I = number of non-zero entries

            S > M(I + 1)
    """
    M = tensor.ndim
    S = tensor.size
    I = tensor.nonzero()[0].size
    return S > (I + 1) * M


def preprocess(tensor):
    """Preprocesses input dense data tensor."""
    if isinstance(tensor, np.ndarray) and is_sparse(tensor):
        tensor = tsp.tensor(tensor)

    if not np.issubdtype(tensor.dtype, int):
        tensor = tensor.astype(int)

    return tensor


def unfolding_dot_khatri_rao(tensor, cp_tensor, mode):
    """
    Alternative implementation of tensorly.cp_tensor.unfolding_dot_khatri_rao.

    Can be much faster.
    """
    assert type(tensor) in [np.ndarray, sparse._coo.core.COO]

    weights, factors = cp_tensor

    assert len(factors) == tensor.ndim
    
    if type(tensor) is np.ndarray:
        # This implements the dense version
        D, K = factors[mode].shape
        Z = np.rollaxis(tensor, mode, 0)
        order = [m for m in range(tensor.ndim) if m != mode]
        Z = np.dot(Z, factors[order[-1]])
        for m in reversed(order[:-1]):
            Z *= factors[m]
            Z = Z.sum(axis=-2)
        if weights is not None:
            Z *= weights
        return Z
    
    else:
        # This implements the sparse version
        values, coords = tensor.data, tensor.coords

        D, K = factors[mode].shape
        out = np.zeros_like(factors[mode])
        for k in range(K):
            tmp = values.astype(float)
            for m, factor_matrix in enumerate(factors):
                if mode == m:
                    continue
                tmp *= factor_matrix[coords[m], k]
            
            out[:, k] += np.bincount(coords[mode],
                                     weights=tmp,
                                     minlength=D)
        if weights is not None:
            out *= weights
        return out


def cp_to_tensor_at(cp_tensor, coords):
    weights, factors = cp_tensor

    M = len(factors)
    K = factors[0].shape[1]

    tmp_IK = np.ones((coords[0].size, K))
    for m in range(M):
        tmp_IK *= factors[m][coords[m], :]
    
    if weights is not None:
        tmp_IK *= weights
    
    return tmp_IK.sum(axis=1)


def log_cp_to_tensor_at(cp_tensor, coords):
    weights, factors = cp_tensor

    M = len(factors)
    K = factors[0].shape[1]

    # move to log space
    factors = [np.log(X) for X in factors]
    if weights is not None:
        weights = np.log(weights)

    tmp_IK = np.zeros((coords[0].size, K))
    for m in range(M):
        tmp_IK += factors[m][coords[m], :]
    
    if weights is not None:
        tmp_IK += weights
    
    return sp.logsumexp(tmp_IK, axis=1)
