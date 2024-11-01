import numpy as np
import numpy.random as rn
import sktensor as skt

import pickle
from path import Path
from time import sleep


def is_binary(X):
    """Checks whether input is a binary integer tensor."""
    if np.issubdtype(X.dtype, int):
        if isinstance(X, skt.sptensor):
            return (X.vals == 1).all()
        else:
            return (X <= 1).all() and (X >= 0).all()
    else:
        return False


def is_sparse(X):
    """Checks whether input tensor is sparse.

    Implements a heuristic definition of sparsity.

    A tensor is considered sparse if:

    M = number of modes
    S = number of entries
    I = number of non-zero entries

            S > M(I + 1)
    """
    M = X.ndim
    S = X.size
    I = X.nonzero()[0].size
    return S > (I + 1) * M


def sptensor_from_dense_array(X):
    """Creates an sptensor from an ndarray or dtensor."""
    subs = X.nonzero()
    vals = X[subs]
    return skt.sptensor(subs, vals, shape=X.shape, dtype=X.dtype)


def preprocess(X):
    """Preprocesses input dense data tensor.

    If data is sparse, returns an int sptensor.
    Otherwise, returns an int dtensor.
    """
    if isinstance(X, skt.sptensor):
        if not np.issubdtype(X.dtype, int):
            X.vals = X.vals.astype(int)
            X.dtype = int
        return X
    else:
        if not np.issubdtype(X.dtype, int):
            X = X.astype(int)
        if is_sparse(X):
            return sptensor_from_dense_array(X)
        else:
            if not isinstance(X, skt.dtensor):
                return skt.dtensor(X)


def uttkrp(X, m, U):
    """
    Alternative implementation of uttkrp in sktensor library.

    The description of that method is copied below:

    Unfolded tensor times Khatri-Rao product:
    :math:`Z = \\unfold{X}{3} (U_1 \kr \cdots \kr U_N)`
    Computes the _matrix_ product of the unfolding
    of a tensor and the Khatri-Rao product of multiple matrices.
    Efficient computations are perfomed by the respective
    tensor implementations.
    Parameters
    ----------
    U : list of array-likes
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode `mode`.
    m : int
        Mode in which the Khatri-Rao product of `U` is multiplied
        with the tensor.
    Returns
    -------
    Z : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of `U`.
    See also
    --------
    For efficient computations of unfolded tensor times Khatri-Rao products
    for specialiized tensors see also
    dtensor.uttkrp, sptensor.uttkrp, ktensor.uttkrp, ttensor.uttkrp
    References
    ----------
    [1] B.W. Bader, T.G. Kolda
        Efficient Matlab Computations With Sparse and Factored Tensors
        SIAM J. Sci. Comput, Vol 30, No. 1, pp. 205--231, 2007
    """
    D, K = U[m].shape
    Z = np.rollaxis(X, m, 0)
    order = range(X.ndim)
    order.remove(m)
    Z = np.dot(Z, U[order[-1]])
    for mode in reversed(order[:-1]):
        Z *= U[mode]
        Z = Z.sum(axis=-2)
    return Z


def sp_uttkrp(vals, subs, m, U):
    """Alternative implementation of the sparse version of the uttkrp.
    ----------
    subs : n-tuple of array-likes
        Subscripts of the nonzero entries in the tensor.
        Length of tuple n must be equal to dimension of tensor.
    vals : array-like
        Values of the nonzero entries in the tensor.
    U : list of array-likes
        Matrices for which the Khatri-Rao product is computed and
        which are multiplied with the tensor in mode `mode`.
    m : int
        Mode in which the Khatri-Rao product of `U` is multiplied
        with the tensor.
    Returns
    -------
    out : np.ndarray
        Matrix which is the result of the matrix product of the unfolding of
        the tensor and the Khatri-Rao product of `U`.
    """
    D, K = U[m].shape
    out = np.zeros_like(U[m])
    for k in range(K):
        tmp = vals.astype(float)
        for mode, matrix in enumerate(U):
            if mode == m:
                continue
            tmp *= matrix[subs[mode], k]
        out[:, k] += np.bincount(subs[m],
                                 weights=tmp,
                                 minlength=D)
    return out


def parafac(matrices, axis=None):
    """Computes the PARAFAC of a set of matrices.

    For a set of N matrices,

        U = {U1, U2, ..., UN}

    where Ui is size (Di X K),

    PARAFAC(U) is defined as the sum of column-wise outer-products:

        PARAFAC(U) = \sum_k U1(:, k) \circ \dots \circ UN(:, k)

    and results in a tensor of size D1 x ... x DN.

    Calls np.einsum repeatedly (instead of all at once) to avoid memory
    usage problems that occur when too many matrices are passed to einsum.

    Parameters
    ----------
    matrices : list of array-likes
        Matrices for which the PARAFAC is computed
    axis : int, optional
        The axis along which all matrices have the same dimensionality.
        Either 0 or 1.  If set to None, it will check which axis
        all matrices agree on. If matrices are square, it defaults to 1.
    Returns
    -------
    out : np.ndarray
        ndarray which is the result of the PARAFAC
    """
    assert len(matrices) > 1
    if axis is None:
        N, M = matrices[0].shape
        axis_0_all_equal = all([X.shape[0] == N for X in matrices[1:]])
        axis_1_all_equal = all([X.shape[1] == M for X in matrices[1:]])
        if axis_1_all_equal:
            axis = 1
        elif axis_0_all_equal:
            axis = 0
        else:
            raise ValueError('Matrices not aligned.')

    if len(matrices) == 2:
        s = 'za,zb->ab' if axis == 0 else 'az,bz->ab'
        return np.einsum(s, matrices[0], matrices[1])
    else:
        s = 'za,zb->zab' if axis == 0 else 'az,bz->abz'
        tmp = np.einsum(s, matrices[0], matrices[1])
        curr = 'ab'

        letters = list('cdefghijklmnopqrstuv')
        for matrix in matrices[2:-1]:
            ltr = letters.pop(0)
            if axis == 0:
                s = 'z%s,z%s->z%s%s' % (curr, ltr, curr, ltr)
            else:
                s = '%sz,%sz->%s%sz' % (curr, ltr, curr, ltr)
            tmp = np.einsum(s, tmp, matrix)
            curr += ltr

        ltr = letters.pop(0)
        if axis == 0:
            s = 'z%s,z%s->%s%s' % (curr, ltr, curr, ltr)
        else:
            s = '%sz,%sz->%s%s' % (curr, ltr, curr, ltr)
        return np.einsum(s, tmp, matrices[-1])


def serialize_bptf(model, out_dir, num=None, desc=None):
    if desc is None:
        desc = 'model'
    out_dir = Path(out_dir)
    assert out_dir.exists()

    if num is None:
        sleep(rn.random() * 5)
        curr_files = out_dir.files('*_%s.npz' % desc)
        curr_nums = [int(f.namebase.split('_')[0]) for f in curr_files]
        num = max(curr_nums + [0]) + 1

    with open(out_dir.joinpath('%d_%s.dat' % (num, desc)), 'wb') as f:
        pickle.dump(model.get_params(), f)

    out_path = out_dir.joinpath('%d_%s.npz' % (num, desc))
    np.savez(out_path,
             E_DK_M=model.E_DK_M,
             G_DK_M=model.G_DK_M,
             gamma_DK_M=model.gamma_DK_M,
             delta_DK_M=model.delta_DK_M,
             beta_M=model.beta_M)
    print (out_path)
    return num
