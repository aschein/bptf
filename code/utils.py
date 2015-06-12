import numpy as np
import numpy.random as rn
import pickle

from path import path
from time import sleep


def make_first_mode(X, m):
    """ Make a mode the first mode in a tensor. """
    order = range(X.ndim)
    if m == 0:
        return X, order
    Y = X.swapaxes(m-1, m)
    order[m], order[m-1] = order[m-1], order[m]
    m -= 1
    while m > 0:
        Y = Y.swapaxes(m-1, m)
        order[m], order[m-1] = order[m-1], order[m]
        m -= 1
    return Y, order


def parafac(factors):
    if len(factors) == 2:
        return np.einsum('az,bz->ab', *factors)
    if len(factors) == 3:
        tmp = np.einsum('az,bz->abz', factors[0], factors[1])
        return np.einsum('cz,abz->abc', factors[2], tmp)
    elif len(factors) == 4:
        tmp = np.einsum('az,bz->abz', factors[0], factors[1])
        tmp = np.einsum('abz,cz->abcz', tmp, factors[2])
        return np.einsum('abcz,dz->abcd', tmp, factors[3])
    elif len(factors) == 5:
        tmp = np.einsum('az,bz->abz', factors[0], factors[1])
        tmp = np.einsum('abz,cz->abcz', tmp, factors[2])
        tmp = np.einsum('abcz,dz->abcdz', tmp, factors[3])
        return np.einsum('abcdz,ez->abcde', tmp, factors[4])
    else:
        raise NotImplementedError


def serialize_bptf(model, out_dir, num=None, desc=None):
    if desc is None:
        desc = 'model'
    assert model.zeta is not None or model.sparse_zeta is not None
    out_dir = path(out_dir)
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
    print out_path
    return num
