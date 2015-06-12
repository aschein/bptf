"""
Bayesian Poisson tensor factorization with variational inference.
"""
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
from sklearn.utils.extmath import fast_dot
from sklearn.base import BaseEstimator, TransformerMixin
from path import path
from argparse import ArgumentParser
from utils_ptf import make_first_mode, serialize_bptf, parafac


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, n_modes=4, n_components=100,  max_iter=200, tol=0.0001,
                 smoothness=100, verbose=True, alpha=0.1):
        self.n_modes = n_modes
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.smoothness = smoothness
        self.verbose = verbose

        self.alpha = alpha                                      # shape hyperparameter
        self.beta_M = np.ones(self.n_modes, dtype=float)        # rate hyperparameter (inferred)

        self.gamma_DK_M = np.empty(self.n_modes, dtype=object)  # variational shapes
        self.delta_DK_M = np.empty(self.n_modes, dtype=object)  # variational rates

        self.E_DK_M = np.empty(self.n_modes, dtype=object)      # arithmetic expectations
        self.G_DK_M = np.empty(self.n_modes, dtype=object)      # geometric expectations

        # Inference cache
        self.sumE_MK = np.empty((self.n_modes, self.n_components), dtype=float)
        self.zeta = None
        self.sparse_zeta = None

    def _zeta(self):
        """sum_k exp(sum_m \E[\ln theta_{delta(m) k}^m])"""
        zeta = self.G_DK_M[0].copy()
        for m in xrange(1, self.n_modes):
            zeta = np.expand_dims(zeta, -2) * self.G_DK_M[m]
        return zeta.sum(axis=-1)

    def _sparse_zeta(self):
        """Only calculate zeta at positions where the data is non-zero."""
        sparse_zeta = np.ones((self.n_points, self.n_components))
        for m in xrange(self.n_modes):
            sparse_zeta *= self.G_DK_M[m][self.sparse_idx[m], :]
        self.sparse_zeta = sparse_zeta.sum(axis=1)
        return self.sparse_zeta

    def _bound(self, mask=None, drop_diag=False):
        """Evidence Lower Bound (ELBO)"""
        if mask is None and not drop_diag:
            B = self.sumE_MK.prod(axis=0).sum()

        elif mask is None and drop_diag:
            assert self.mode_dims[0] == self.mode_dims[1]
            mask = np.abs(np.identity(self.mode_dims[0]) - 1)
            tmp = np.zeros(self.n_components)
            for k in xrange(self.n_components):
                tmp[k] = (mask * np.outer(self.E_DK_M[0][:, k], self.E_DK_M[1][:, k])).sum()
            B = (self.sumE_MK[2:].prod(axis=0) * tmp).sum()
        else:
            tmp = mask.copy()
            tmp = fast_dot(tmp, self.E_DK_M[-1])
            for i in xrange(self.n_modes - 2, 0, -1):
                tmp *= self.E_DK_M[i]
                tmp = tmp.sum(axis=-2)
            assert tmp.shape == self.E_DK_M[0].shape
            B = (tmp * self.E_DK_M[0]).sum()

        B -= np.log(self.sparse_data + 1).sum()
        B += (self.sparse_data * np.log(self._sparse_zeta())).sum()

        K = self.n_components
        for m in xrange(self.n_modes):
            D = self.mode_dims[m]

            B += (self.a - 1.) * (np.log(self.G_DK_M[m]).sum())
            B -= (self.a * self.beta_M[m])*(self.sumE_MK[m, :].sum())
            B -= K*D*(sp.gammaln(self.a) - self.a*np.log(self.a * self.beta_M[m]))

            gamma_DK = self.gamma_DK_M[m]
            delta_DK = self.delta_DK_M[m]

            B += (-(gamma_DK - 1.)*sp.psi(gamma_DK) - np.log(delta_DK)
                  + gamma_DK + sp.gammaln(gamma_DK)).sum()
        return B

    def _init_sparse(self, data):
        self.sparse_idx = data.nonzero()
        self.sparse_data = data[self.sparse_idx]
        self.n_points = self.sparse_data.size

    def _init_all_components(self, mode_dims):
        assert len(mode_dims) == self.n_modes
        self.mode_dims = mode_dims
        for m, D in enumerate(mode_dims):
            self._init_component(m, D)
        self.sparse_zeta = self._sparse_zeta()

    def _init_component(self, m, dim):
        assert self.mode_dims[m] == dim
        K = self.n_components
        s = self.smoothness
        gamma_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        delta_DK = s * rn.gamma(s, 1. / s, size=(dim, K))
        self.gamma_DK_M[m] = gamma_DK
        self.delta_DK_M[m] = delta_DK
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK) - np.log(delta_DK))
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def set_component(self, m, E_DK, G_DK, gamma_DK, delta_DK):
        assert E_DK.shape[1] == self.n_components
        self.E_DK_M[m] = E_DK.copy()
        self.sumE_MK[m, :] = E_DK.sum(axis=0)
        self.G_DK_M[m] = G_DK.copy()
        self.gamma_DK_M[m] = gamma_DK.copy()
        self.delta_DK_M[m] = delta_DK.copy()
        self.beta_M[m] = 1. / E_DK.mean()

    def set_component_like(self, m, model, idx=None):
        # assert model.zeta is not None or model.sparse_zeta is not None
        assert model.n_modes == self.n_modes
        assert model.n_components == self.n_components
        D = model.E_DK_M[m].shape[0]
        if idx is None:
            idx = np.arange(D)
        assert min(idx) >= 0 and max(idx) < D
        E_DK = model.E_DK_M[m][idx, :].copy()
        G_DK = model.G_DK_M[m][idx, :].copy()
        gamma_DK = model.gamma_DK_M[m][idx, :].copy()
        delta_DK = model.delta_DK_M[m][idx, :].copy()
        self.set_component(m, E_DK, G_DK, gamma_DK, delta_DK)

    def _check_component(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.gamma_DK_M[m]).all()
        assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_gamma(self, m, data):
        self.gamma_DK_M[m][:, :] = self.a
        tmp = data[self.sparse_idx] / self._sparse_zeta()
        for k in xrange(self.n_components):
            bar = tmp.copy()
            for mode in xrange(self.n_modes):
                idx = self.sparse_idx[mode]
                bar *= self.G_DK_M[mode][idx, k]
            self.gamma_DK_M[m][:, k] += np.bincount(self.sparse_idx[m],
                                                    weights=bar,
                                                    minlength=self.mode_dims[m])

    def _update_delta(self, m, mask=None, drop_diag=False):
        self.delta_DK_M[m][:, :] = self.a * self.beta_M[m]
        if mask is None and not drop_diag:
            self.sumE_MK[m, :] = 1.
            self.delta_DK_M[m][:, :] += self.sumE_MK.prod(axis=0)
            assert np.isfinite(self.delta_DK_M[m]).all()

        elif mask is None and drop_diag:
            assert self.mode_dims[0] == self.mode_dims[1]
            mask = np.abs(np.identity(self.mode_dims[0]) - 1)
            if m > 1:
                tmp = np.zeros(self.n_components)
                for k in xrange(self.n_components):
                    tmp[k] = (mask * np.outer(self.E_DK_M[0][:, k], self.E_DK_M[1][:, k])).sum()
                assert tmp.shape == (self.n_components,)
                self.sumE_MK[m, :] = 1.
            else:
                tmp = np.dot(mask, self.E_DK_M[np.abs(m-1)])
                assert tmp.shape == self.E_DK_M[m].shape
            self.delta_DK_M[m][:, :] += self.sumE_MK[2:].prod(axis=0) * tmp
            assert np.isfinite(self.delta_DK_M[m]).all()

        else:
            if drop_diag:
                diag_idx = np.identity(self.mode_dims[0]).astype(bool)
                assert (mask[diag_idx] == 0).all()
            tmp = mask.copy()
            tmp, order = make_first_mode(tmp, m)
            tmp = fast_dot(tmp, self.E_DK_M[order[-1]])
            for i in xrange(self.n_modes - 2, 0, -1):
                tmp *= self.E_DK_M[order[i]]
                tmp = tmp.sum(axis=-2)
            self.delta_DK_M[m][:, :] += tmp
            assert np.isfinite(self.delta_DK_M[m]).all()

    def _update_cache(self, m):
        gamma_DK = self.gamma_DK_M[m]
        delta_DK = self.delta_DK_M[m]
        self.E_DK_M[m] = gamma_DK / delta_DK
        self.sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.G_DK_M[m] = np.exp(sp.psi(gamma_DK)) / delta_DK

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data, modes=None, mask=None, drop_diag=False):
        if modes is not None:
            modes = list(set(modes))
            assert all(m in range(self.n_modes) for m in modes)
        else:
            modes = range(self.n_modes)

        curr_bound = -np.inf
        for itn in xrange(self.max_iter):
            s = time.time()
            for m in modes:
                self._update_gamma(m, data)
                self._update_delta(m, mask, drop_diag)
                self._update_cache(m)
                self._update_beta(m)  # must come after cache update!
                self._check_component(m)
            bound = self._bound(mask=mask, drop_diag=drop_diag)
            delta = (bound - curr_bound) / abs(curr_bound) if itn > 0 else np.nan
            e = time.time() - s
            if self.verbose:
                print 'ITERATION %d:\t\
                       Time: %f\t\
                       Objective: %.2f\t\
                       Change: %.5f\t'\
                       % (itn, e, bound, delta)
            assert ((delta >= 0.0) or (itn == 0))
            curr_bound = bound
            if delta < self.tol:
                break

    def fit(self, data, mask=None, drop_diag=False):
        if mask is not None:
            assert data.shape == mask.shape
            assert (mask >= 0).all() and (mask <= 1).all()
            assert np.issubdtype(mask.dtype, int)
            if drop_diag:
                assert data.shape[0] == data.shape[1]
                diag_idx = np.identity(data.shape[0]).astype(bool)
                mask[diag_idx] = 0
            data = data * mask
        self._init_sparse(data)
        self._init_all_components(data.shape)
        self._update(data, mask=mask, drop_diag=drop_diag)
        return self

    def transform(self, data, modes=(0,), mask=None, drop_diag=False, geom=True):
        assert len(data.shape) == self.n_modes
        assert all(m in range(self.n_modes) for m in modes)
        if mask is not None:
            assert data.shape == mask.shape
            assert (mask >= 0).all() and (mask <= 1).all()
            assert np.issubdtype(mask.dtype, int)
            if drop_diag:
                assert data.shape[0] == data.shape[1]
                diag_idx = np.identity(data.shape[0]).astype(bool)
                mask[diag_idx] = 0

        self.mode_dims = data.shape
        for m, D in enumerate(self.mode_dims):
            if m not in modes:
                if self.E_DK_M[m].shape[0] != D:
                    raise ValueError('There are no pre-trained components.')
                assert self.G_DK_M[m].shape[0] == D
            else:
                self._init_component(m, D)
        self._init_sparse(data)
        self._update(data, modes=modes, mask=mask, drop_diag=drop_diag)
        if geom:
            return tuple([self.G_DK_M[m] for m in modes])
        else:
            return tuple([self.E_DK_M[m] for m in modes])

    def fit_transform(self, data, modes=(0,), mask=None, drop_diag=False, geom=True):
        assert data.ndim == self.n_modes
        assert all(m in range(self.n_modes) for m in modes)

        self.fit(data, mask=mask, drop_diag=drop_diag)
        if geom:
            return tuple([self.G_DK_M[m] for m in modes])
        else:
            return tuple([self.E_DK_M[m] for m in modes])

    def reconstruct(self, weights={}, drop_diag=False, geom=True):
        """Reconstruct data using point estimates of latent factors.

        Currently supported only up to 5-way tensors.
        """
        if geom:
            tmp = [G_DK.copy() for G_DK in self.G_DK_M]
        else:
            tmp = [E_DK.copy() for E_DK in self.E_DK_M]
        if weights.keys():
            assert all(m in range(self.n_modes) for m in weights.keys())
            for m, weight_matrix in weights.iteritems():
                tmp[m] = weight_matrix
        Y_pred = parafac(tmp)
        if drop_diag:
            diag_idx = np.identity(Y_pred.shape[0]).astype(bool)
            Y_pred[diag_idx] = 0
        return Y_pred


def main():
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-m', '--mask', type=path, default=None)
    p.add_argument('-k', '--n_components', type=int, required=True)
    p.add_argument('-n', '--max_iter', type=int, default=200)
    p.add_argument('-t', '--tol', type=float, default=1e-4)
    p.add_argument('-s', '--smoothness', type=int, default=100)
    p.add_argument('-v', '--verbose', action="store_true", default=False)
    p.add_argument('-a', '--alpha', type=float, default=0.1)
    p.add_argument('--keep_diag', action="store_true", default=False)
    args = p.parse_args()

    args.out.makedirs_p()

    assert args.data.exists() and args.out.exists()
    data = np.load(args.data)

    if 'Y' in data.files:
        Y = data['Y']
    else:
        Y = data['Y_train']

    mask = None
    if 'mask' in data.files:
        mask = data['mask']
        Y = Y * mask

    elif args.mask is not None:
        mask = np.load(args.mask)['mask']
        assert mask.shape == Y.shape
        assert mask.dtype == bool
        mask = mask.astype(int)
        Y = Y * mask

    drop_diag = False
    if not args.keep_diag and Y.shape[0] == Y.shape[1]:
        diag_idx = np.identity(Y.shape[0]).astype(bool)
        drop_diag = (Y[diag_idx] == 0).all()
        if drop_diag:
            print 'Treating diagonal of first two modes as missing data.'

    bptf = BPTF(n_modes=Y.ndim,
                n_components=args.n_components,
                max_iter=args.max_iter,
                tol=args.tol,
                smoothness=args.smoothness,
                verbose=args.verbose,
                alpha=args.alpha)

    bptf.fit(Y, mask=mask, drop_diag=drop_diag)
    serialize_bptf(bptf, args.out, num=None, desc='trained_model')


if __name__ == '__main__':
    main()
