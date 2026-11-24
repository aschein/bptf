"""
Bayesian Poisson tensor factorization with variational inference.
"""
import time
import warnings
import numpy as np
import numpy.random as rn
import scipy.special as sp
import scipy.stats as st

import sparse
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted

from pathlib import Path
from tqdm import tqdm

from bptf.tensor_utils import is_binary, preprocess, unfolding_dot_khatri_rao, cp_to_tensor_at, log_cp_to_tensor_at


def _gamma_bound_term(pa, pb, qa, qb, compute_constant=False):
    out = sp.gammaln(qa) - pa * np.log(qb) + \
          (pa - qa) * sp.psi(qa) + qa * (1 - pb / qb)
    
    if compute_constant:
        # These terms do not depend on variational parameters.
        out += pa * np.log(pb) - sp.gammaln(pa)

    return out


class BPTF(BaseEstimator, TransformerMixin):
    def __init__(self, data_shape, n_components, alpha=0.1):
        # fixed hyperparameters
        self.data_shape = data_shape
        self.n_modes = len(data_shape)
        self.n_components = K = n_components
        self.alpha = alpha

        # updated hyperpameter
        self.beta_M = np.ones(self.n_modes, dtype=float)

        # variational parameters
        self.shp_DK_M = np.array([np.ones((D, K)) for D in self.data_shape], dtype='object')
        self.rte_DK_M = np.array([np.ones((D, K)) for D in self.data_shape], dtype='object')

        # arithmetic and geometric expectation of factor matrices
        self.E_DK_M = np.array([np.ones((D, K)) for D in self.data_shape], dtype='object')  
        self.G_DK_M = np.array([np.ones((D, K)) for D in self.data_shape], dtype='object')

        # Inference cache
        self._sumE_MK = np.array([E_DK.sum(axis=0) for E_DK in self.E_DK_M])

    def reconstruct(self, mask=None, fill_value=0, drop_diag=False, style='arithmetic', n_samples=1000):
        """Reconstruct data using expectation of latent factors."""
        assert style in ['arithmetic', 'geometric', 'post_pred']

        if style in ['arithmetic', 'geometric']:
            factors = self.G_DK_M if style == 'geometric' else self.E_DK_M
            Y_recon = tl.cp_to_tensor(cp_tensor=(None, factors))
        else:
            Y_recon = np.zeros(self.data_shape)
            for _ in range(n_samples):
                factors = [rn.gamma(self.shp_DK_M[m], 1./self.rte_DK_M[m]) for m in range(self.n_modes)]
                Y_recon += tl.cp_to_tensor(cp_tensor=(None, factors))
            Y_recon /= float(n_samples)

        if mask is not None:
            assert Y_recon.shape == mask.shape
            Y_recon[mask] = fill_value

        if drop_diag:
            # This is specific to tensors corresponding to networks where the
            # first two modes are actor x actor. In this case self-actions/edges
            # may be undefined, in which case we do not reconstruct the diagonal.
            assert Y_recon.shape[0] == Y_recon.shape[1]
            Y_recon[np.identity(Y_recon.shape[0]).astype(bool)] = fill_value

        return Y_recon

    def log_posterior_predictive_prob_at(self, values, coords=None, n_samples=1000):
        if coords is None:
            assert values.shape == self.data_shape
            values = values.ravel()
            
        log_likelihood_SI = np.empty((n_samples,) + values.size)
        for s in tqdm(range(n_samples)):
            factors = [rn.gamma(self.shp_DK_M[m], 1./self.rte_DK_M[m]) for m in range(self.n_modes)]

            if coords is None:
                recon = tl.cp_to_tensor(cp_tensor=(None, factors)).ravel()
            else:
                recon = cp_to_tensor_at(cp_tensor=(None, factors), coords=coords)

            log_likelihood_SI[s, :] = st.poisson.logpmf(values, recon)
            
        log_post_pred_I = sp.logsumexp(log_likelihood_SI, axis=1) - np.log(n_samples)
        return log_post_pred_I if coords is None else log_post_pred_I.reshape(self.data_shape)

    def _elbo(self, data, mask=None, missing_val=1):
        """Computes the Evidence Lower Bound (ELBO)."""
        if mask is None:
            uttkrp_K = self._sumE_MK.prod(axis=0)
        else:
            if missing_val == 1:
                uttkrp_DK = self._sumE_MK.prod(axis=0) / self._sumE_MK[0, :]
                uttkrp_DK = uttkrp_DK - unfolding_dot_khatri_rao(tensor=mask, 
                                                                 cp_tensor=(None, self.E_DK_M),
                                                                 mode=0)
            else:
                uttkrp_DK = unfolding_dot_khatri_rao(tensor=mask, 
                                                     cp_tensor=(None, self.E_DK_M),
                                                     mode=0)
            uttkrp_K = (self.E_DK_M[0] * uttkrp_DK).sum(axis=0)
        bound = -uttkrp_K.sum()

        if isinstance(data, np.ndarray):
            coords = data.nonzero(); values = data[coords]
        elif isinstance(data, sparse._coo.core.COO):
            coords, values = data.coords, data.data

        bound += values @ log_cp_to_tensor_at(cp_tensor=(None, self.G_DK_M), coords=coords)

        K = self.n_components
        for m in range(self.n_modes):
            bound += _gamma_bound_term(pa=self.alpha,
                                       pb=self.alpha * self.beta_M[m],
                                       qa=self.shp_DK_M[m],
                                       qb=self.rte_DK_M[m]).sum()
            bound += K * self.data_shape[m] * self.alpha * np.log(self.beta_M[m])
        return bound

    def _init(self, modes=None, **kwargs):
        modes = range(self.n_modes) if modes is None else list(set(modes))
        assert all(m in range(self.n_modes) for m in modes)

        shp = kwargs.get('init_shp', 100.)
        rte = kwargs.get('init_rte', 1.)

        K = self.n_components
        for m in modes:
            self._init_mode(m, **kwargs)
    
    def _init_mode(self, m, **kwargs):
        shp = kwargs.get("init_shp", 100.)
        rte = kwargs.get("init_rte", 1.)
        D = self.data_shape[m]
        K = self.n_components
        self.shp_DK_M[m] = shp_DK = rn.gamma(shp, 1. / rte, size=(D, K))
        self.rte_DK_M[m] = rte_DK = rn.gamma(shp, 1. / rte, size=(D, K))
        self.G_DK_M[m] = np.exp(sp.psi(shp_DK) - np.log(rte_DK))
        self.E_DK_M[m] = shp_DK / rte_DK
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()
        self._sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self._check_mode(m)

    def _check_mode(self, m):
        assert np.isfinite(self.E_DK_M[m]).all()
        assert np.isfinite(self.G_DK_M[m]).all()
        assert np.isfinite(self.shp_DK_M[m]).all()
        assert np.isfinite(self.rte_DK_M[m]).all()

    def _update_variational_params(self, m, data, mask=None, missing_val=1):
        # This implements equations 4-5 in the paper.

        # Update variational shape parameters (equation 4)
        if isinstance(data, np.ndarray):
            tmp = data / tl.cp_to_tensor(cp_tensor=(None, self.G_DK_M))

        elif isinstance(data, sparse._coo.core.COO):
            tmp = data.astype(float)
            tmp.data /= cp_to_tensor_at(cp_tensor=(None, self.G_DK_M), coords=data.coords)

        uttkrp_DK = unfolding_dot_khatri_rao(tensor=tmp, 
                                             cp_tensor=(None, self.G_DK_M),
                                             mode=m)
        
        self.shp_DK_M[m][:, :] = self.alpha + self.G_DK_M[m] * uttkrp_DK

        # Update variational rate parameters (equation 5)
        if mask is None:
            uttkrp_DK = self._sumE_MK.prod(axis=0) / self._sumE_MK[m, :]
        else:
            if missing_val == 1:
                uttkrp_DK = self._sumE_MK.prod(axis=0) / self._sumE_MK[m, :]
                uttkrp_DK = uttkrp_DK - unfolding_dot_khatri_rao(tensor=mask, 
                                                                 cp_tensor=(None, self.E_DK_M),
                                                                 mode=m)
            else:
                uttkrp_DK = unfolding_dot_khatri_rao(tensor=mask, 
                                                     cp_tensor=(None, self.E_DK_M),
                                                     mode=m)
    
        self.rte_DK_M[m][:, :] = self.alpha * self.beta_M[m] + uttkrp_DK

        self._update_cache(m)

    def _update_cache(self, m):
        shp_DK, rte_DK = self.shp_DK_M[m], self.rte_DK_M[m]
        self.G_DK_M[m] = np.exp(sp.psi(shp_DK) - np.log(rte_DK))
        self.E_DK_M[m] = shp_DK / rte_DK
        self._sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)

    def _update_beta(self, m):
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def _update(self, data, mask=None, modes=None, **kwargs):
        modes = range(self.n_modes) if modes is None else list(set(modes))
        assert all(m in range(self.n_modes) for m in modes)

        for m in range(self.n_modes):
            if m not in modes:
                self._clamp_component(m)

        curr_elbo = self._elbo(data, mask=mask)
        if not np.isfinite(curr_elbo):
            raise FloatingPointError('Initial ELBO must be finite.')
        
        if kwargs.get('verbose', True):
            print ('ITERATION %d:\t'\
                  'Time: %f\t'\
                  'Objective: %.2f\t'\
                  'Change: %.5e\t'\
                % (0, 0.0, curr_elbo, np.nan))

        for itn in tqdm(range(kwargs.get('max_iter', 200))):
            
            s = time.time()
            for m in modes:
                self._update_variational_params(m, data, mask)
                self._update_beta(m)
                self._check_mode(m)
            bound = self._elbo(data, mask=mask)
            if not np.isfinite(bound):
                raise FloatingPointError(f'ELBO must be finite at iteration {itn + 1}.')
            # Guard the diagnostic's denominator; convergence uses isclose below.
            delta = (bound - curr_elbo) / max(abs(curr_elbo), 1e-12)
            e = time.time() - s
            
            if kwargs.get('verbose', True):
                print ('ITERATION %d:\t'\
                      'Time: %f\t'\
                      'Objective: %.2f\t'\
                      'Change: %.5e\t'\
                      % (itn+1, e, bound, delta))
            
            # Numerical tolerance stays independent of the stopping tolerance.
            if bound < curr_elbo and not np.isclose(
                    bound, curr_elbo, rtol=1e-12, atol=1e-12):
                raise RuntimeError('ELBO decreased beyond numerical tolerance '
                                   f'at iteration {itn + 1}: {curr_elbo} -> {bound}.')
            converged = np.isclose(bound, curr_elbo,
                                   rtol=kwargs.get('tol', 1e-4), atol=1e-12)
            curr_elbo = bound
            if converged:
                break

    def _clamp_component(self, m):
        """Make a component a constant.
        This amounts to setting the expectations under the
        Q-distribution to be equal to the geometric expectations.
        """
        self.E_DK_M[m][:, :] = self.G_DK_M[m]
        self._sumE_MK[m, :] = self.E_DK_M[m].sum(axis=0)
        self.beta_M[m] = 1. / self.E_DK_M[m].mean()

    def fit(self, data, mask=None, missing_val=1, **kwargs):
        assert data.ndim == self.n_modes
        data = preprocess(data)
        
        if mask is not None:
            mask = preprocess(mask)
            assert data.shape == mask.shape
            assert is_binary(mask)

        self._init()

        self._update(data, mask=mask, missing_val=missing_val, **kwargs)
        self.is_fitted_ = True
        return self

    def transform(self, data, *, mode, mask=None, missing_val=1,
                  expectation='geometric', max_iter=200, tol=1e-4, verbose=False):
        """Infer one factor matrix without changing the fitted model.

        Parameters
        ----------
        data : numpy.ndarray or sparse.COO
            Finite, nonnegative integer counts. Only the selected axis may
            differ in size from training. Other axes must contain the same
            entities in the same order; labels are not checked automatically.
        mode : int
            Axis whose factors to infer. Returns an array of shape
            ``(data.shape[mode], n_components)`` in input order.
        mask : numpy.ndarray or sparse.COO, optional
            Binary array with the same shape as data. Entries equal to
            ``missing_val`` are unobserved and excluded from inference.
            Sparse data and masks must have zero fill values. Unstored data
            entries are observed zeros unless masked out.
        missing_val : {0, 1}, default=1
            Mask value indicating an unobserved entry.
        expectation : {'geometric', 'arithmetic'}, default='geometric'
            Statistic of the inferred Gamma distributions to return.
        max_iter : int, default=200
            Maximum number of local variational updates.
        tol : float, default=1e-4
            Stop each row independently when its maximum shape-parameter
            change, divided by max(1, previous shape), is at most tol.
        verbose : bool, default=False
            Display iteration progress.

        Notes
        -----
        The other modes' full variational distributions and all hyperparameters
        remain fixed. Initialization is deterministic and each converged row is
        frozen, so processing rows separately or in batches gives the same
        result up to floating-point error. No model arrays or random state are
        modified. A ConvergenceWarning is issued for unfinished rows.
        """
        check_is_fitted(self, 'is_fitted_')
        if (isinstance(mode, (bool, np.bool_)) or
                not isinstance(mode, (int, np.integer)) or
                not 0 <= mode < self.n_modes):
            raise ValueError('mode must be a single integer axis in [0, n_modes).')
        if expectation not in ('geometric', 'arithmetic'):
            raise ValueError("expectation must be 'geometric' or 'arithmetic'.")
        if (isinstance(max_iter, (bool, np.bool_)) or
                not isinstance(max_iter, (int, np.integer)) or max_iter < 1):
            raise ValueError('max_iter must be a positive integer.')
        if not np.isscalar(tol) or not np.isfinite(tol) or tol < 0:
            raise ValueError('tol must be finite and nonnegative.')
        if missing_val not in (0, 1):
            raise ValueError('missing_val must be 0 or 1.')

        def values_of(array, name):
            if not isinstance(array, (np.ndarray, sparse.COO)):
                raise TypeError(f'{name} must be a numpy.ndarray or sparse.COO.')
            if isinstance(array, sparse.COO) and array.fill_value != 0:
                raise ValueError(f'{name} must have a zero sparse fill value.')
            values = array.data if isinstance(array, sparse.COO) else array
            if (array.dtype.kind not in 'biuf' or
                    not np.isfinite(values).all() or (values < 0).any() or
                    (values != np.floor(values)).any()):
                raise ValueError(f'{name} must contain finite nonnegative integers.')
            return values

        values_of(data, 'data')
        if data.ndim != self.n_modes:
            raise ValueError('data must have the same number of modes as training.')
        if any(data.shape[m] != self.data_shape[m]
               for m in range(self.n_modes) if m != mode):
            raise ValueError('All untransformed axes must match the training shape.')
        if mask is not None:
            mask_values = values_of(mask, 'mask')
            if mask.shape != data.shape or (mask_values > 1).any():
                raise ValueError('mask must be binary and have the same shape as data.')

        D, K = data.shape[mode], self.n_components
        if D == 0:
            return np.empty((0, K), dtype=float)

        if isinstance(data, sparse.COO):
            coords, counts = data.coords, data.data
        else:
            coords = np.array(data.nonzero())
            counts = data[tuple(coords)]
        if mask is not None:
            masked = mask[tuple(coords)]
            if isinstance(masked, sparse.COO):
                masked = masked.todense()
            observed = masked != missing_val
            coords, counts = coords[:, observed], counts[observed]
        rows = coords[mode]
        counts = counts.astype(float, copy=False)

        # Only the fixed modes contribute to the Gamma rates. In particular,
        # never recompute beta from the new batch or collapse E[theta] to G[theta].
        means = [np.asarray(factor, dtype=float) for factor in self.E_DK_M]
        means[mode] = np.empty((D, K))  # Its values are unused by the contraction.
        exposure = np.ones(K)
        log_fixed = np.zeros((counts.size, K))
        for m in range(self.n_modes):
            if m != mode:
                exposure *= means[m].sum(axis=0)
                # Use the Gamma parameters to avoid taking log of underflowed G.
                log_fixed += (sp.digamma(self.shp_DK_M[m][coords[m]])
                              - np.log(self.rte_DK_M[m][coords[m]]))
        if mask is not None:
            marked_exposure = unfolding_dot_khatri_rao(
                mask.astype(float), (None, means), mode)
            exposure = (np.maximum(exposure - marked_exposure, 0.0)
                        if missing_val == 1 else marked_exposure)
        rate = np.broadcast_to(self.alpha * self.beta_M[mode] + exposure, (D, K))

        # Equal initial allocations depend only on each row's observed counts.
        totals = np.bincount(rows, weights=counts, minlength=D)
        shape = np.broadcast_to(self.alpha + totals[:, None] / K, (D, K)).copy()
        active = totals > 0
        for _ in tqdm(range(max_iter), disable=not verbose):
            if not active.any():
                break
            log_allocations = sp.digamma(shape[rows]) - np.log(rate[rows])
            log_allocations += log_fixed
            responsibilities = np.exp(
                log_allocations - sp.logsumexp(log_allocations, axis=1, keepdims=True))
            updated = np.full((D, K), self.alpha, dtype=float)
            for k in range(K):
                updated[:, k] += np.bincount(
                    rows, weights=counts * responsibilities[:, k], minlength=D)
            change = np.max(np.abs(updated - shape) / np.maximum(1.0, shape), axis=1)
            shape[active] = updated[active]
            active &= change > tol
        if active.any():
            warnings.warn(f'transform did not converge for {active.sum()} rows '
                          f'within {max_iter} iterations.', ConvergenceWarning,
                          stacklevel=2)
        if expectation == 'arithmetic':
            return shape / rate
        return np.exp(sp.digamma(shape) - np.log(rate))

    def fit_transform(self, modes, data, mask=None, expectation='geometric', **kwargs):
        assert all(m in range(self.n_modes) for m in modes)
        self.fit(data, mask=mask, **kwargs)
        return self.G_DK_M[modes] if expectation == 'geometric' else self.E_DK_M[modes]


def save_bptf(model, outdir=Path('.'), filename='bptf', filepath=None):
    if filepath is None:
        d = 1
        filepath = outdir.joinpath(f'{filename}_{d}.npz')
        while filepath.exists():
            d += 1
            filepath = outdir.joinpath(f'{filename}_{d}.npz')
    filepath.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(filepath,
                        E_DK_M=model.E_DK_M,
                        G_DK_M=model.G_DK_M,
                        shp_DK_M=model.shp_DK_M,
                        rte_DK_M=model.rte_DK_M,
                        beta_M=model.beta_M,
                        alpha=model.alpha)
    print(filepath)


def load_bptf(filepath):
    file = np.load(filepath, allow_pickle=True)
    E_DK_M = file['E_DK_M']

    data_shape = tuple(E_DK.shape[0] for E_DK in E_DK_M)
    n_components = E_DK_M[0].shape[1]
    alpha = float(file['alpha'])

    model = BPTF(data_shape=data_shape, n_components=n_components, alpha=alpha)
    for m in range(model.n_modes):
        model.E_DK_M[m][:, :] = file['E_DK_M'][m]
        model.G_DK_M[m][:, :] = file['G_DK_M'][m]
        model.shp_DK_M[m][:, :] = file['shp_DK_M'][m]
        model.rte_DK_M[m][:, :] = file['rte_DK_M'][m]
        model._update_cache(m)
        model._check_mode(m)
    model.beta_M[:]= file['beta_M']
    model.is_fitted_ = True
    
    return model
