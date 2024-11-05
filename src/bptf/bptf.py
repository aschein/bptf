"""
Bayesian Poisson tensor factorization with variational inference.
"""
import time
import numpy as np
import numpy.random as rn
import scipy.special as sp
import scipy.stats as st

import sparse
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from sklearn.base import BaseEstimator, TransformerMixin

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
            delta = (bound - curr_elbo) / abs(curr_elbo)
            e = time.time() - s
            
            if kwargs.get('verbose', True):
                print ('ITERATION %d:\t'\
                      'Time: %f\t'\
                      'Objective: %.2f\t'\
                      'Change: %.5e\t'\
                      % (itn+1, e, bound, delta))
            
            assert delta >= 0.0
            curr_elbo = bound
            if delta < kwargs.get('tol', 1e-4):
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
        return self

    def transform(self, modes, data, mask=None, expectation='geometric', **kwargs):
        """Transform new data given a pre-trained model."""
        assert all(m in range(self.n_modes) for m in modes)
        assert type(data) in [sparse._coo.core.COO, np.ndarray]
        assert data.ndim == self.n_modes
        data = preprocess(data)

        if mask is not None:
            mask = preprocess(mask)
            assert type(mask) in [sparse._coo.core.COO, np.ndarray]
            assert data.shape == mask.shape
            assert is_binary(mask)
            assert np.issubdtype(mask.dtype, int)
        
        assert self.data_shape == data.shape

        for m, D in enumerate(self.data_shape):
            if m not in modes:
                if self.E_DK_M[m].shape[0] != D:
                    raise ValueError('Pre-trained components dont match new data.')
            else:
                self._init_mode(m, D, **kwargs)
        self._update(data, mask=mask, modes=modes)

        return self.G_DK_M[modes] if expectation == 'geometric' else self.E_DK_M[modes]

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
    
    return model