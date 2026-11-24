from unittest.mock import Mock

import numpy as np
import pytest

from bptf import BPTF


@pytest.fixture
def scripted_fit(monkeypatch):
    """Exercise the training loop with controlled ELBO values and real updates."""
    random_state = np.random.get_state()
    np.random.seed(42)

    def run(elbos, tol=1e-4):
        data = np.ones((3, 5), dtype=int)
        model = BPTF(data.shape, 1)
        objective = Mock(side_effect=[np.float64(value) for value in elbos])
        monkeypatch.setattr(model, '_elbo', objective)
        # Extra iterations ensure the loop must actually stop at convergence.
        with np.errstate(divide='raise', invalid='raise'):
            model.fit(data, max_iter=len(elbos) + 1, tol=tol, verbose=False)
        assert objective.call_count == len(elbos)
        assert model.is_fitted_
        return model

    try:
        yield run
    finally:
        np.random.set_state(random_state)


@pytest.mark.parametrize('tol', [1e-4, 1e-8])
@pytest.mark.parametrize('old, new', [
    (1256.6193620515396, 1256.6193620515394),  # Original audit failure.
    (-1256.6193620515396, np.nextafter(-1256.6193620515396, -np.inf)),
    (1e12, 1e12 - 0.01),  # Roundoff tolerance must account for objective scale.
    (0.0, -5e-13),
])
def test_roundoff_decrease_converges(scripted_fit, old, new, tol):
    scripted_fit([old, new], tol=tol)


@pytest.mark.parametrize('tol', [1e-4, 0.5])
@pytest.mark.parametrize('old, new', [
    (100.0, 99.999),  # Within convergence tolerance, beyond roundoff tolerance.
    (-100.0, -100.001),
    (0.0, -1e-6),
])
def test_real_decrease_raises_even_with_loose_convergence_tolerance(
        scripted_fit, old, new, tol):
    with pytest.raises(RuntimeError, match='ELBO decreased beyond numerical tolerance'):
        scripted_fit([old, new], tol=tol)


@pytest.mark.parametrize('elbos', [
    [100.0, 100.001],
    [-100.0, -99.999],
    [0.0, 5e-13],
    [0.0, 0.0],
])
def test_small_improvement_or_plateau_converges(scripted_fit, elbos):
    scripted_fit(elbos)


@pytest.mark.parametrize('elbos', [
    [-100.0, -90.0, -89.999],
    [0.0, 1.0, 1.000001],
])
def test_meaningful_improvement_continues_training(scripted_fit, elbos):
    scripted_fit(elbos)


@pytest.mark.parametrize('bad_elbo', [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize('initial', [False, True])
def test_nonfinite_elbo_raises(scripted_fit, bad_elbo, initial):
    elbos = [bad_elbo] if initial else [100.0, bad_elbo]
    with pytest.raises(FloatingPointError, match='ELBO.*finite'):
        scripted_fit(elbos)


@pytest.mark.parametrize('tol', [1e-4, 1e-8])
def test_original_rank_one_fit_finishes_and_recovers_rates(tol):
    rng = np.random.default_rng(42)
    shape = (15, 18, 20)
    factors = [rng.gamma(1.5, 1.0, size=(d, 1)) for d in shape]
    true_rate = np.einsum('ik,jk,lk->ijl', *factors)
    data = rng.poisson(true_rate)
    model = BPTF(shape, 1, alpha=0.1)
    random_state = np.random.get_state()
    try:
        np.random.seed(42)
        with np.errstate(divide='raise', invalid='raise'):
            result = model.fit(data, max_iter=200, tol=tol, verbose=False)
    finally:
        np.random.set_state(random_state)

    assert result is model
    assert model.is_fitted_
    estimate = model.reconstruct()
    assert np.isfinite(estimate).all()
    assert (estimate >= 0).all()
    relative_error = np.linalg.norm(estimate - true_rate) / np.linalg.norm(true_rate)
    assert relative_error < 0.15
    assert np.corrcoef(estimate.ravel(), true_rate.ravel())[0, 1] > 0.95
