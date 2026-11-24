import pickle

import numpy as np
import pytest
from scipy.special import digamma
from sklearn.exceptions import ConvergenceWarning, NotFittedError
import sparse

from bptf import BPTF, load_bptf, save_bptf


def fit_model(shape=(9, 5), components=2):
    data = np.random.default_rng(12).poisson(2.0, size=shape)
    random_state = np.random.get_state()
    try:
        np.random.seed(42)
        model = BPTF(shape, components, alpha=0.5)
        model.fit(data, max_iter=1 if components == 1 else 15, verbose=False)
    finally:
        np.random.set_state(random_state)
    return model


@pytest.fixture
def model():
    return fit_model()


@pytest.mark.parametrize('n_cells', [0, 1, 3, 5, 17])
@pytest.mark.parametrize('as_sparse', [False, True])
def test_new_cell_count_and_no_mutation(model, n_cells, as_sparse):
    # Five new cells makes the transformed matrix square, unlike training.
    data = np.random.default_rng(2).poisson(1.5, size=(n_cells, 5))
    if as_sparse:
        data = sparse.COO.from_numpy(data)
    model_before = pickle.dumps(model)
    data_before = pickle.dumps(data)
    random_before = pickle.dumps(np.random.get_state())
    reconstruction = model.reconstruct().copy()

    result = model.transform(data, mode=0, max_iter=1000)

    assert result.shape == (n_cells, 2)
    assert np.isfinite(result).all()
    assert (result >= 0).all()
    assert pickle.dumps(model) == model_before
    assert pickle.dumps(data) == data_before
    assert pickle.dumps(np.random.get_state()) == random_before
    np.testing.assert_array_equal(model.reconstruct(), reconstruction)
    result[:] = -1  # Returned factors must not alias training arrays.
    assert pickle.dumps(model) == model_before


@pytest.mark.parametrize('mode', [0, 1, 2])
@pytest.mark.parametrize('as_sparse', [False, True])
@pytest.mark.parametrize('expectation', ['arithmetic', 'geometric'])
@pytest.mark.parametrize('masked', [False, True])
def test_batch_and_permutation_consistency(mode, as_sparse, expectation, masked):
    model = fit_model((4, 5, 6))
    shape = list(model.data_shape)
    shape[mode] = 7
    rng = np.random.default_rng(21)
    data = rng.poisson(0.8, size=shape)
    mask = (rng.random(shape) < 0.4).astype(int) if masked else None
    before = pickle.dumps(model)

    def project(indices):
        batch = np.take(data, indices, axis=mode)
        batch_mask = None if mask is None else np.take(mask, indices, axis=mode)
        if as_sparse:
            batch = sparse.COO.from_numpy(batch)
            if batch_mask is not None:
                batch_mask = sparse.COO.from_numpy(batch_mask)
        return model.transform(batch, mode=mode, mask=batch_mask,
                               expectation=expectation, max_iter=1000, tol=1e-6)

    together = project(np.arange(7))
    batches = np.concatenate([project(np.arange(0, 1)), project(np.arange(1, 4)),
                              project(np.arange(4, 7))])
    separately = np.concatenate([project([i]) for i in range(7)])
    permutation = np.array([6, 0, 4, 1, 5, 3, 2])
    np.testing.assert_allclose(batches, together, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(separately, together, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(project(permutation), together[permutation], rtol=1e-10, atol=1e-12)
    assert pickle.dumps(model) == before


@pytest.mark.parametrize('mode', [0, 1])
@pytest.mark.parametrize('sparse_data', [False, True])
@pytest.mark.parametrize('sparse_mask', [False, True])
@pytest.mark.parametrize('missing_val', [0, 1])
@pytest.mark.parametrize('expectation', ['arithmetic', 'geometric'])
def test_rank_one_matches_exact_gamma_posterior(mode, sparse_data, sparse_mask,
                                              missing_val, expectation):
    model = fit_model(components=1)
    rng = np.random.default_rng(8)
    shape = list(model.data_shape)
    shape[mode] = 4
    data = rng.poisson(1.2, size=shape)
    observed = rng.random(shape) > 0.5
    selector = [slice(None)] * 2
    selector[mode] = 0
    observed[tuple(selector)] = False  # An entirely missing cell retains its prior.
    selector[mode] = 1
    data[tuple(selector)] = 0  # Observed zeros still contribute exposure.
    other = 1 - mode
    mean = np.asarray(model.E_DK_M[other][:, 0], dtype=float)
    mean_shape = [1, 1]
    mean_shape[other] = mean.size
    a = model.alpha + (data * observed).sum(axis=other)
    b = model.alpha * model.beta_M[mode] + (observed * mean.reshape(mean_shape)).sum(axis=other)
    expected = a / b if expectation == 'arithmetic' else np.exp(digamma(a) - np.log(b))
    mask = (~observed if missing_val == 1 else observed).astype(int)
    masked_counts = data.copy()
    masked_counts[~observed] = 1000  # Ignored counts cannot affect the answer.
    if sparse_data:
        masked_counts = sparse.COO.from_numpy(masked_counts)
    if sparse_mask:
        mask = sparse.COO.from_numpy(mask)
    before = pickle.dumps((model, masked_counts, mask))

    result = model.transform(masked_counts, mode=mode, mask=mask,
                             missing_val=missing_val, expectation=expectation)

    np.testing.assert_allclose(result[:, 0], expected, rtol=1e-12, atol=1e-12)
    assert pickle.dumps((model, masked_counts, mask)) == before


def test_multicomponent_projection_matches_scalar_coordinate_ascent(model):
    data = np.array([[0, 3, 1, 8, 2], [2, 0, 0, 1, 9]])
    results = []
    for cell in data:
        a = np.full(2, model.alpha + cell.sum() / 2)
        b = model.alpha * model.beta_M[0] + model.E_DK_M[1].sum(axis=0)
        for _ in range(1000):
            next_a = np.full(2, model.alpha)
            for gene, count in enumerate(cell):
                weights = np.exp(digamma(a) - np.log(b)) * model.G_DK_M[1][gene]
                next_a += count * weights / weights.sum()
            change = np.max(np.abs(next_a - a) / np.maximum(1, a))
            a = next_a
            if change <= 1e-8:
                break
        results.append(a / b)
    actual = model.transform(data, mode=0, expectation='arithmetic', max_iter=1000, tol=1e-8)
    np.testing.assert_allclose(actual, results, rtol=1e-10, atol=1e-12)


def test_dense_and_sparse_agree_with_observed_zeros(model):
    data = np.random.default_rng(13).poisson(0.2, size=(11, 5))
    dense = model.transform(data, mode=0, max_iter=1000)
    coo = model.transform(sparse.COO.from_numpy(data), mode=0, max_iter=1000)
    np.testing.assert_allclose(dense, coo, rtol=1e-12, atol=1e-12)


def test_saved_model_can_transform(model, tmp_path):
    path = tmp_path / 'bptf.npz'
    save_bptf(model, filepath=path)
    restored = load_bptf(path)
    data = np.ones((3, 5), dtype=int)
    np.testing.assert_allclose(restored.transform(data, mode=0, max_iter=1000),
                               model.transform(data, mode=0, max_iter=1000))


def test_unfitted_model_is_rejected():
    with pytest.raises(NotFittedError):
        BPTF((3, 5), 2).transform(np.ones((7, 5)), mode=0)


@pytest.mark.parametrize('mode', [[0], (0,), -1, 2, 0.0, True])
def test_invalid_mode(model, mode):
    with pytest.raises(ValueError, match='mode must be a single integer'):
        model.transform(np.ones((3, 5)), mode=mode)


@pytest.mark.parametrize('data', [np.ones((3, 6)), np.ones((3, 5, 1))])
def test_fixed_axes_must_match(model, data):
    with pytest.raises(ValueError, match='training'):
        model.transform(data, mode=0)


@pytest.mark.parametrize('bad_value', [-1, 0.25, np.nan, np.inf])
def test_invalid_counts(model, bad_value):
    data = np.ones((3, 5))
    data[0, 0] = bad_value
    with pytest.raises(ValueError, match='finite nonnegative integers'):
        model.transform(data, mode=0)


def test_invalid_options(model):
    data = np.ones((3, 5))
    for options in ({'expectation': 'median'}, {'max_iter': 0}, {'max_iter': 1.5},
                    {'tol': -1}, {'tol': np.nan}, {'missing_val': 2},
                    {'mask': np.ones((3, 6))}, {'mask': np.full((3, 5), 2)}):
        with pytest.raises(ValueError):
            model.transform(data, mode=0, **options)


def test_sparse_nonzero_fill_is_rejected(model):
    data = sparse.COO(np.array([[0], [0]]), [2], shape=(3, 5), fill_value=1)
    with pytest.raises(ValueError, match='zero sparse fill'):
        model.transform(data, mode=0)


def test_nonconvergence_warns_without_modifying_model(model):
    before = pickle.dumps(model)
    with pytest.warns(ConvergenceWarning, match='did not converge'):
        model.transform(np.ones((3, 5)), mode=0, max_iter=1, tol=0)
    assert pickle.dumps(model) == before
