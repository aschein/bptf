import numpy as np

from path import path
from argparse import ArgumentParser

from ptf import PoissonTF
from evaluate import evaluate
from utils_ptf import serialize_ptf

import pickle


def get_test_set_from_mode_idx(Y_test, obs_modes, obs_idx_M):
    X_test = Y_test.copy()
    for m, obs_idx in zip(obs_modes, obs_idx_M):
        X_test = np.take(X_test, indices=obs_idx, axis=m)
    return X_test


def get_eval_idx_from_mode_idx(Y_test, obs_modes, obs_idx_M):
    raise NotImplementedError


def predict_slices_with_ptf(trained_model, Y_test, obs_modes, obs_idx_M):
    X_test = get_test_set_from_mode_idx(Y_test, obs_modes, obs_idx_M)
    Y_pred, test_model = predict_with_ptf(trained_model, X_test, obs_modes, obs_idx_M)
    eval_idx = get_eval_idx_from_mode_idx(Y_test, obs_modes, obs_idx_M)
    return Y_pred, test_model, evaluate(Y_pred[eval_idx], Y_pred[eval_idx])


def predict_with_ptf(trained_model, X_test, obs_modes, obs_idx_M, drop_diag=True, geom=True):
    test_model = PoissonTF(**trained_model.get_params())
    for m, idx in zip(obs_modes, obs_idx_M):
        test_model.set_component_like(m, trained_model, idx)
    trans_modes = filter(lambda m: m not in obs_modes, range(X_test.ndim))
    weights_M = test_model.transform(X_test, modes=trans_modes)
    Y_pred = trained_model.reconstruct(weights=dict(zip(trans_modes, weights_M)), drop_diag=drop_diag, geom=geom)
    return Y_pred, test_model


def predict_top_N_with_ptf(trained_model, Y_test, N, drop_diag=True):
    X_test = Y_test[N:, N:]
    n_actors = Y_test.shape[0]
    obs_modes = [0, 1]
    obs_idx_M = [range(N, n_actors), range(N, n_actors)]
    Y_pred, test_model = predict_with_ptf(trained_model, X_test, obs_modes, obs_idx_M, drop_diag=drop_diag, geom=geom)
    return Y_pred, test_model, evaluate(Y_pred[:N, :N], Y_test[:N, :N])


def predict_bottom_N_with_ptf(trained_model, Y_test, N, drop_diag=True):
    X_test = Y_test[:N, :N]
    n_actors = Y_test.shape[0]
    obs_modes = [0, 1]
    obs_idx_M = [range(N), range(N)]
    Y_pred, test_model = predict_with_ptf(trained_model, X_test, obs_modes, obs_idx_M, drop_diag=drop_diag, geom=geom)
    return Y_pred, test_model, evaluate(Y_pred[N:n_actors, N:n_actors], Y_test[N:n_actors, N:n_actors])

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('-d', '--data', type=path, required=True)
    p.add_argument('-o', '--out', type=path, required=True)
    p.add_argument('-m', '--model', type=path, required=True)
    args = p.parse_args()

    assert args.data.exists() and args.out.exists() and args.model.exists()

    model = np.load(args.model)
    E_DK_M = model['E_DK_M']
    L_DK_M = model['L_DK_M']
    alpha_DK_M = model['alpha_DK_M']
    beta_DK_M = model['beta_DK_M']
    n_modes = len(E_DK_M)
    n_components = E_DK_M[0].shape[1]

    param_file = path(args.model.abspath().replace('npz', 'dat'))
    if param_file.exists():
        with open(param_file, 'rb') as f:
            params = pickle.load(f)
        train_model = PoissonTF(**params)
    else:
        train_model = PoissonTF(n_modes=n_modes, n_components=n_components)
        params = train_model.get_params()
        with open(param_file, 'wb') as f:
            pickle.dump(params, f)

    train_model = PoissonTF(**params)
    for m in xrange(n_modes):
        train_model.set_component(m, E_DK_M[m], L_DK_M[m], alpha_DK_M[m], beta_DK_M[m])

    data = np.load(args.data)
    Y_train = data['Y_train']
    Y_test = data['Y_test']

    for N in [25, 50, 100]:
        Y_pred, test_model, evals = predict_top_N_with_ptf(train_model, Y_test, N, drop_diag=True, geom=True)
        out_dir = args.out.joinpath('top_%d' % N)
        out_dir.makedirs_p()

        num = int(args.model.namebase.split('_')[0])
        serialize_ptf(test_model, out_dir, num=num, desc='test_model')
        np.savez(out_dir.joinpath('%d_%s.npz' % (num, 'pred')), Y_pred=Y_pred)

        with open(out_dir.joinpath('%d_eval.txt' % num), 'w+') as f:
            f.write('BAYESIAN PTF\n------------\n')
            for func, val in evals.iteritems():
                f.write('%f\t%s\n' % (val, func))

        Y_pred, test_model, evals = predict_bottom_N_with_ptf(train_model, Y_test, N, drop_diag=True, geom=True)
        out_dir = args.out.joinpath('c_top_%d' % N)
        out_dir.makedirs_p()

        num = int(args.model.namebase.split('_')[0])
        serialize_ptf(test_model, out_dir, num=num, desc='test_model')
        np.savez(out_dir.joinpath('%d_%s.npz' % (num, 'pred')), Y_pred=Y_pred)

        with open(out_dir.joinpath('%d_eval.txt' % num), 'w+') as f:
            f.write('BAYESIAN PTF\n------------\n')
            for func, val in evals.iteritems():
                f.write('%f\t%s\n' % (val, func))
