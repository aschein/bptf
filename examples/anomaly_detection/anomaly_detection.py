import numpy.random as rn
import scipy.stats as st
from sklearn.metrics import precision_score, recall_score, f1_score

import tensorly as tl
from tensorly.cp_tensor import CPTensor

from bptf import BPTF

def generate(shp=(30, 30, 20, 10), K=5, alpha=0.1, beta=0.1):
    """Generate a count tensor from the BPTF model.

    PARAMS:
    shp -- (tuple) shape of the generated count tensor
    K -- (int) number of latent components
    alpha -- (float) shape parameter of gamma prior over factors
    beta -- (float) rate parameter of gamma prior over factors

    RETURNS:
    Mu -- (np.ndarray) true Poisson rates
    Y -- (np.ndarray) generated count tensor
    """
    Theta_DK_M = [rn.gamma(alpha, 1./beta, size=(D, K)) for D in shp]
    Mu = tl.cp_to_tensor(CPTensor((None, Theta_DK_M)))
    assert Mu.shape == shp
    Y = rn.poisson(Mu)
    return Mu, Y


def corrupt(Y, p=0.05):
    """Corrupt a count tensor with anomalies.

    The corruption noise model is:

        corrupt(y) = y * g, where g ~ Gamma(10, 2)

    PARAMS:
    p -- (float) proportion of tensor entries to corrupt

    RETURNS:
    out -- (np.ndarray) corrupted count tensor
    mask -- (np.ndarray) boolean array, same shape as count tensor
                         True means that entry was corrupted.
    """
    out = Y.copy()
    mask = (rn.random(size=out.shape) < p).astype(bool)
    out[mask] = rn.poisson(out[mask] * rn.gamma(10., 2., size=out[mask].shape))
    return out, mask


def detect(Y, K=5, alpha=0.1, thresh=1e-5):
    """Detect anomalies using BPTF.

    This method fits BPTF to Y and obtains Mu, which is the model's
    reconstruction of Y (computed from the inferred latent factors).
    Anomalies are then all entries of Y whose probability given Mu
    is less than a given threshold.

        If P(y | mu) < thresh ==> y is  anomaly!

        Here P(y | mu) = Pois(y; mu), the PMF of the Poisson distribution.

    PARAMS:
    Y -- (np.ndarray) data count tensor
    K -- (int) number of latent components
    alpha -- (float) shape parameter of gamma prior over factors
    thresh -- (float) anomaly threshold (between 0 and 1).
    """
    bptf = BPTF(data_shape=Y.shape,
                n_components=K,
                alpha=alpha)
    bptf.fit(Y, max_iter=100, tol=1e-4, smoothness=100, verbose=False)
    Mu = bptf.reconstruct()
    return st.poisson.pmf(Y, Mu) < thresh

if __name__ == '__main__':
    # Generate true Poisson rates and data count tensor
    K = 5
    alpha = 0.1
    Mu, Y = generate(K=K, alpha=alpha)

    # Obtain the corrupted count tensor
    p = 0.01
    corrupted_Y, mask = corrupt(Y, p=p)
    assert corrupted_Y.shape == Y.shape
    assert mask.shape == Y.shape
    print('%d entries corrupted' % (mask.sum()))

    thresh = 1e-5

    # Calculate an upper bound on the results (using the true underlying Mu)
    print('\n----Upper bound on results (using true Mu)----')
    detected = st.poisson.pmf(corrupted_Y, Mu) < thresh
    assert detected.shape == Y.shape
    print('precision: %f' % precision_score(mask.ravel(), detected.ravel()))
    print('recall: %f' % recall_score(mask.ravel(), detected.ravel()))
    print('f1: %f' % f1_score(mask.ravel(), detected.ravel()))

    # Calculate results using BPTF
    print('\n----Results (using BPTF)----')
    detected = detect(corrupted_Y, K=K, alpha=alpha, thresh=thresh)
    assert detected.shape == Y.shape
    print('precision: %f' % precision_score(mask.ravel(), detected.ravel()))
    print('recall: %f' % recall_score(mask.ravel(), detected.ravel()))
    print('f1: %f' % f1_score(mask.ravel(), detected.ravel()))

