import numpy as np
from tensorly.contrib.sparse import tensor as sptensor

from IPython import embed

from pathlib import Path

from bptf import BPTF, save_bptf, load_bptf
from plot_components import plot_all_components

# load compressed data
data = np.load(Path(__file__).parent.joinpath('dat/1995-2013-M/data_compressed.npz'))

# get the dyadic event tensor and labels
Y, actors, actions, dates = (data[key] for key in ['Y', 'actors', 'actions', 'dates'])

# set whether to treat the diagonal as missing
drop_diag = False

mask = None
if drop_diag:
    mask = np.zeros(Y.shape).astype(int)
    mask[np.identity(mask.shape[0]).astype(bool)] = 1
    mask = sptensor(mask).astype(int)

# set K and alpha and create model
n_components = 50
alpha = 0.1
model = BPTF(data_shape=Y.shape, 
             n_components=n_components, 
             alpha=alpha)

# define where to save results
outdir = Path(__file__).parent.joinpath('results')
if mask is None:
    outdir = outdir.joinpath(f'without_mask_K_{n_components}')
else:
    outdir = outdir.joinpath(f'with_mask_K_{n_components}')

model.fit(data=Y, 
          mask=mask, 
          missing_val=1, 
          verbose=True, 
          max_iter=500)

save_bptf(model, 
          outdir=outdir, 
          filename='bptf', 
          filepath=None)

# you can also load a model that has neen saved
# model = load_bptf(outdir.joinpath('bptf_1.npz'))

plot_all_components(model=model, 
                    actors_N=actors, 
                    actions_A=actions, 
                    dates_T=dates, 
                    title='',
                    figdir=outdir)

embed()