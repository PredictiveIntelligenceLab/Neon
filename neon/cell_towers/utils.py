import jax.numpy as jnp
from jax import random, jit, vmap
from jax import jit, pmap, tree_map, process_index, device_get, local_device_count
from flax.training import checkpoints
from flax import jax_utils

from torch.utils import data
from functools import partial

from generate_data import soln_dim

def save_checkpoint(state, workdir):
    if process_index() == 0:
        state = device_get(tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=3, overwrite=True)

def restore_checkpoint(model, workdir):
    state = checkpoints.restore_checkpoint(workdir, model.state)
    model.state = jax_utils.replicate(state) 
    return model


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, w, 
                 ensemble_size=8, batch_size=64, 
                 rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = jnp.tile(s[jnp.newaxis,...], (ensemble_size,1,1))
        self.w = jnp.tile(w[jnp.newaxis,...], (ensemble_size,1,1))
        self.N = u.shape[0]
        self.ensemble_size = ensemble_size
        self.batch_size = batch_size
        self.key = rng_key
        self.num_devices = local_device_count()

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        keys = random.split(subkey, self.num_devices)
        inputs, targets, weights = self.__data_generation(keys)
        return inputs, targets, weights

    @partial(jit, static_argnums=(0,))
    @partial(pmap, static_broadcasted_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        z = random.normal(key, (self.ensemble_size, self.ensemble_size))
        u = self.u[idx,...]
        y = self.y[idx,...]
        s = self.s[:,idx,...]
        w = self.w[:,idx,...]
        # Construct batch
        inputs = (u, y, z)
        targets = s
        weights = w
        return inputs, targets, weights
    


@jit
def re_organize(us, ys, ws=None, ss=None):
    '''
    Inputs:
        us; shape (..., N, m_in)
        ys; shape (..., N, m_out, P)
        ws; shape (..., N, 1) [optional]
        ss; shape (..., N, m_out, sol_dim) [optional]
    Outputs
        us; shape (..., N*m_out, m_in)
        ys; shape (..., N*m_out, P)
        ws; shape (..., N*m_out, 1)
        ss; shape (..., N*m_out, 2)
    '''
    m_out = ys.shape[-2]
    N = us.shape[-2]
    P = ys.shape[-1]

    assert (ys.shape[-3] == N), 'Wrong dimensions'
    us = jnp.repeat(us, m_out, axis=-2) # shape (..., N*m_out, m_in)
    ys = ys.reshape(ys.shape[:-3] + (N*m_out, P)) # shape (..., N*m_out, 2)
    
    if ws is not None:
        assert (ws.shape[-2] == N), 'Wrong dimensions'
        ws = jnp.repeat(ws, m_out, axis=-2) # shape (..., N*m_out, 1)
    
    if ss is not None:
        assert (ss.shape[-3] == N) and (ss.shape[-2] == m_out), 'Wrong dimensions'
        ss = ss.reshape(ss.shape[:-3] + (N*m_out, ss.shape[-1])) # shape (..., N*m_out, 1)

    return tuple(filter(lambda x : x is not None,
                        (us, ys, ws, ss)))




def compute_relative_error(config, model, batch, norm_stats=(0.,1.,0.,1.,0.,1.), max_b_size=50, demean=False):
    '''
    Average L2 error over all functions
    '''
    key = random.PRNGKey(config.seed)
    u, y, s, w = batch # unpack batch
    mu_u, sig_u, mu_y, sig_y, mu_s, sig_s = norm_stats # unpack normalization statistics
    u = (u-mu_u)/sig_u
    y = (y-mu_y)/sig_y
    z_test = random.normal(key, (config.ensemble_size, config.ensemble_size))
    N = u.shape[0] # number of functions
    num_batches =  int(jnp.ceil(N / max_b_size))
    tot_er = 0.
    pred_fn = lambda u_aux, y_aux : vmap(model.state.apply_fn, in_axes=(0,None,None,None))(model.state.params, u_aux, y_aux, z_test)[0,...]
    pred_fn = jit(pred_fn)
    for i in range(num_batches):
        start = i*max_b_size
        stop = min((i+1)*max_b_size, N)
        num_in_batch = stop-start
        u_aux, y_aux, s_aux, w_aux = re_organize(u[start:stop], y[start:stop], w[start:stop], s[start:stop])
        s_pred = pred_fn(u_aux, y_aux)
        s_mean = s_pred.mean(axis=0) # average prediction; shape (N*m,sol_dim)
        s_mean = s_mean.reshape((num_in_batch,-1,soln_dim)) # shape (N,m,sol_dim)
        s_mean = (s_mean*sig_s)+mu_s
        error = (s_mean - s[start:stop])**2 # point-wise absolute L2 error; shape (N, m, sol_dim)
        if demean:
            alt_norm = jnp.linalg.norm(s[start:stop]-mu_s, axis=(-1,-2), keepdims=True) # shape (N,...)
            error = error / (alt_norm**2) # point-wise relative error; shape (N,m, sol_dim)
        else:
            error = error * (w[start:stop][:,None]) # point-wise relative error; shape (N,m, sol_dim)
        error = jnp.sqrt(error.sum(axis=(-1,-2))) # vector of relative errors; shape (N,)
        tot_er = tot_er + error.mean()*num_in_batch/N

    return tot_er # scalar