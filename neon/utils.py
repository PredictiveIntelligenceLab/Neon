import jax.numpy as jnp
from jax import random
from jax import jit, pmap, tree_map, process_index, device_get, local_device_count
from flax.training import checkpoints
from flax import jax_utils

from torch.utils import data
from functools import partial

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