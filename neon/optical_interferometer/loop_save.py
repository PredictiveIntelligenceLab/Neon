# import os
# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
# os.environ['TF_CUDNN_DETERMINISTIC'] ='1'  # For better reproducible!  ~35% slower !

from absl import app
from absl import flags

import jax
import jax.numpy as jnp
from jax import random
from flax.training.train_state import TrainState
from flax import jax_utils

import pickle
from ml_collections import config_flags


import sequential
from generate_data import generate_lhs_data, import_data

import sys
sys.path.append('..') # makes modules in parent repository available to import
from models import OperatorRegression, _create_optimizer


FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=False)
flags.DEFINE_integer('seed', None, 'Seed for model initialization')


def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir
    seed = FLAGS.seed
    config.seed = seed

    # initializes training dataset
    num_init = 15 # number of initial examples
    if config.seed in [0,1,2,3,4]: # import initial data from datesets from other papers for fair comparison
        dataset = import_data(num_init, config.seed)
    else: # create new random dataset
        dataset = generate_lhs_data(num_init, config.seed)
    
    # initializes model
    key, init_key = random.split(random.PRNGKey(config.seed))
    if config.continued_training:
        model = OperatorRegression(config, pre_train=True, different_init_key=init_key)
    else:
        model = OperatorRegression(config, pre_train=False, different_init_key=init_key)

    for iteration in range(config.sequential.num_iters):
        # iteratively augments dataset with new points
        dataset, model = sequential.train_and_collect(config, dataset, model, iteration)
        if config.continued_training: # reset optimizer state but maintain parameters
            model.state = model.reset_optimizer() # keeps parameters but resets optimizer
        else: # restarts model completely
            key, init_key = random.split(key)
            model = OperatorRegression(config, pre_train=False, different_init_key=init_key)
        
        # clear jax cache every few iterations to avoid OoM issues
        if iteration % 5 == 4:
            jax.clear_caches()
    
    # create file name to save collected points
    file_name = 'collected_datasets/dataset_' + str(config.sequential.num_iters) + '_'
    if config.sequential.acquisition in ['EE', 'GEE', 'GEE_R']:
        file_name += config.sequential.acquisition +'_sig' + str(config.sequential.sig)
    elif config.sequential.acquisition in ['LCB']:
        file_name += config.sequential.acquisition +'_kappa' + str(config.sequential.kappa)
    elif config.sequential.acquisition in ['EI_leaky']:
        file_name += config.sequential.acquisition +'_alpha' + str(config.sequential.alpha)
    else:
        file_name += config.sequential.acquisition
    try:
        if not (config.sequential.output_weights(jnp.ones((2,16))) == jnp.ones((2,))).all():
            file_name += '_weighted'
    except AttributeError:
        pass
    file_name += '_q' + str(config.sequential.q) + '_seed' + str(config.seed) + '.npz'
    with open(file_name, 'wb') as f:
        pickle.dump(dataset, f)


if __name__ == "__main__":
    flags.mark_flags_as_required(['config', 'workdir'])
    app.run(main)
