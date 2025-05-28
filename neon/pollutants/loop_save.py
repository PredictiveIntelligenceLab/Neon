# import os
# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
# os.environ['TF_CUDNN_DETERMINISTIC'] ='1'  # For better reproducible!  ~35% slower !

from absl import app
from absl import flags

import jax
import jax.numpy as jnp
from jax import random

import pickle

from ml_collections import config_flags


import sequential
from sequential import get_initial_dataset

import sys
sys.path.append('..') # makes modules in parent repository available to import
from models import OperatorRegression

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

    # generate initial dataset
    key = random.PRNGKey(seed)
    dataset = get_initial_dataset(key)

    # initializes model
    key, init_key = random.split(random.PRNGKey(config.seed))
    model = OperatorRegression(config, pre_train=False, different_init_key=init_key)

    for iteration in range(config.sequential.num_iters):
        # iteratively augments dataset with new points
        dataset = sequential.train_and_collect(config, dataset, iteration, model)
        # re initializes model using new key
        key, init_key = random.split(key)
        model = OperatorRegression(config, pre_train=False, different_init_key=init_key)
    
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
