# import os
# os.environ["XLA_FLAGS"] = '--xla_gpu_autotune_level=0'
# os.environ['TF_CUDNN_DETERMINISTIC'] ='1'  # For better reproducible!  ~35% slower !


from absl import app
from absl import flags

import jax
import jax.numpy as jnp

import pickle

from ml_collections import config_flags


import sequential
from sequential import initial_dataset
from generate_data import generate_data

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

    dataset = generate_data(config.seed)

    for iteration in range(config.sequential.num_iters):
        # iteratively augments dataset with new points
        dataset = sequential.train_and_collect(config, dataset, iteration)
    
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
