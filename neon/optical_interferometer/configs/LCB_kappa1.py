import ml_collections
import jax.numpy as jnp
from flax import linen as nn


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.job_type = 'bayesian_optimization_optical_interferometer'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = 'optical_interferometer_late_feb_24'
    wandb.name = None
    wandb.tag = None

    # Simulation settings
    config.input_dim = (4,)
    config.query_dim = 2
    config.objective_eps = 0.
    config.huber_delta = 1. # setting to 1 equals to regular L2 loss for the optical interferometer case
    config.eps_dim = 16
    config.ensemble_size = 16
    config.scale = 1.
    config.continued_training = False

    # Network output settings
    config.exponential_output = True
    config.output_activation = lambda x: nn.sigmoid(x[...,:16])*jnp.exp(-nn.softplus(x[...,16:]))

    # Encoder Architecture
    config.encoder_arch = encoder_arch = ml_collections.ConfigDict()
    encoder_arch.name = 'MlpEncoder'
    encoder_arch.latent_dim = 96
    encoder_arch.num_layers = 4
    encoder_arch.hidden_dim = 64
    encoder_arch.activation = nn.gelu

    # Decoder Architecture
    config.decoder_arch = decoder_arch = ml_collections.ConfigDict()
    decoder_arch.name = 'SplitDecoder'
    decoder_arch.num_layers = 6
    decoder_arch.hidden_dim = 128
    decoder_arch.output_dim = 16*2
    decoder_arch.pos_enc = ml_collections.ConfigDict({'type': 'fourier', 'freq': 10.})
    decoder_arch.activation: jnp.gelu
    decoder_arch.output_activation: lambda x: x

    # EpiNet settings
    config.epitrain_arch = epitrain_arch = ml_collections.ConfigDict()
    epitrain_arch.name = 'EpiTrain'
    epitrain_arch.num_layers = 3
    epitrain_arch.hidden_dim = 64
    epitrain_arch.output_dim = config.ensemble_size*config.decoder_arch.output_dim
    epitrain_arch.activation = nn.gelu

    config.epiprior_arch = epiprior_arch = ml_collections.ConfigDict()
    epiprior_arch.name = 'EpiPrior'
    epiprior_arch.num_layers = 2
    epiprior_arch.hidden_dim = 8
    epiprior_arch.output_dim = config.decoder_arch.output_dim
    epiprior_arch.activation = nn.gelu

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 4*4096
    training.max_steps = 15_000
    training.save_every_steps = None
    training.restart_checkpoint = None

    # Optimizer
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam_cos'
    optim.grad_clip = 5e-7
    optim.init_learning_rate = 1e-4
    optim.peak_learning_rate = 5e-2
    optim.warmup_steps = 300
    optim.decay_steps = training.max_steps # must be positive
    optim.end_value = 3e-3 # set to None if no desire for lower bound on lr schedule

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 200
    logging.log_losses = True
    logging.log_regret = True

    # Sequential data acquisition
    config.sequential = sequential = ml_collections.ConfigDict()
    sequential.num_iters = 85
    sequential.acquisition = 'LCB'
    sequential.q = 1
    sequential.norm_order = 2
    sequential.num_restarts = 500
    sequential.log_regret = True
    sequential.kappa = 1.
    sequential.required_init_pts = 15

    return config

