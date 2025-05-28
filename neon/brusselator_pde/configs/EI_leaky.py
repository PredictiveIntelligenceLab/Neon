import ml_collections
import jax.numpy as jnp
from flax import linen as nn


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.job_type = 'active_learning'

    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = 'brusselator_pde_new'
    wandb.name = None
    wandb.tag = None

    # Simulation settings
    config.input_dim = (4,)
    config.query_dim = 2
    config.eps_dim = 16
    config.ensemble_size = 16
    config.scale = 1.0

    # Network output settings
    config.exponential_output = False
    config.output_activation = lambda x: x

    # Encoder Architecture
    config.encoder_arch = encoder_arch = ml_collections.ConfigDict()
    encoder_arch.name = 'MlpEncoder'
    encoder_arch.latent_dim = 96
    encoder_arch.num_layers = 2
    encoder_arch.hidden_dim = 64
    encoder_arch.activation = nn.gelu

    # Decoder Architecture
    config.decoder_arch = decoder_arch = ml_collections.ConfigDict()
    decoder_arch.name = 'SplitDecoder'
    decoder_arch.num_layers = 3
    decoder_arch.hidden_dim = 64
    decoder_arch.output_dim = 2
    decoder_arch.pos_enc = ml_collections.ConfigDict({'type': 'fourier', 'freq': 5.})
    decoder_arch.activation: jnp.gelu
    decoder_arch.output_activation: lambda x: x

    # EpiNet settings
    config.epitrain_arch = epitrain_arch = ml_collections.ConfigDict()
    epitrain_arch.name = 'EpiTrain'
    epitrain_arch.num_layers = 2
    epitrain_arch.hidden_dim = 32
    epitrain_arch.output_dim = config.ensemble_size*config.decoder_arch.output_dim
    epitrain_arch.activation = nn.gelu

    config.epiprior_arch = epiprior_arch = ml_collections.ConfigDict()
    epiprior_arch.name = 'EpiPrior'
    epiprior_arch.num_layers = 2
    epiprior_arch.hidden_dim = 5
    epiprior_arch.output_dim = config.decoder_arch.output_dim
    epiprior_arch.activation = nn.gelu

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.batch_size = 32
    training.max_steps = 4_000
    training.save_every_steps = None
    training.restart_checkpoint = None

    # Optimizer
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = 'Adam'
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.8
    optim.decay_steps = 1_000

    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 100
    logging.log_losses = True
    logging.log_regret = True

    # Sequential data acquisition
    config.sequential = sequential = ml_collections.ConfigDict()
    sequential.num_iters = 30
    sequential.acquisition = 'EI_leaky'
    sequential.q = 1
    sequential.norm_order = 2
    sequential.num_restarts = 500
    sequential.log_regret = True
    sequential.alpha = 0.01

    return config