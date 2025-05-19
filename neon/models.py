import jax.numpy as jnp
from jax import random, grad, vmap, pmap, tree_map, local_device_count
from jax.lax import pmean, stop_gradient, scan
from flax import linen as nn
from flax import jax_utils
from flax.training.train_state import TrainState
from flax.core import frozen_dict
import optax

from functools import partial
from typing import Callable

import archs


def _create_encoder_arch(config):
    if config.name == 'MlpEncoder':
        arch = archs.MlpEncoder(**config)

    elif config.name == 'ConvEncoder':
        arch = archs.ConvEncoder(**config)

    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')

    return arch

def _create_decoder_arch(config):
    if config.name == 'LinearDecoder':
        arch = archs.LinearDecoder(**config)

    elif config.name == 'ConcatDecoder':
        arch = archs.ConcatDecoder(**config)

    elif config.name == 'SplitDecoder':
        arch = archs.SplitDecoder(**config)

    elif config.name == 'MrhDecoder':
        arch = archs.MrhDecoder(**config)

    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')

    return arch

def _create_epitrain_arch(config):
    if config.name == 'EpiTrain':
        arch = archs.EpiTrain(**config)

    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')

    return arch

def _create_epiprior_arch(config):
    if config.name == 'EpiPrior':
        arch = archs.EpiPrior(**config)

    else:
        raise NotImplementedError(
            f'Arch {config.name} not supported yet!')

    return arch

def _create_optimizer(config):
    def create_mask():
        mask = {'params': {'base_net': 'base_net', 
                           'epi_train': 'epi_train', 
                           'epi_prior': 'epi_prior'}}
        return frozen_dict.freeze(mask)

    def zero_grads():
        def init_fn(_): 
            return ()
        def update_fn(updates, state, params=None):
            return tree_map(jnp.zeros_like, updates), ()
        return optax.GradientTransformation(init_fn, update_fn)

    if config.optimizer == 'Adam':
        lr = optax.exponential_decay(init_value=config.learning_rate,
                                     transition_steps=config.decay_steps,
                                     decay_rate=config.decay_rate)
        
        optimizer = optax.multi_transform({'base_net': optax.adam(learning_rate=lr),
                                           'epi_train': optax.adam(learning_rate=lr),
                                           'epi_prior': zero_grads()}, 
                                            create_mask())
        optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                                optimizer)    
    elif config.optimizer == 'Adam_warmup_exponential':
        lr = optax.warmup_exponential_decay_schedule(init_value=config.init_learning_rate,
                                                     peak_value=config.peak_value,
                                                     warmup_steps=config.warmup_steps,
                                                     transition_steps=config.transition_steps,
                                                     decay_rate=config.decay_rate,
                                                     transition_begin=config.transition_begin,
                                                     end_value=config.end_value)
        
        optimizer = optax.multi_transform({'base_net': optax.adam(learning_rate=lr),
                                           'epi_train': optax.adam(learning_rate=lr),
                                           'epi_prior': zero_grads()}, 
                                            create_mask())
        optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                                optimizer)
    
    elif config.optimizer == 'Adam_cos': # Adam with cossine anealing
        lr = optax.warmup_cosine_decay_schedule(init_value=config.init_learning_rate,
                                                peak_value=config.peak_learning_rate,
                                                warmup_steps=config.warmup_steps,
                                                decay_steps=config.decay_steps,
                                                end_value=config.end_value)
        
        optimizer = optax.multi_transform({'base_net': optax.adam(learning_rate=lr),
                                           'epi_train': optax.adam(learning_rate=lr),
                                           'epi_prior': zero_grads()}, 
                                            create_mask())
        optimizer = optax.chain(optax.clip_by_global_norm(config.grad_clip),
                                optimizer)

    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def _create_train_state(config, params=None, pre_train=False, different_init_key=None):
    # Build architecture
    encoder = _create_encoder_arch(config.encoder_arch)
    decoder = _create_decoder_arch(config.decoder_arch)
    base_net = NeuralOperator(encoder, decoder)
    epi_train = _create_epitrain_arch(config.epitrain_arch)
    epi_prior = _create_epiprior_arch(config.epiprior_arch)
    arch = ENN(base_net, epi_train, epi_prior, config.ensemble_size, config.scale, config.output_activation)
    # Initialize parameters
    u = jnp.ones(config.input_dim)
    y = jnp.ones(config.query_dim)
    z = jnp.ones(config.ensemble_size)
    key = random.PRNGKey(config.seed)
    key, subkey = random.split(key)
    if params is None: # initialize new parameters
        if different_init_key is not None: # use different key if asked
            subkey = different_init_key
        params = arch.init(subkey, u, y, z)
        params = frozen_dict.freeze(params) #TODO make sure this is correct
    else: # use given parameters
        params = params
    # print(arch.tabulate(key, u, y, z))
    # Vectorized apply function across a mini-batch
    apply_fn = vmap(vmap(arch.apply, in_axes=(None, 0, 0, None)), in_axes=(None, None, None, 0))
    # Optimizer
    if pre_train:
        tx = _create_optimizer(config.pre_train)
    else:
        tx = _create_optimizer(config.optim)
    # Create state
    state = TrainState.create(apply_fn=apply_fn,
                              params=params,
                              tx=tx)
    # Replicate state across devices
    state = jax_utils.replicate(state) 
    return state


class NeuralOperator(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    @nn.compact
    def __call__(self, u, y):
        latents = self.encoder(u)
        outputs, features = self.decoder(latents, y)
        return outputs, latents, features

    def _encode(self, u):
        latents = self.encoder(u)
        return latents

    def _decode(self, beta, y):
        outputs, _ = self.decoder(beta, y)
        return outputs


class ENN(nn.Module):
    base_net: nn.Module
    epi_train: nn.Module
    epi_prior: nn.Module
    ensemble_size: int=8
    scale: float=1.0
    output_activation: Callable=lambda x : x
        
    @nn.compact
    def __call__(self, u, y, z):
        ''' Accepts a single data-pair, then use vmap to vectorize'''
        # Base net
        outputs, latents, features = self.base_net(u, y)
        # EpiNet (trainable)
        x_tilde = stop_gradient(jnp.concatenate([latents, 
                                                 features, 
                                                 jnp.tile(y, (latents.shape[-1]//y.shape[-1],))], axis=-1))
        sigma_L = self.epi_train(x_tilde, z)
        # PriorNet (fixed)
        x_tilde = jnp.tile(x_tilde, (self.ensemble_size, 1))
        sigma_P = self.epi_prior(x_tilde, z)
        # Final outputs
        outputs = outputs + sigma_L + self.scale*sigma_P
        return self.output_activation(outputs)


# Define the model
class OperatorRegression:
    def __init__(self, config, params=None, pre_train=False, different_init_key=None): 
        self.config = config
        self.different_init_key = different_init_key
        self.state = _create_train_state(config, params, pre_train, different_init_key)

    # Computes total loss
    def loss(self, params, batch):
        inputs, targets, weights = batch
        u, y, z = inputs
        w = weights
        outputs = self.state.apply_fn(params, u, y, z)
        if self.config.huber_delta is not None:
            loss = (w*optax.huber_loss(outputs, targets, delta=self.config.huber_delta).sum((-1,-2))).mean()
        else:
            loss = (w*optax.l2_loss(outputs, targets).sum((-1,-2))).mean()
        return loss

    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def eval_loss(self, params, batch):
        inputs, targets, weights = batch
        u, y, z = inputs
        w = weights
        outputs = self.state.apply_fn(params, u, y, z)
        if self.config.huber_delta is not None:
            loss = (w*optax.huber_loss(outputs, targets, delta=self.config.huber_delta).sum((-1,-2))).mean()
        else:
            loss = (w*optax.l2_loss(outputs, targets).sum((-1,-2))).mean()
        return loss

    # Define a compiled update step
    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def step(self, state, batch):
        grads = grad(self.loss)(state.params, batch)
        grads = pmean(grads, 'num_devices')
        state = state.apply_gradients(grads=grads)
        return state
    
    # resets the train_state optimizer
    def reset_optimizer(self):
        reset_state = _create_train_state(self.config, params=jax_utils.unreplicate(self.state.params))
        return reset_state


    def posterior_samples(self, key, u, y, num_samples=512):
        ensemble_size = self.config.ensemble_size
        num_devices = local_device_count()
        assert num_samples % (ensemble_size*num_devices) == 0, \
            "Number of samples needs to be divisible by (ensemble_size*num_devices)."
        
        num_steps = num_samples // (ensemble_size*num_devices)
        keys = random.split(key, num_steps)
        sample_fn = lambda params, z: self.state.apply_fn(params, u, y, z)

        def body_fn(keys, i):
            z = random.normal(keys[i], (num_devices, ensemble_size, ensemble_size))
            s = pmap(sample_fn, in_axes=(0,0))(self.state.params, z)
            return keys, s

        _, samples = scan(body_fn, keys, jnp.arange(num_steps))
        _, _, _, num_queries, out_dim = samples.shape
        return samples.reshape(num_steps*num_devices*ensemble_size, num_queries, out_dim)
