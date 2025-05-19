import jax.numpy as jnp
from flax import linen as nn
from jax import vmap

from typing import Any, Callable, Sequence, Optional, Union, Dict

from layers import Dense, Conv, FourierEnc, MultiresEnc, get_voxel_vertices

# identity function
identity = lambda x : x

def periodic_encoding(x, L=1.0):
    x = jnp.hstack([jnp.cos(2.0*jnp.pi*x/L), jnp.sin(2.0*jnp.pi*x/L)])
    return x 

class MLP(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(self.hidden_dim)(x)
            x = self.activation(x)
        x = Dense(self.output_dim)(x)
        return x

class MlpWithFeatures(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for _ in range(self.num_layers):
            x = Dense(self.hidden_dim)(x)
            x = self.activation(x)
        outputs = Dense(self.output_dim)(x)
        return outputs, x     

class EpiTrain(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu
    
    @nn.compact
    def __call__(self, x, z):
        inputs = jnp.concatenate([x, z], axis=-1)
        outputs = MLP(self.num_layers,
                      self.hidden_dim,
                      self.output_dim,
                      self.activation)(inputs)
        outputs = jnp.reshape(outputs, (outputs.shape[0]//z.shape[0], z.shape[0]))
        outputs = jnp.dot(outputs, z)
        return outputs
    

class EpiPrior(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    activation: Callable=nn.gelu
        
    @nn.compact
    def __call__(self, x, z):
        ensemble = nn.vmap(MLP,
                           in_axes=0, out_axes=0,
                           variable_axes={'params': 0},
                           split_rngs={'params': True})
        outputs = ensemble(self.num_layers,
                           self.hidden_dim,
                           self.output_dim,
                           self.activation)(x)
        outputs = jnp.einsum('ij,i->j', outputs, z)
        return outputs
    

class MlpEncoder(nn.Module):
    latent_dim: int=1
    num_layers: int=2
    hidden_dim: int=64
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        x = MLP(self.num_layers,
                self.hidden_dim,
                self.latent_dim,
                self.activation)(x)
        return x

class ConvEncoder(nn.Module):
    latent_dim: int
    out_channels: Sequence[int]
    activation: Callable=nn.gelu

    @nn.compact
    def __call__(self, x):
        for i in range(len(self.out_channels)):
            x = Conv(self.out_channels[i],  kernel_size=(2,2), strides=(2,2), padding="SAME")(x)
            x = self.activation(x)
        x = x.flatten()
        x = Dense(self.out_dim)(x)
        return x

class LinearDecoder(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    pos_enc: Union[None, Dict] = None
    activation: Callable=nn.gelu
    output_activation: Callable=identity

    @nn.compact
    def __call__(self, beta, y):
        if self.pos_enc['type'] == 'periodic':
            y = periodic_encoding(y, self.pos_enc['L'])
        elif self.pos_enc['type'] == 'fourier':
            y = FourierEnc(self.pos_enc['freq'], beta.shape[-1])(y)
        y = MLP(self.num_layers,
                self.hidden_dim,
                self.output_dim,
                self.activation)(y)
        features = jnp.concatenate([beta, y], axis=-1)
        outputs = jnp.sum(beta*y, axis=-1, keepdims=True)
        return self.output_activation(outputs), features

class ConcatDecoder(nn.Module):
    num_layers: int=2
    hidden_dim: int=64
    output_dim: int=1
    pos_enc: Union[None, Dict] = None
    activation: Callable=nn.gelu
    output_activation: Callable=identity

    @nn.compact
    def __call__(self, beta, y):
        if self.pos_enc is None:
            y = jnp.tile(y, (beta.shape[-1]//y.shape[0],))
        elif self.pos_enc['type'] == 'periodic':
            y = jnp.tile(y, (beta.shape[-1]//2,))
            y = periodic_encoding(y, self.pos_enc['L'])
        elif self.pos_enc['type'] == 'fourier':
            y = FourierEnc(self.pos_enc['freq'], beta.shape[-1])(y)
        elif self.pos_enc['type'] == 'multires':
            y = MultiresEnc(self.pos_enc['num_levels'],
                            self.pos_enc['min_res'],
                            self.pos_enc['max_res'],
                            self.pos_enc['hash_size'],
                            self.pos_enc['num_features'])(y)
        elif self.pos_enc['type'] == 'none':
            y = jnp.tile(y, (beta.shape[-1]//y.shape[0],))
        outputs, features = MlpWithFeatures(self.num_layers,
                                            self.hidden_dim,
                                            self.output_dim,
                                            self.activation)(jnp.concatenate([beta, y], axis=-1))
        return self.output_activation(outputs), features

class SplitDecoder(nn.Module):
    num_layers: int=8
    hidden_dim: int=256
    output_dim: int=2
    pos_enc: Union[None, Dict] = None
    activation: Callable=nn.gelu
    output_activation: Callable=identity

    @nn.compact
    def __call__(self, beta, y):
        beta = jnp.split(beta, self.num_layers)
        if self.pos_enc['type'] == 'fourier':
            y = FourierEnc(self.pos_enc['freq'], beta[0].shape[-1])(y)
        elif self.pos_enc['type'] == 'multires':
            y = MultiresEnc(self.pos_enc['num_levels'],
                            self.pos_enc['min_res'],
                            self.pos_enc['max_res'],
                            self.pos_enc['hash_size'],
                            self.pos_enc['num_features'])(y)
        elif self.pos_enc['type'] == 'none':
            y = jnp.tile(y, (beta[0].shape[-1]//y.shape[0],))
        for i in range(self.num_layers):
            y = Dense(self.hidden_dim)(jnp.concatenate([y, beta[i]]))
            y = self.activation(y)
        outputs = Dense(self.output_dim)(y)
        return self.output_activation(outputs), y

