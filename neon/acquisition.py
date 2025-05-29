import numpy as onp
from jax import numpy as jnp
from jax import jit, vjp, random
from jax.scipy.special import expit as sigmoid
from jax.scipy.special import logsumexp
from jax.nn import leaky_relu
from scipy.optimize import minimize
#from jax.scipy.optimize import minimize
from functools import partial
from pyDOE import lhs
from tqdm import trange

def minimize_lbfgs(objective, x0, bnds = None, verbose = False, maxfun = 15000):
    if verbose:
        def callback_fn(params):
            print("Loss: {}".format(objective(params)[0]))
    else:
        callback_fn = None
        
    result = minimize(objective, x0, jac=True,
                      method='L-BFGS-B', bounds = bnds,
                      callback=callback_fn, options = {'maxfun':maxfun})
    return result.x, result.fun

class MCAcquisition:
    def __init__(self, posterior, bounds, *args, 
                acq_fn = 'LCB', 
                norm = lambda x: jnp.linalg.norm(x,  axis=-1, ord=2),
                output_weights=lambda x: jnp.ones(x.shape[0]),
                sig=1,
                beta = 0.2):  
        self.posterior = posterior
        self.bounds = bounds            # domain bounds
        self.args = args                # arguments required by different acquisition functions
        self.acq_fn = acq_fn            # a string indicating the chosen acquisition function
        self.norm = norm
        self.weights = output_weights   # a callable function returning the likelihood weighted weights
        self.sig = sig
        self.beta = beta

    def evaluate(self, x):
        # Inputs are (q x d), use vmap to vectorize across a batch
        # samples[...,0]  corresponds to the objective function
        # samples[...,1:] corresponds to the constraints
        # samples[...,i] are (q x ensemble_size x queries x values)
        q = x.shape[0]
        # Common acquisition functions
        if self.acq_fn == 'EI': #maximizes expected improvement
            best = self.args[0]
            samples = self.posterior(x)[...,0]
            reparam = jnp.maximum(best-samples, 0)
            EI = jnp.mean(jnp.max(reparam, axis=0))
            return -EI
        if self.acq_fn == 'EI_leaky': #maximizes expected improvement, with a twist!
            # this version should only be used when q=1
            best = self.args[0]
            samples = self.posterior(x)[...,0]
            alpha = self.args[1]
            reparam = leaky_relu(best-samples, negative_slope=alpha)
            EI = jnp.mean(jnp.max(reparam, axis=0))
            return -EI
        if self.acq_fn == 'qEI_leaky': #maximizes expected improvement, with a twist!
            # this version works for q >= 1
            best = self.args[0]
            samples = self.posterior(x)[...,0]
            alpha = self.args[1]
            reparam = leaky_relu(best-samples, negative_slope=alpha)
            # here the '+' plays the role of 'or' in a way jax can easily jit compile
            weights = jnp.where((reparam==jnp.max(reparam, axis=0)) + (reparam<0), 1, alpha) # same shape as reparam
            EI = jnp.mean((reparam*weights).sum(0))
            return -EI
        elif self.acq_fn == 'LCB':
            kappa = self.args[0]
            samples = self.posterior(x)[...,0]
            mu = jnp.mean(samples, axis=1, keepdims=True)
            weights = self.weights(x).reshape(q,1)
            #uncertainty = self.norm(samples-mu)
            uncertainty = jnp.abs(samples-mu)
            reparam = mu - jnp.sqrt(0.5*jnp.pi*kappa) * weights * uncertainty
            LCB = jnp.mean(jnp.min(reparam, axis=0))
            return LCB
        elif self.acq_fn == 'TS':
            rng_key = self.args[0]
            samples = self.posterior(x)[...,0]
            idx = random.randint(rng_key, (1,), minval=0, maxval=samples.shape[0])
            reparam = samples[idx,:].reshape(1,q)
            TS = jnp.mean(jnp.min(reparam, axis=0))
            return TS
        elif self.acq_fn == 'CLSF':
            kappa = self.args[0]
            samples = self.posterior(x)[...,0]
            mu = jnp.mean(samples, axis=1, keepdims=True)
            weights = self.weights(x).reshape(1,q)
            reparam = jnp.abs(jnp.sqrt(0.5*jnp.pi) / (jnp.abs(mu)**(1.0/kappa) + 1e-8) * weights * jnp.abs(samples - mu))
            CLSF = jnp.mean(jnp.max(reparam, axis=-1))
            return -jnp.log(CLSF)
        
        # That's all for now..
        else:
            raise NotImplementedError

    @partial(jit, static_argnums=(0,))
    def acq_value_and_grad(self, inputs):
        primals, f_vjp = vjp(self.evaluate, inputs)
        grads = f_vjp(jnp.ones_like(primals))[0]
        return primals, grads

    def next_best_point(self, q = 1, num_restarts = 10, maxfun=15000, required_init_pts=None):
        lb, ub = self.bounds   
        dim = lb.shape[0]
        if self.acq_fn == 'random': #inactive policies
            x_new = onp.random.normal(size=(q,dim)) # random normal
            x_new = jnp.array(x_new.flatten())
        elif self.acq_fn == 'random_unif': #inactive policies
            x_new = onp.random.uniform(low=lb, high=ub, size=(q,dim)) # random uniform
            x_new = jnp.array(x_new.flatten())
        else:
            # Define objective that returns float64 NumPy arrays
            def objective(x):
                x = x.reshape(q, dim)
                value, grads = self.acq_value_and_grad(x)
                out = (onp.array(value, dtype=onp.float64), 
                    onp.array(grads.flatten(), dtype=onp.float64))
                return out
            # Optimize with random restarts
            loc, acq = [], []
            if required_init_pts is None:
                init = lb + (ub-lb)*lhs(dim, q*num_restarts)
                x0 = init.reshape(num_restarts, q, dim)
            else:
                num_new = num_restarts-len(required_init_pts)
                init = lb + (ub-lb)*lhs(dim, q*num_new)
                x0 = init.reshape(num_new, q, dim)
                x0 = jnp.concatenate([required_init_pts, x0], axis=0)
            dom_bounds = tuple(map(tuple, onp.tile(onp.vstack((lb, ub)).T,(q,1))))
            for i in trange(num_restarts):
                pos, val = minimize_lbfgs(objective, x0[i,...].flatten(), bnds = dom_bounds, maxfun=maxfun)
                loc.append(pos)
                acq.append(val)
            loc = jnp.vstack(loc)
            acq = jnp.vstack(acq)
            idx_best = jnp.argmin(acq)
            x_new = loc[idx_best,:]

        return x_new