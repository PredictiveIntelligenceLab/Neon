import jax.numpy as jnp
from jax import vmap, random, jit

from jax.scipy.special import logsumexp

import wandb
from tqdm.auto import trange
import pickle
import gc


import time
import sys
sys.path.append('..') # makes modules in parent repository available to import

from models import OperatorRegression
from acquisition import MCAcquisition
from utils import DataGenerator
from ground_truth import gt_op

N_y = 64 # grid definition
soln_dim = 16
output_dim = (N_y, N_y, soln_dim)
xx, yy = jnp.meshgrid(jnp.arange(N_y) / N_y, jnp.arange(N_y) / N_y ) # shapes (N_y,N_y) and (N_y,N_y)
y0 = jnp.concatenate([xx[:, :, None], yy[:, :, None]], axis=-1).reshape((-1, 2)) # shape (N_y**2, 2) # values of y; shape (N_y^2, 2)
P = y0.shape[-1] # this should be 2 


dim_u = 4 # grid definition of intput functions
dim_s = N_y**2 # grid definition of output functions
lb = -jnp.ones((dim_u,)) # lower brounds; shape (4,)
ub = jnp.ones((dim_u,)) # upper brounds; shape (4,)
bounds = (lb, ub) # bounds of sampled functions



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



def compute_relative_error(config, model, batch, norm_stats=(0.,1.,0.,1.,0.,1.), max_b_size=50):
    '''
    Average L2 error over all test functions
    '''
    key = random.PRNGKey(config.seed)
    u, y, s, w = batch # unpack batch
    mu_u, sig_u, mu_y, sig_y, mu_s, sig_s = norm_stats # unpack normalization statistics
    u = (u-mu_u)/sig_u
    y = (y-mu_y)/sig_y
    N = u.shape[0] # number of functions
    num_batches =  int(jnp.ceil(N / max_b_size))
    tot_er = 0.
    for i in range(num_batches):
        start = i*max_b_size
        stop = min((i+1)*max_b_size, N)
        num_in_batch = stop-start
        u_aux, y_aux, s_aux, w_aux = re_organize(u[start:stop], y[start:stop], w[start:stop], s[start:stop])
        z_test = random.normal(key, (config.ensemble_size, config.ensemble_size))
        pred_fn = lambda params : model.state.apply_fn(params, u_aux, y_aux, z_test)
        s_pred = vmap(pred_fn)(model.state.params)[0,...] # shape (ensamble_size, N*m, 1)
        s_mean = s_pred.mean(axis=0) # average prediction; shape (N*m,1)
        s_mean = s_mean.reshape((num_in_batch,-1,soln_dim))
        s_mean = (s_mean*sig_s)+mu_s
        error = (s_mean - s[start:stop])**2 # point-wise absolute L2 error; shape (N, m)
        error = error * (w[start:stop][:,None]) # point-wise relative error; shape (N,m)
        error = jnp.sqrt(error.sum(axis=(-1,-2))) # vector of relative errors; shape (N,)
        tot_er = tot_er + error.mean()*num_in_batch/N

    return tot_er # scalar

def eval_step(config, model, batch):

    params = model.state.params
    log_dict = {}

    if config.logging.log_losses:
        loss = model.eval_loss(params, batch)
        loss = loss.mean()
        log_dict['loss'] = loss
    return log_dict

# function mapping vectorial output to scalar obective value
def objective(s, eps=0.):
    s = s.reshape((N_y,N_y,16)).transpose((2,0,1)) + eps
    intens = jnp.exp(-((xx - 0.5) ** 2 + (yy - 0.5) ** 2) / (0.95) ** 2) * s
    ivec = jnp.sum(intens, axis = (-1, -2))
    smax = logsumexp(ivec, -1)
    smin = -logsumexp(-ivec, -1)
    return - (smax - smin) / (smax + smin) # scalar

def train_and_collect(config, dataset, model, iteration):
    # wandb stuff
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project,
               name=wandb_config.name,
               config=dict(config),
               group='trial%d_it%d' % (config.seed, iteration))
    # Define the custom x axis and regret metric
    wandb.define_metric('iteration')
    wandb.define_metric('regret', step_metric='iteration')

    # import test samples
    u_test, y_test, s_test, w_test = pickle.load(open('test_examples.npz', "rb"))
    n_test = len(u_test)

    print('Testing data shapes')
    print('u: {}'.format(u_test.shape))
    print('y: {}'.format(y_test.shape))
    print('s: {}'.format(s_test.shape))
    print('w: {}\n'.format(w_test.shape))

    # Defines a batch consisting of all test data
    TEST_BATCH = (u_test, y_test, s_test, w_test)

    # unpacking dataset
    u_train, y_train, s_train, w_train = dataset
    print('Training data')
    print('u: {}'.format(u_train.shape))
    print('y: {}'.format(y_train.shape))
    print('s: {}'.format(s_train.shape))
    print('w: {}'.format(w_train.shape))

    # parameter space statistics (not dependent on data distribution)
    mu_u = (ub+lb)/2
    sig_u = (ub-lb)/jnp.sqrt(12)
    mu_y = y0.mean(0)
    sig_y = y0.std(0)
    mu_s = 0. # assumes ENN output is between 0 and 1 (sigmoid output activation)
    #sig_s = jnp.log(256) # assumes log transform of s and ENN output is between 0 and 1 (sigmoid output activation)
    sig_s = 255.
    
    norm_stats = (mu_u, sig_u, mu_y, sig_y, mu_s, sig_s)

    u,y,w,s = re_organize((u_train-mu_u)/sig_u,
                          (y_train-mu_y)/sig_y,
                          w_train,
                          (s_train-mu_s)/sig_s)

    # Creating data loader
    data_loader = DataGenerator(u,
                                y,
                                s,
                                w,
                                config.ensemble_size,
                                min(config.training.batch_size, len(u)))
    data = iter(data_loader)
    loss_log = []

    # Training
    num_steps = int(config.training.max_steps*(1 + 0.2*iteration / config.sequential.num_iters))


    # does an extra loop of training at the very first BO iteration if using continued training
    if (iteration == 0) and (config.continued_training):
        pbar = trange(config.pre_train.num_steps)
        for step in pbar:
            # Acquiring + organizing batch
            batch = next(data)
            model.state = model.step(model.state, batch)            
            # logging
            if step % config.logging.log_every_steps == 0:
                log_dict = eval_step(config, model, batch)
                #wandb.log(log_dict, step) don't log losses on this initial train loop
                pbar.set_postfix({'loss' : log_dict['loss']})
        model.state = model.reset_optimizer() # keeps parameters but resets optimizer
    
    # regular training loop
    pbar = trange(num_steps)
    t_start = time.time()
    for step in pbar:
        batch = next(data) # Acquiring batch
        model.state = model.step(model.state, batch) # updating model
        # logging
        if step % config.logging.log_every_steps == 0:
            log_dict = eval_step(config, model, batch)
            wandb.log(log_dict, step)
            pbar.set_postfix({'loss' : log_dict['loss']})
    t_end = time.time()
    wandb.log({'train_time':t_end-t_start})


    
    # Compute test error
    if config.sequential.log_regret == True:
        test_er = compute_relative_error(config, model, TEST_BATCH, norm_stats=norm_stats)
        print(f"Relative L2 test error is {test_er : .3%}")
        train_er = compute_relative_error(config, model, batch=dataset, norm_stats=norm_stats)
        log_dict = {'test_er': test_er, 'train_er': train_er, 'iteration': iteration}
        wandb.log(log_dict)
        print(f"Relative L2 train error is {train_er : .3%}")


    @vmap
    def posterior(x, num_samples=128, key=random.PRNGKey(0)):
        x = (x-mu_u)/sig_u
        x = jnp.tile(x, (dim_s,1)) # shape (dim_s,dim_u)
        y = (y_train[0,...]-mu_y)/sig_y
        samples = model.posterior_samples(key, x, y, num_samples) # shape (num_samples, m, 1)
        samples = (samples*sig_s)+mu_s
        samples = jnp.clip(samples, 0., 255.) # makes sure samples are physically meaningful
        return vmap(objective, in_axes=(0, None))(samples, config.objective_eps)[:,None] # shape (num_samples, 1)


    # Acquisition
    if config.sequential.acquisition == 'LCB':
        args = (config.sequential.kappa,)
    elif config.sequential.acquisition == 'EI':
        objs = vmap(objective)(s_train)
        best = objs.min()
        args = (best,)
    elif config.sequential.acquisition == 'EI_leaky':
        objs = vmap(objective)(s_train)
        best = objs.min()
        args = (best, config.sequential.alpha)
    elif config.sequential.acquisition == 'qEI_leaky':
        objs = vmap(objective)(s_train)
        best = objs.min()
        args = (best, config.sequential.alpha)
    else:
        args = ()
    acq_model = MCAcquisition(posterior, (lb, ub),
                            *args,
                            acq_fn = config.sequential.acquisition,
                            norm = lambda x: jnp.linalg.norm(x,  axis=-1, ord=config.sequential.norm_order))

    q = config.sequential.q # number of new examples to be collected
    t_start = time.time()
    if config.sequential.required_init_pts is not None:
        objs = vmap(objective)(s_train)
        idx = jnp.argsort(objs)[:config.sequential.required_init_pts]
        required_init_pts = u_train[idx] # the best few points collected so far
        required_init_pts = jnp.expand_dims(required_init_pts, 1)
        new_coeffs = acq_model.next_best_point(q=q,
                                               num_restarts=config.sequential.num_restarts,
                                               required_init_pts=required_init_pts)
    else:
        new_coeffs = acq_model.next_best_point(q=q,
                                               num_restarts=config.sequential.num_restarts)
    t_end = time.time()
    wandb.log({'acq_time':t_end-t_start})
    new_coeffs = new_coeffs.reshape(q,-1) # shape (q, dim_u)

  

    # creating new pairs
    # shapes below are (q, dim_u); (q, dim_s, P); (q, dim_s); (q, 1)
    u_new = new_coeffs # (q, dim_u)
    y_new = jnp.tile(y0, (q, 1, 1)) # (q, dim_s, P)
    s_new = jnp.array([gt_op(u) for u in new_coeffs]) # (q, dim_s, sol_dim)
    w_new = 1/jnp.linalg.norm(s_new, axis=(-1,-2))[:,None]**2 # (q,1)

    # logs mean and std of posterior samples of newly acquired point
    samples = posterior(new_coeffs)
    wandb.log({'posterior_mean':samples.mean(), 'posterior_std':samples.std(), 'true_value':vmap(objective)(s_new)})


    # augumenting dataset
    u_train = jnp.concatenate([u_train, u_new], axis=0) # shape (N+q, dim_u)
    y_train = jnp.concatenate([y_train, y_new], axis=0) # shape (N+q, dim_s, P)
    s_train = jnp.concatenate([s_train, s_new], axis=0) # shape (N+q, dim_s)
    w_train = jnp.concatenate([w_train, w_new], axis=0) # shape (N+q, 1)

    new_dataset = (u_train, y_train, s_train, w_train)

    print('New training data shapes')
    print('u: {}'.format(u_train.shape))
    print('y: {}'.format(y_train.shape))
    print('s: {}'.format(s_train.shape))
    print('w: {}\n'.format(w_train.shape))

    # Compute regret
    if config.sequential.log_regret == True:
        variances = vmap(objective)(s_train)
        regret = variances.min()
        log_dict = {'regret': regret}
        wandb.log(log_dict)
        print(f"Regret is {regret}")

    wandb.finish()



    return new_dataset, model