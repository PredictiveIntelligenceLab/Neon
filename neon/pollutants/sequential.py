import jax.numpy as jnp
from jax import vmap, random, jit

import wandb
from tqdm.auto import trange

import time
import sys
sys.path.append('..') # makes modules in parent repository available to import

from models import OperatorRegression
from acquisition import MCAcquisition
from utils import DataGenerator

from ground_truth import gt_op

N = 5 # number of initial examples
dim_u = 4 # grid definition of intput functions
dim_s = 12 # grid definition of output functions
lb = jnp.array([7.0, 0.02, 0.01, 30.01]) # lower bounds
ub = jnp.array([13.0, 0.12, 3.0, 30.295]) # upper bounds
bounds = (lb, ub) # bounds of sampled functions

# True parameters
true_x = jnp.array([10.0, 0.07, 1.505, 30.1525])
true_s = gt_op(true_x)

s1 = jnp.array([0.0, 1.0, 2.5]) # positions
t1 = jnp.array([15.0, 30.0, 45.0, 60.0]) # times
T, X = jnp.meshgrid(t1, s1) # shapes (3,4) and (3,4)
y0 = jnp.hstack([T.flatten()[:,None], X.flatten()[:,None]]) # shape (12,2)
P = y0.shape[-1] # this should be 2 


@jit
def re_organize(us, ys, ws=None, ss=None):
    '''
    Inputs:
        us; shape (..., N, m_in)
        ys; shape (..., N, m_out, P)
        ws; shape (..., N, 1) [optional]
        ss; shape (..., N, m_out) [optional]
    Outputs
        us; shape (..., N*m_out, m_in)
        ys; shape (..., N*m_out, 1)
        ws; shape (..., N*m_out, 1)
        ss; shape (..., N*m_out, 1)
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
        assert (ss.shape[-2] == N) and (ss.shape[-1] == m_out), 'Wrong dimensions'
        ss = ss.reshape(ss.shape[:-2] + (N*m_out, 1)) # shape (..., N*m_out, 1)

    return tuple(filter(lambda x : x is not None,
                        (us, ys, ws, ss)))


def create_data(u, y, op=gt_op):
    '''Generates a single datapoint for the pollutants problem.

    Inputs:
        u; shape (m,)
        y; shape (m,1)
    '''
    P = y.shape[0]
    s = op(u) # forward pass of ground truth operator; shape (m,1)
    w = 1.0/jnp.linalg.norm(s, 2)**2 # float
    return u, y, s[:,0], w[...,None] # shapes (m,) & (m,) & (m,) (m,1) & (1,)

def sample_pairs(rng_key, y, n_eig, op=gt_op):
    '''Generates several input output pairs of the ground truth operator.

    Inputs:
        y; shape (P,1)
        op: function that takes in (u,y) and returns s. Signature ( (b_size, m), (b_size,1) ) -> (b_size, 1)
    '''
    P = y.shape[0]
    # Sample a random u
    u = random.uniform(rng_key, shape=(dim_u,), minval=lb, maxval=ub ) # shape (dim_x,)
    # compute data given u
    u, y, s, w = create_data(u, y, op=op)
    return u, y, s, w # shapes (m,m) & (m,1) & (m,1) (m,1) & (m,1)


# Generate training samples
key = random.PRNGKey(1) # seed for training samples
gen_fn = lambda key: sample_pairs(key, y0, gt_op)
def get_initial_dataset(key=key, N=5, verbose=True):
    keys = random.split(key, N)
    u_train, y_train, s_train, w_train = vmap(gen_fn)(keys)
    if verbose:
        print('Initial training data shapes')
        print('u: {}'.format(u_train.shape))
        print('y: {}'.format(y_train.shape))
        print('s: {}'.format(s_train.shape))
        print('w: {}\n'.format(w_train.shape))
    return u_train, y_train, s_train, w_train

# Generate testing samples
N = 512
key = random.PRNGKey(6) # seed for testing samples
keys = random.split(key, N)
u_test, y_test, s_test, w_test = vmap(gen_fn)(keys)
print('Testing data shapes')
print('u: {}'.format(u_test.shape))
print('y: {}'.format(y_test.shape))
print('s: {}'.format(s_test.shape))
print('w: {}'.format(w_test.shape))

# Defines a batch consisting of all test data
TEST_BATCH = (u_test, y_test, s_test, w_test)

def compute_relative_error(config, model, batch=TEST_BATCH, norm_stats=(0.,1.,0.,1.,0.,1.)):
    '''
    Average L2 error over all test functions
    '''
    key = random.PRNGKey(config.seed)
    u, y, s, w = batch # unpack batch
    mu_u, sig_u, mu_y, sig_y, mu_s, sig_s = norm_stats # unpack normalization statistics
    u = (u-mu_u)/sig_u
    y = (y-mu_y)/sig_y
    N = u.shape[0] # number of functions
    u_aux, y_aux, s_aux, w_aux = re_organize(u, y, w, s)
    z_test = random.normal(key, (config.ensemble_size, config.ensemble_size))
    pred_fn = lambda params : model.state.apply_fn(params, u_aux, y_aux, z_test)
    s_pred = vmap(pred_fn)(model.state.params)[0,...] # shape (ensamble_size, N*m, 1)
    s_mean = s_pred.mean(axis=0) # average prediction; shape (N*m,1)
    s_mean = s_mean.reshape((N,-1))
    s_mean = (s_mean*sig_s)+mu_s
    error = (s_mean - s)**2 # point-wise absolute L2 error; shape (N, m)
    error = error * (w) # point-wise relative error; shape (N,m)
    error = jnp.sqrt(error.sum(axis=-1)) # vector of relative errors; shape (N,)

    return error.mean() # scalar

def eval_step(config, model, batch):

    params = model.state.params
    log_dict = {}

    if config.logging.log_losses:
        loss = model.eval_loss(params, batch)
        loss = loss.mean()
        log_dict['loss'] = loss

    return log_dict

def train_and_collect(config, dataset, iteration, model):
    # wandb stuff
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project,
               name=wandb_config.name,
               config=dict(config),
               group='trial%d_it%d' % (config.seed, iteration))
    # Define the custom x axis and regret metric
    wandb.define_metric('iteration')
    wandb.define_metric('regret', step_metric='iteration')


    # unpacking dataset
    u_train, y_train, s_train, w_train = dataset
    print('Training data')
    print('u: {}'.format(u_train.shape))
    print('y: {}'.format(y_train.shape))
    print('s: {}'.format(s_train.shape))
    print('w: {}'.format(w_train.shape))

    mu_u = u_train.mean(0)
    sig_u = u_train.std(0)
    mu_y = y0.mean(0)
    sig_y = y0.std(0)
    mu_s = s_train.mean(0)
    sig_s = s_train.std(0)
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
                                min(config.training.batch_size, len(u_train)))
    data = iter(data_loader)
    loss_log = []

    # Training
    num_steps = int(config.training.max_steps*(1 + iteration / config.sequential.num_iters))
    pbar = trange(num_steps)
    t_start = time.time()
    for step in pbar:
        # Acquiring + organizing batch
        batch = next(data)

        model.state = model.step(model.state, batch)
        # logging
        if step % config.logging.log_every_steps == 0:
            log_dict = eval_step(config, model, batch)
            wandb.log(log_dict, step)
            pbar.set_postfix({'loss' : log_dict['loss']})
    t_end = time.time()
    wandb.log({'train_time':t_end-t_start})


    
    # Compute test error
    if config.sequential.log_regret == True:
        test_er = compute_relative_error(config, model, norm_stats=norm_stats)
        train_er = compute_relative_error(config, model, batch=dataset, norm_stats=norm_stats)
        log_dict = {'test_er': test_er, 'train_er': train_er, 'iteration': iteration}
        wandb.log(log_dict)
        print(f"Relative L2 train error is {train_er : .3%}")
        print(f"Relative L2 test error is {test_er : .3%}")


    @vmap
    def posterior(x, num_samples=128, key=random.PRNGKey(0)):
        x = (x-mu_u)/sig_u
        x = jnp.tile(x, (dim_s,1)) # shape (dim_s,dim_u)
        y = (y_train[0,...]-mu_y)/sig_y
        samples = model.posterior_samples(key, x, y, num_samples) # shape (num_samples, m, 1)
        samples = (samples*sig_s[:,None])+mu_s[:,None]
        return ((true_s-samples)**2).mean(-2) # shape (num_samples, 1)


    # Acquisition
    if config.sequential.acquisition == 'LCB':
        args = (config.sequential.kappa,)
    elif config.sequential.acquisition == 'EI':
        errors = ((true_s.squeeze()-s_train)**2).mean(-1)
        best = errors.min()
        args = (best,)
    elif config.sequential.acquisition == 'EI_leaky':
        errors = ((true_s.squeeze()-s_train)**2).mean(-1)
        best = errors.min()
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
        objs = ((true_s.T-s_train)**2).mean(-1)
        cutoff = min(config.sequential.required_init_pts, len(u_train))
        idx = jnp.argsort(objs)[:cutoff]
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
    u_new, y_new, s_new, w_new = vmap(create_data, in_axes=(0,None))(new_coeffs, y_train[0,])

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
        errors = ((true_s.squeeze()-s_train)**2).mean(-1)
        regret = errors.min()
        log_dict = {'regret': regret}
        wandb.log(log_dict)
        print(f"Regret is {regret}")

    wandb.finish()

    return new_dataset