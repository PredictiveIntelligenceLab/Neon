import jax.numpy as jnp
import numpy as onp

from pyDOE import lhs
import pickle

from ground_truth import gt_op



N_y = 64 # grid definition
output_dim = (N_y, N_y, 2)
P1 = output_dim[0] # space grid definition
P2 = output_dim[1] # time grid definition
arr_s = jnp.linspace(0, 1, P1) # space 1D grid; shape (N_y,)
arr_t = jnp.linspace(0, 1, P2) # time 1D grid; shape (N_y,)
s_grid, t_grid = jnp.meshgrid(arr_s, arr_t) # meshgrids for space and time; shapes (N_y,N_y) each
y0 = jnp.concatenate([s_grid[:, :, None], t_grid[:, :, None]], axis=-1).reshape((-1, 2)) # values of y; shape (N_y^2, 2)
P = y0.shape[-1] # this should be 2 


lb = jnp.array([0.1, 0.1, 0.01, 0.01]) # lower bounds
ub = jnp.array([5.0, 5.0, 5.0, 5.0]) # upper bounds
dim_u = 4

def generate_data(seed, n=5):
    '''
    seed (int): seed for RNG
    n (int): number of functions in dataset
    '''
    onp.random.seed(seed)
    u = lb + (ub - lb) * lhs(dim_u, n)
    y = jnp.tile(y0, (n,1,1))
    s = jnp.array([gt_op(u) for u in u])
    w = 1./jnp.linalg.norm(s, axis=(-2,-1))[:,None]**2
    return u, y, s, w


def main():
    # Generate training samples
    onp.random.seed(45)
    n_train = 5
    u_train = lb + (ub - lb) * lhs(dim_u, n_train)
    y_train = jnp.tile(y0, (n_train,1,1))
    s_train = jnp.array([gt_op(u) for u in u_train])
    w_train = 1./jnp.linalg.norm(s_train, axis=(-2,-1))[:,None]**2

    print('Initial training data shapes')
    print('u: {}'.format(u_train.shape))
    print('y: {}'.format(y_train.shape))
    print('s: {}'.format(s_train.shape))
    print('w: {}\n'.format(w_train.shape))

    initial_dataset = (u_train, y_train, s_train, w_train)

    file_name = f'initial_examples.npz'
    with open(file_name, 'wb') as f:
            pickle.dump(initial_dataset, f)


    # Generate test samples
    onp.random.seed(49)
    dim_u = 4
    n_test = 512
    u_test = lb + (ub - lb) * lhs(dim_u, n_test)
    y_test = jnp.tile(y0, (n_test,1,1))
    s_test = jnp.array([gt_op(u) for u in u_test])
    w_test = 1./jnp.linalg.norm(s_test, axis=(-2,-1))[:,None]**2

    print('Testing data shapes')
    print('u: {}'.format(u_test.shape))
    print('y: {}'.format(y_test.shape))
    print('s: {}'.format(s_test.shape))
    print('w: {}\n'.format(w_test.shape))

    test_dataset = (u_test, y_test, s_test, w_test)

    file_name = f'test_examples.npz'
    with open(file_name, 'wb') as f:
            pickle.dump(test_dataset, f)

if __name__ == "__main__":
    main()
