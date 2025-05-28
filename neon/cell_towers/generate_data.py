import jax.numpy as jnp
import numpy as onp

from pyDOE import lhs
import pickle

from ground_truth import gt_op, bounds, N_y

# vectorial input space dimension and its search space
dim_u = 30 # dimension of input u
soln_dim = 2 # dimension of solution at each point s
P = 2 # dimension of each query point y
lb, ub = bounds # unpack lower and upper bounds

# vectorial output space dimension
xx, yy = jnp.meshgrid(jnp.arange(N_y), jnp.arange(N_y)) # shapes (N_y,N_y) and (N_y,N_y)
y0 = jnp.concatenate([xx[:, :, None], yy[:, :, None]], axis=-1).reshape((-1, 2)) # shape (N_y**2, 2)
output_dim = (N_y, N_y, soln_dim)
dim_y = soln_dim*N_y**2


def generate_lhs_data(n, seed):
    # Generate training samples
    onp.random.seed(45+seed)
    u = lb + (ub - lb) * lhs(dim_u, n)
    y = jnp.tile(y0, (n,1,1))
    s = jnp.array([gt_op(u) for u in u]) # shape (n, N_y**2, soln_dim)
    norms = jnp.linalg.norm(s, axis=(-2,-1))[:,None]
    w = 1./norms**2

    return (u, y, s, w)
      

def main():

    '''
    # Generate training samples
    n_train = 15
    initial_dataset = generate_lhs_data(n_train, 0)
    u_train, y_train, s_train, w_train = initial_dataset
    
    print('Initial training data shapes')
    print('u: {}'.format(u_train.shape))
    print('y: {}'.format(y_train.shape))
    print('s: {}'.format(s_train.shape))
    print('w: {}\n'.format(w_train.shape))

    file_name = f'initial_examples.npz'
    with open(file_name, 'wb') as f:
            pickle.dump(initial_dataset, f)
    '''


    # Generate test samples
    n_test = 512
    test_dataset = generate_lhs_data(n_test, 12345)
    u_test, y_test, s_test, w_test = test_dataset

    print('Testing data shapes')
    print('u: {}'.format(u_test.shape))
    print('y: {}'.format(y_test.shape))
    print('s: {}'.format(s_test.shape))
    print('w: {}\n'.format(w_test.shape))    

    file_name = f'test_examples.npz'
    with open(file_name, 'wb') as f:
            pickle.dump(test_dataset, f)

if __name__ == "__main__":
    main()
