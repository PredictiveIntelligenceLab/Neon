import jax.numpy as jnp
import numpy as onp

from pyDOE import lhs
import pickle

from ground_truth import gt_op

# vectorial input space dimension and its search space
dim_u = 4
sol_dim = 16
P = 2
lb = -jnp.ones((dim_u,)) # shape (4,)
ub = jnp.ones((dim_u,)) # shape (4,)
bounds = (lb, ub)

# vectorial output space dimension
N_y = 64
xx, yy = jnp.meshgrid(jnp.arange(N_y) / N_y, jnp.arange(N_y) / N_y ) # shapes (N_y,N_y) and (N_y,N_y)
y0 = jnp.concatenate([xx[:, :, None], yy[:, :, None]], axis=-1).reshape((-1, 2)) # shape (N_y**2, 2)
output_dim = (N_y, N_y, 16)
dim_y = 16*N_y**2


def generate_lhs_data(n, seed):
    # Generate training samples
    onp.random.seed(45+seed)
    u = lb + (ub - lb) * lhs(dim_u, n)
    y = jnp.tile(y0, (n,1,1))
    s = jnp.array([gt_op(u) for u in u])
    norms = jnp.linalg.norm(s, axis=(-2,-1))[:,None]
    w = 1./norms**2

    return (u, y, s, w)
      

def import_data(n, seed):
    # seed should be between 0 and 4
    path = 'rpn_bo_data'
    u = jnp.load(path+'/X_loc_'+str(seed)+'.npy')[:n]
    y = jnp.tile(y0, (n,1,1))
    s = jnp.array([gt_op(u) for u in u])
    norms = jnp.linalg.norm(s, axis=(-2,-1))[:,None]
    w = 1./norms**2
    
    return(u, y, s, w)


def main():
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
