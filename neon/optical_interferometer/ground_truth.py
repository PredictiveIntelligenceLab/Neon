import jax.numpy as jnp

from gym_interf import InterfEnv

# vectorial output space dimension and DeepONet functional evaluation points
N_y = 64 # grid definition

# function mapping the vectorial input x to the vectorial output consisting of the 16 images
def f(x):
    gym = InterfEnv()
    gym.reset(actions=(1e-4, 1e-4, 1e-4, 1e-4))
    action = x[:4]
    state = gym.step(action) # state[0] is shape (16,N_y,N_y)
    return jnp.array(state[0].transpose((1,2,0))) # shape (N_y,N_y,16)



# Define ground truth operator
gt_op = lambda x : f(x).reshape((N_y**2,16))