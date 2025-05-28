import jax.numpy as jnp

def c(s,t,M,D,L,tau):
    c1 = M/jnp.sqrt(4*jnp.pi*D*t)*jnp.exp(-s**2/4/D/t)
    c2 = M/jnp.sqrt(4*jnp.pi*D*(t-tau))*jnp.exp(-(s-L)**2/4/D/(t-tau))   
    return jnp.where(t>tau, c1+c2, c1)
s1 = jnp.array([0.0, 1.0, 2.5])
t1 = jnp.array([15.0, 30.0, 45.0, 60.0])
ST = jnp.meshgrid(s1, t1)
STo = jnp.array(ST).T
    
# function mapping the vectorial input x [shape (4,)] to the vectorial output [shape (3,4)] consisting of the concentration evaluation at 3x4 grid points
def f(x):
    res = []
    for i in range(STo.shape[0]):
        resl = []
        for j in range(STo.shape[1]):
            resl.append( c(STo[i,j,0],STo[i,j,1],x[0],x[1],x[2],x[3]) )
        res.append(jnp.array(resl)) 
    return jnp.array(res) # shape (3,4)

# Define ground truth operator
gt_op = lambda x : f(x).reshape((12,1))