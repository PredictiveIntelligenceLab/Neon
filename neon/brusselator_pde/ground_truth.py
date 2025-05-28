import jax.numpy as jnp
from pde import PDE, FieldCollection, ScalarField, UnitGrid
import numpy as onp

# vectorial output space dimension and DeepONet functional evaluation points
N_y = 64 # grid definition

# code adapted from the website
# https://py-pde.readthedocs.io/en/latest/examples_gallery/pde_brusselator_expression.html#sphx-glr-examples-gallery-pde-brusselator-expression-py
def f(x):
    a = x[0]
    b = x[1]
    d0 = x[2]
    d1 = x[3]

    eq = PDE(
        {
            "u": f"{d0} * laplace(u) + {a} - ({b} + 1) * u + u**2 * v",
            "v": f"{d1} * laplace(v) + {b} * u - u**2 * v",
        }
    )

    rng = onp.random.default_rng(10)

    # initialize state
    grid = UnitGrid([N_y, N_y])
    u = ScalarField(grid, a, label="Field $u$")
    v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$", rng=rng)
    #v = b / a + 0.1 * ScalarField.random_normal(grid, label="Field $v$", seed=10)
    state = FieldCollection([u, v])

    sol = eq.solve(state, t_range=20, dt=1e-3)
    
    sol_tensor = []
    sol_tensor.append(sol[0].data)
    sol_tensor.append(sol[1].data)
    sol_tensor = onp.array(sol_tensor)
    
    ss = sol_tensor[onp.isnan(sol_tensor)]
    sol_tensor[onp.isnan(sol_tensor)] = 1e5 * onp.random.randn(*ss.shape)
    #sol_tensor[onp.isnan(sol_tensor)] = 1e5 * rng.standard_normal(ss.shape)
    
    return jnp.transpose(jnp.array(sol_tensor),(1,2,0))



# Define ground truth operator
gt_op = lambda x : f(x).reshape((N_y**2,2))