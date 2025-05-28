import jax.numpy as jnp
from flax.linen import sigmoid
import numpy as onp
import torch

import glob
import json

import sys
sys.path.append("CCO-in-ORAN/cco_standalone_icassp_2021/") # Path to simulation solver
from simulated_rsrp import SimulatedRSRP

N_y = 50 # grid resolution for power and interference maps

min_Tx_power_dBm, max_Tx_power_dBm = 30, 50
lower_bounds = jnp.array([0.0] * 15 + [min_Tx_power_dBm] * 15)
upper_bounds = jnp.array([10.0] * 15 + [max_Tx_power_dBm] * 15)
bounds = (lower_bounds, upper_bounds)


powermaps = glob.glob("CCO-in-ORAN/cco_standalone_icassp_2021/data/power_maps/powermatrixDT*.json")
all_pmap_dicts = []
for pmap_loc in powermaps:
    with open(pmap_loc, "r") as f:
        all_pmap_dicts.append(json.load(f))

downtilts_maps = {}
for i in range(11):
    downtilts_maps[float(i)] = SimulatedRSRP.build_single_powermap(
        all_pmap_dicts[i]
    )

simulated_rsrp = SimulatedRSRP(
    downtilts_maps=downtilts_maps,
    min_TX_power_dBm=min_Tx_power_dBm,
    max_TX_power_dBm=max_Tx_power_dBm,
)


# function mapping the vectorial input theta to the vectorial output consisting of the 2 images
def simulate(theta, return_high_res=False):
    downtilts = theta[:15].astype(int)
    tx_pwrs = theta[15:]
    (
        rsrp_powermap,
        interference_powermap,
        _,
    ) = simulated_rsrp.get_RSRP_and_interference_powermap((downtilts, tx_pwrs))
    # return torch.stack(
    #     (torch.tensor(rsrp_powermap), torch.tensor(interference_powermap))
    # )
    highd_res = torch.stack(
        (torch.tensor(rsrp_powermap), torch.tensor(interference_powermap))
    )

    if return_high_res: # returns high resulution images
        highd_res = jnp.array(highd_res) # shape (2, 241, 241)
        return highd_res.transpose((1,2,0)) # shape (241, 241, 2)
    else: # downsamples to 50x50 images
        out = torch.nn.functional.interpolate(highd_res.unsqueeze(0), (50, 50))[0] # shape (2,50,50)
        return jnp.array(out).transpose((1,2,0)) # shape (50,50,2)



# Define ground truth operator
gt_op = lambda theta : simulate(onp.array(theta)).reshape((N_y**2,2))



def construct_both_objectives(samples):
    rsrp_map = samples[..., 0, :, :]
    interference_map = samples[..., 1, :, :]

    weak_coverage_threshold = -80.0
    over_coverage_threshold = 6.0
    f_weak_coverage = sigmoid(weak_coverage_threshold - rsrp_map).sum((-1, -2))
    #size = onp.prod(rsrp_map.shape[-2:])

    # over_coverage_area = (rsrp_map >= weak_coverage_threshold) & (
    #    interference_map + over_coverage_threshold > rsrp_map
    # )
    rsrp_gt_threshold = sigmoid(rsrp_map - weak_coverage_threshold)
    if_gt_threshold = sigmoid(
        (interference_map + over_coverage_threshold) - rsrp_map
    )
    over_coverage_area = rsrp_gt_threshold * if_gt_threshold

    over_coverage_map = (
        interference_map * over_coverage_area
        + over_coverage_threshold
        - rsrp_map * over_coverage_area
    )
    # over_coverage_map = (
    #     interference_map[over_coverage_area]
    #     + over_coverage_threshold
    #     - rsrp_map[over_coverage_area]
    # )
    g_weak_coverage = sigmoid(over_coverage_map).sum((-1, -2))
    return f_weak_coverage, g_weak_coverage

# this is a scalarization for now
def objective_fn(samples, return_both=False, weight=0.25): # samples is shape (N_y**2,2)
    #N_y = int(round(jnp.sqrt(samples.shape[0])))
    N_y = 50
    samples = samples.reshape((N_y, N_y, 2)) # shape (N_y,N_y,2)
    samples = samples.transpose((2,0,1)) # shape (2,N_y,N_y)
    f_weak_coverage, g_weak_coverage = construct_both_objectives(samples)
    if return_both:
        return  f_weak_coverage, g_weak_coverage
    else:
        return -(weight * f_weak_coverage + (1.0 - weight) * g_weak_coverage)