# Neon: Neural Epistemic Operator Networks
Repository for the paper "Composite Bayesian Optimization In Function Spaces Using NEON -- Neural Epistemic Operator Networks", available at https://www.nature.com/articles/s41598-024-79621-7.

![neon_diagrams](https://github.com/user-attachments/assets/823c9bb2-0a96-4b26-bc03-451094b64188)


## Paper Summary

This work introduces NEON (Neural Epistemic Operator Networks), a novel architecture for uncertainty-aware operator learning that enables epistemic uncertainty quantification using a single model, without relying on computationally expensive ensembles. NEON builds on the Epistemic Neural Network (ENN) framework, augmenting a deterministic neural operator with a lightweight EpiNet module to generate diverse predictions conditioned on a latent epistemic index. This design dramatically reduces the number of trainable parameters—by up to two orders of magnitude—while retaining or exceeding the performance of deep ensembles.

To improve the stability and effectiveness of Bayesian Optimization with NEON, we also introduce Leaky Expected Improvement (L-EI)—a novel acquisition function that addresses gradient vanishing issues commonly encountered with traditional Expected Improvement (EI). By replacing the standard ReLU with a LeakyReLU, L-EI ensures meaningful gradients even in regions where the model underpredicts the objective, facilitating more efficient and reliable optimization in practice. We show that L-EI retains roughly the same optima as EI while being significantly easier to optimize, particularly in high-dimensional and non-convex settings typical of operator learning tasks.

We apply NEON to composite Bayesian Optimization (BO) problems, where the target function has the structure $f=g\circ h$, with $h$ being a costly-to-evaluate operator and $g$ a known, cheap-to-compute functional. Across a suite of synthetic and real-world benchmarks—including PDEs, inverse problems, and wireless network optimization—NEON consistently outperforms deep ensembles and Gaussian Processes baselines from literature. Our results demonstrate NEON’s ability to combine scalability, accuracy, and calibrated uncertainty—making it an effective surrogate model for high-dimensional and function-valued optimization tasks.

## Code Summary

This code requires common libraries of the JAX environment, such as Flax (for neural network design) and Optax (for training and optimization). Plotting is done using Matplotlb.

Inside the `neon` folder, you will find the main files used for running the experiments in our paper. Each experiment is compartimentalized into sub folders (`pollutants`, `brusselator_pde`, `optical_interferometer` and `cell_towers`). Running the Optical Interferometer problem requires the Interferbot package (https://github.com/dmitrySorokin/interferobotProject), and running the Cell Towers problem requires the CCO-in-ORAN package (https://github.com/Ryandry1st/CCO-in-ORAN).

## Citation

If you would like to cite this work, please use the BibTeX entry below:

```
@article{guilhoto2024neon,
  title={Composite bayesian optimization in function spaces ising NEON—Neural Epistemic Operator Networks},
  author={Guilhoto, Leonardo Ferreira and Perdikaris, Paris},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={29199},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
