# Sparse Spatial Sampling ($S^3$)
Mesh reduction algorithm for CFD post-processing

## TODO
- generalization for arbitrary geometries and domain boundaries
- post-processing of the sparse grid -> fitting to original domain boundaries & geometry
- optimization of the code (wrt execution time), especially the removal / checking for invalid cells can be
parallelized etc.
- write unit tests
- perform tests on larger cases and datasets
- complete documentation of code and repository
- determination of parameters (`n_cells_per_iter`, `level_bounds`, metric for computing the gain, ...)

## Potential features / Ideas
- automatic determination of optimal number of cells in sparse grid -> stopping criteria of refinement process
- maybe modify `n_cells_per_iter` during runtime -> at beginning larger (decreases runtime)
- generalize / simplify pre-processing
- parallelization, more usage of numba / jit to improve runtime, especially for larger datasets
- dealing with datasets, which doesn't fit into the RAM at once

## References
Idea & 1D implementation taken from [Andre Weiner](https://github.com/AndreWeiner)
