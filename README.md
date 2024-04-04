# Sparse Spatial Sampling ($S^3$)
Mesh reduction algorithm for CFD post-processing

## TODO
- post-processing of the sparse grid -> fitting to original domain boundaries & geometry
- further optimization of the code (wrt execution time)
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
- generalization for arbitrary geometries and domain boundaries, handle stl files as input

## References
Idea & 1D implementation taken from [Andre Weiner](https://github.com/AndreWeiner)
