# Sparse Spatial Sampling ($S^3$)
Mesh reduction algorithm for CFD post-processing

## TODO
- further optimization of the code (wrt execution time) -> maybe with cython (depending on support and runtime)
- refine grid near geometry objects (post-processing of the coarse grid prior mapping the original data onto it)
- write unit tests
- perform tests on larger cases and datasets
- complete documentation of code and repository
- provide general interface for export routine -> not only for for OpenFoam, goal:  
  -> only pass field for each time step into export function
  -> export function interpolates the given field onto coarser grid for each time step and writes it into xdmf & HDF5
- determination of parameters (`n_cells_per_iter`, `level_bounds`, metric for computing the gain, ...) -> especially 
`n_cells_per_iter` can decrease runtime significantly

## Potential features / Ideas
- automatic determination of optimal number of cells in sparse grid -> stopping criteria of refinement process
- maybe modify `n_cells_per_iter` during runtime -> at beginning larger (decreases runtime)
- generalize / simplify pre-processing
- parallelization, more usage of numba / jit to improve runtime, especially for larger datasets
- dealing with datasets, which doesn't fit into the RAM at once
- generalization for arbitrary geometries and domain boundaries, handle stl files as input

## References
- Exisiting version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
