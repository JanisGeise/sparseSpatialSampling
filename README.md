# Sparse Spatial Sampling ($S^3$)
Mesh reduction algorithm for CFD post-processing

## TODO
- further optimization of the code (wrt execution time) -> maybe with cython (depending on support and runtime)
-> computation of cell faces still requires long time (although already in numba) -> more efficient algorithm?
- write unit tests
- perform tests on larger cases and datasets -> test current implementation of calculating `n_cells_per_iter`
- complete documentation of code and repository
- provide general interface for export routine -> not only for for OpenFoam, goal:  
  -> only pass field for each time step into export function  
  -> export function interpolates the given field onto coarser grid for each time step and writes it into xdmf & HDF5
- parameter studies for `_stop_thr`, `_cells_per_iter_start`, `_cells_per_iter_end`

## Potential features / Ideas
- generalize / simplify pre-processing
- parallelization, more usage of numba / jit to improve runtime, especially for larger datasets
- automatic determination of reasonable value for min. of `level_bounds`
- dealing with datasets, which doesn't fit into the RAM at once
- generalization for arbitrary geometries and domain boundaries, handle stl files as input

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
