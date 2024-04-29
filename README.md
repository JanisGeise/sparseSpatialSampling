# Sparse Spatial Sampling ($S^3$)
Mesh reduction algorithm for CFD post-processing

## TODO
- perform tests on larger cases and datasets -> test current implementation of calculating `n_cells_per_iter`
- complete documentation of code and repository
- use captured variance of the original data as stopping criteria and max. number as cells as 2nd optional stopping criteria
instead of the gradient of the global gain
- disable geometry refinement by default, but leave as option (if captured variance or max. number of cells is
used as stopping criteria, geometry refinement would alter the grid leading to much finer grid than specified,
because it is executed at the very end after the grid generation itself)
- parameter studies for `_stop_thr`, `_cells_per_iter_start`, `_cells_per_iter_end`

## Potential features / Ideas
- automatic determination of reasonable value for min. of `level_bounds` for uniform refinement
- testing different metrics and combinations thereof -> we need a second metric for e.g. sher stress to account
for boundary layers etc.
- generalization for arbitrary geometries and domain boundaries, handle stl files as input

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
