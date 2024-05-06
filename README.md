# Sparse Spatial Sampling ($S^3$)
Mesh reduction algorithm for CFD post-processing

## TODO
- test handling of STL files for domain and for 3D cases (currently only 2D is tested and runs without issues)
- complete documentation of repository

## Potential features / Ideas
- automatic determination of reasonable value for min. of `level_bounds` for uniform refinement
- testing different metrics and combinations thereof -> we need a second metric for e.g. sher stress to account
for boundary layers etc.

## Notes
`s_cube.py, line (in __init__ - method, line 119 & 120)`:  
- starting value `n_cells_per_iter` = 1% of original grid size  
`self._cells_per_iter_start = int(0.01 * vertices.size()[0])`


- end value `n_cells_per_iter` = 5% of start value 
`self._cells_per_iter_end = int(0.05 * self._cells_per_iter_start)`

-> if target variance is not met accurately, then the value for `self._cells_per_iter_start` needs to be decreased, e.g.,
by one order of magnitude

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
