# Sparse Spatial Sampling ($S^3$)
A different version of the existing $S^3$ algorithm (see [references](#References)) for processing large amounts of CFD data. 
The Idea is to create a grid based on a metric provided by the user, e.g., the standard deviation of the pressure fields 
over time. The $S^3$ algorithm then generates a grid which captures *x%* of the metric from the original grid or contains
a specified max. number of cells, depending on the setup given by the user. After generating the grid, the original CFD 
data is interpolated onto the sampled grid and exported to HDF5 & XDMF files.

## TODO & current issues
### TODO
- test handling of STL files for domain and for 3D cases (currently only 2D is tested and runs without issues)
- complete documentation of repository

### Current issues
- delta level constraint is not always fulfilled (if active)
- for interpolated volume solution, there exist cells (mostly near geometries) in which the interpolated values 
don't make sense and don't change wrt time   
-> not sure if this is an issue resulting from the KNN interpolator or
if this is an algorithmic issue
-> this issue is present independently of the delta level constraint for all test cases
-> interpolation error vanishes with increasing mesh size (higher percentage of captured metric) -> most likely an issue 
with the KNN interpolator

## Potential features / Ideas
- testing different metrics and combinations thereof -> we need a second metric for e.g. shear stress to account
for boundary layers etc.

## Notes
`s_cube.py, line (in __init__ - method, line 119 & 120)`:  
- starting value `n_cells_per_iter` = 1% of original grid size  
`self._cells_per_iter_start = int(0.01 * vertices.size()[0])`


- end value `n_cells_per_iter` = 5% of start value 
`self._cells_per_iter_end = int(0.05 * self._cells_per_iter_start)`

-> if target metric is not met accurately, then the value for `self._cells_per_iter_start` needs to be decreased, e.g.,
by one order of magnitude

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
