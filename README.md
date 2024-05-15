# Sparse Spatial Sampling ($S^3$)
A different version of the existing $S^3$ algorithm (see [references](#References)) for processing large amounts of CFD data. 
The Idea is to create a grid based on a metric provided by the user, e.g., the standard deviation of the pressure fields 
over time. The $S^3$ algorithm then generates a grid which captures *x%* of the metric from the original grid or contains
a specified max. number of cells, depending on the setup given by the user. After generating the grid, the original CFD 
data is interpolated onto the sampled grid and exported to HDF5 & XDMF files.

## TODO & current issues
### TODO
- write unit tests for relevant methods and functions
- if possible: try to use pyVista for 2D STL file handling as well
- complete documentation of repository

### Current issues
- delta level constraint is not always fulfilled (if active)

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

### Cell centered solution near geometries
- for interpolated cell solution at the cell center it may happen that the interpolation is slightly off near geometries, 
especially if the grid is coarse. These errors are typically small and constant over time, but it is best to apply the
`Points to Cell` filter in Paraview to compute an improved version of the cell centered solution if this is an issue.

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
