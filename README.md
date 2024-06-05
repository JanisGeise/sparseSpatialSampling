# Sparse Spatial Sampling ($S^3$)
A different version of the existing $S^3$ algorithm (see [references](#References)) for processing large amounts of CFD data. 
The Idea is to create a grid based on a metric provided by the user, e.g., the standard deviation of the pressure fields 
over time. The $S^3$ algorithm then generates a grid which captures *x%* of the metric from the original grid or contains
a specified max. number of cells, depending on the setup given by the user. After generating the grid, the original CFD 
data is interpolated onto the sampled grid and exported to HDF5 & XDMF files.

## Getting started

### Overview
The repository contains the following directories:

1. `s_cube`: implementation of the *sparseSpatialSampling* algorithm
2. `tests`: unit tests
3. `examples`: example scripts for executing $S^3$ for different test cases
4. `post_processing`: scripts for analysis and visualization of the results

For executing $S^3$, the following things have to be provided:

1. the simulation data as point cloud and main dimensions of the numerical domain
2. a metric for each point 
3. geometries within the domain (optional)

The general workflow will be explained in more detail below.

### 1. Providing the original CFD data
- the coordinates of the original grid have to be provided as tensor with a shape of `[N_cells, N_dimensions]`
- for CFD data generated with OpenFoam, e.g., [flowtorch](https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/index.html) 
can be used for loading the cell centers

### 2. Computing a metric
- a metric for each cell has to be computed, the itself metric depends on the goal. For example, to capture variances over time,
the standard deviation of the velocity field with respect to time can be used as a metric (refer to examples in the 
`examples` directory).
- the metric has to be a 1D tensor in the shape of `[N_cells, ]`

### 3. Providing the numerical domain and geometries

#### Domain:
- the main dimensions of the numerical domain must be provided as dict
- for more information it is referred to `s_cube/execute_grid_generation.py` and `s_cube/geometry.py`
- an example input for a 2D box may look like 


    `domain = {"name": "example domain", "bounds": [[xmin, ymin], [xmax, ymax]], "type": "cube", "is_geometry": False}`

**Note:** in case no more geometries are present, the dict for the domain has to be wrapped into a list prior passing it to 
the `execute_grid_generation` function, since this functions expects a list of geometries

#### More geometries:
- there can be added as many further geometries as required to avoid generating a grid in these areas
- for simple geometries (cube, rectangle, sphere or circle), only the main dimensions and positions need to be passed
- if coordinates of the geometries should be used, they have to be provided either as an enclosed area (for 2D case) or 
as STL file (for 3D case)

For more information on the required format of the input dicts it is referred to `s_cube/execute_grid_generation.py` 
and `s_cube/geometry.py` or the provided `examples`.

### Interpolation of the original CFD data
- once the grid is generated, the original fields from CFD can be interpolated onto this grid by calling the `fit_data` 
method of the `DataWriter` instance
- therefore, each field that should be interpolated has to be provided as tensor with the size `[n_cells, n_dimensions, n_snapshots]`.
- a scalar field has to be of the size `[n_cells, 1, n_snapshots]`
- a vector field has to be of the size `[n_cells, n_entries, n_snapshots]`
- the snapshots can either be passed into `fit_data` method all at once, in batches, or each snapshot separately
depending on the size of the dataset and available RAM (refer to section memory requirements). 
An example for exporting the fields snapshot-by-snapshot or in batches can be found in `examples/s3_for_surfaceMountedCube_large.py`

### Results & output files
- once the original fields are interpolated onto the new grid, they can be saved to a HDMF file calling the 
`export_data()` method of the `DataWriter` class
- for data from `OpenFoam`, the function `export_openfoam_fields` in `execute_grid_generation` can be used to either 
export all snapshots at once or snapshot-by-snapshot
- the data is saved as temporal grid structure in an HDMF & XDMF file for analysis, e.g., in ParaView
- one HDMF & XDMF file is created for each field
- additionally, a summary of the refinement process and mesh characteristics is stored as property in the `DataWriter`
instance called `mesh_info`, which can be saved with `pt.save(...)`

## General notes
### Memory requirements
The RAM needs to be large enough to hold at least:
- a single snapshot of the original grid
- the original grid
- the interpolated grid (size depends on the specified target metric)
- the levels of the interpolated grid (size depends on the specified target metric)
- a snapshot of the interpolated field (size depends on the specified target metric)

The required memory can be estimated based on the original grid and the target metric. Consider the example of a single 
snapshot having a size of 30 MB and the original grid of 10 MB. The target metric is set to *75%*, leading to an 
approximate size of *7.5* MB for the generated grid and cell levels, and *22.5* MB for a single snapshot of the 
interpolated field. Consequently, interpolation and export of a single snapshot requires at least *~80* MB of additional RAM.

### Reaching the specified target metric
- if the target metric is not reached with sufficient accuracy, the parameter `n_cells_iter_start` and
`n_cells_iter_end` have to be decreased. If none provided, they are automatically set to:  
  
&emsp;&emsp; `n_cells_iter_start` = 1% of original grid size  
&emsp;&emsp; `n_cells_iter_end` = 5% of `n_cells_iter_start` 


- the refinement of the grid near geometries requires approximately the same amount of time as the adaptive refinement, 
so unless a high resolution of geometries is required, it is recommended to set `_refine_geometry = False`

### Projection of the coordinates for 2D in x-y-plane
- for 2D cases, the coordinates of the generated grid are always exported in the *x-y-*plane, independently of the 
orientation of the original CFD data 

## Unit tests
- tests can be executed with `pytest` inside the `tests` directory
- `pytest` can be installed via: `pip install pytest`

## Issues
If you have any questions or something is not working as expected, fell free to open up a new 
[issue](https://github.com/JanisGeise/sparseSpatialSampling/issues).

### Known issues
**Visibility of internal nodes in ParaView**
- for 3D cases, the internal nodes are not or only partially displayed in Paraview, although they are present in the 
HDF5 file
- the fields and all points are present, each node and each center has a value, which is displayed correctly

This seems to be a rendering issue in Paraview resulting from the sorting of the nodes. However, this issue
should not be affecting any computations or operations done in ParaView or with the interpolated data in general.

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weine, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
