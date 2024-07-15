# Sparse Spatial Sampling ($S^3$)
A different version of the existing $S^3$ algorithm (see [references](#References)) for processing large amounts of CFD data. 
The idea is to create a grid based on a metric provided by the user, e.g., the standard deviation of the pressure fields 
over time. The $S^3$ algorithm then generates a grid which captures *x%* of the metric from the original grid or contains
a specified max. number of cells, depending on the setup given by the user. After generating the grid, the original CFD 
data is interpolated onto the sampled grid and exported to HDF5 & XDMF files.

## Getting started

**Note: the code is currently refactored (in progress) to improve its usability.** 
The documentation as well as the code basis will be updated once the refactoring is completed.

### Overview
The repository contains the following directories:

1. `s_cube`: implementation of the *sparseSpatialSampling* algorithm
2. `tests`: unit tests
3. `examples`: example scripts for executing $S^3$ for different test cases
4. `post_processing`: scripts for analysis and visualization of the results

For executing $S^3$, the following things have to be provided:

1. the simulation data as point cloud
2. either the main dimensions of the numerical domain or the numerical domain as STL file (3D) or coordinates forming 
an enclosed area (2D)
3. a metric for each point 
4. geometries within the domain (optional)

The general workflow will be explained more detailed below. Currently, $S^3$ can't handle decomposed CFD data as input.

### 1. Providing the original CFD data
- the coordinates of the original grid have to be provided as tensor with a shape of `[N_cells, N_dimensions]`
- for CFD data generated with OpenFoam, e.g., [flowtorch](https://flowmodelingcontrol.github.io/flowtorch-docs/1.2/index.html) 
can be used for loading the cell centers

### 2. Computing a metric
- a metric for each cell has to be computed, the metric itself depends on the goal. For example, to capture variances over time,
the standard deviation of the velocity field with respect to time can be used as a metric (refer to examples in the 
`examples` directory).
- the metric has to be a 1D tensor in the shape of `[N_cells, ]`

### 3. Providing the numerical domain and geometries

The $S^3$ package currently provides the following shapes representing numerical domains or geometries:
1. `CubeGeometry`: rectangles (2D) or cubes (3D)
2. `SphereGeometry`: circles (2D) or spheres (3D)
3. `GeometryCoordinates2D`: arbitrary 2D geometries, the coordinates must be provided as an enclosed area
4. `GeometrySTL3D`: arbitrary 3D geometries, an STL file with a manifold and closed surface must be provided

These geometry classes are located in `s_cube.geometry`.

#### Providing the numerical domain:
- exactly one geometry object needs to be declared as domain, which can be done by passing `keep_inside=True` to the 
geometry object indicating that the points inside the object should be kept as grid
- the domain can be represented by any of the available geometry object classes

#### Adding geometries:
- there can be added as many geometries as required to avoid generating a grid in areas where a geometry in 
the CFD simulation is present
- for all geometries, which are not domains, `keep_inside = False` has to be set indicating that there shouldn't be cells
generated inside these objects

For more information on the required format of the input dicts it is referred to `s_cube.geometry` or the
provided `examples`.

#### Putting it all together
Once the numerical domain and optional geometries are defined, we can execute $S^3$. An example input may look like:

    from s_cube.sparse_spatial_sampling import SparseSpatialSampling
    from s_cube.geometry import CubeGeometry, SphereGeometry, GeometrySTL3D

    # 2D box as numerical domain
    domain_2d = CubeGeometry(name="domain", keep_inside=True, lower_bound=[0, 0], upper_bound=[2.2, 0.41])

    # 3D box as numerical domain
    domain_3d = CubeGeometry(name="domain", keep_inside=True, lower_bound=[0, 0, 0], upper_bound=[14.5, 9, 2])
    
    # alternatively, if the domain is provided as STL file for the 3D box, we can use the GeometrySTL3D class as well
    domain_3d = GeometrySTL3D("cube", False, join("..", "tests", "cube.stl"))

    # if we have geometries inside the domain, we can add the same way as we did for the domain. the keyword `refine`
    # indicates that we want to refine the mesh near the geometry after it is creates for a better resolution of the 
    # geometry (by default this is the max. refinement level present at the geometry)
    geometry = SphereGeometry("cylinder", False, position=[0.2, 0.2], radius=0.05, refine=True)

    # alternatively, we could also define a min. refinement level with which we want to resolve the geometry. In case we
    # set a min_refinement_level but we keep refine=False then refine is automatically set to True
    geometry = SphereGeometry("cylinder", False, position=[0.2, 0.2], radius=0.05, refine=True, min_refinement_level=6)

    # analogously, we can define a geometry for the 3D case
    domain_3d = CubeGeometry(name="cube", keep_inside=False, lower_bound=[3.5, 4, -1], upper_bound=[4.5, 5, 1])

    # create a S^3 instance, the coordinates are the corrdinates of the cell centers in the original grid while metric 
    # is the metric based on which the grid is created
    s_cube = SparseSpatialSampling(coordinates, metric, [domain_*d, geometry_*d], save_path, save_name, grid_name, 
                                   min_metric=min_metric)

    # execute S^3 to generate a grid bassed on the given metric
    export = s_cube.execute_grid_generation()

### Interpolation of the original CFD data
- once the grid is generated, the original fields from CFD can be interpolated onto this grid by calling the `fit_data` 
method of the `DataWriter` instance
- therefore, each field that should be interpolated has to be provided as tensor with the size `[n_cells, n_dimensions, n_snapshots]`.
- a scalar field has to be of the size `[n_cells, 1, n_snapshots]`
- a vector field has to be of the size `[n_cells, n_entries, n_snapshots]`
- the snapshots can either be passed into `fit_data` method all at once, in batches, or each snapshot separately
depending on the size of the dataset and available RAM (refer to section memory requirements). 

example for interpolating and exporting a field:

    # times are the time steps of the simulation, need to be either a str or a list[str]
    export.times = times
    export.export_data(cooridnates, snapshots_original_field, field_name)


An example for exporting the fields snapshot-by-snapshot or in batches can be found in 
`examples/s3_for_surfaceMountedCube_large.py` (for large datasets, which are not fitting into the RAM all at once).

### Results & output files
- once the original fields are interpolated onto the new grid, they can be saved to a HDMF file calling the 
`export_data()` method of the `DataWriter` class
- for data from `OpenFoam`, the function `export_openfoam_fields` in `execute_grid_generation` can be used to either 
export all snapshots for a given list of fields at once or snapshot-by-snapshot (more convenient)
- the data is saved as temporal grid structure in an HDMF & XDMF file for analysis, e.g., in ParaView
- one HDMF & XDMF file is created for each field
- additionally, a `mesh_info` file containing a summary of the refinement process and mesh characteristics is saved
- once the grid generation is completed, the `DataWriter`instance is saved. This avoids the necessity to execute the 
grid generation again in case additional fields should be interpolated afterward

### Executing $S^3$
#### Local machine
For executing $S^3$, it is recommended to create a virtual environment. Otherwise, it need to be ensured that the Numpy 
version is $>= 1.22$ (requirement for numba).

    # install venv
    sudo apt update && sudo apt install python3.8-venv

    # clone the S^3 repository 
    git clone https://github.com/JanisGeise/sparseSpatialSampling.git

    # create a virtual environment inside the repository
    python3 -m venv s_cube_venv

    # activate the environment and install all dependencies
    source s_cube_venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt

    # once everything is installed, leave the environment
    deactivate

For executing the example scripts in `examples/`, the CFD data must be provided. Further the paths to the data as well
as the setup needs to be adjusted accordingly. A script can then be executed as
    
    # start the virtual environment
    source s_cube_venv/bin/activate

    # add the path to the repository
    . source_path

    # execute a script
    cd examples/
    python3 s3_for_cylinder2D.py

#### HPC
The setup for executing $S^3$ on an HPC cluster is the same as for the local machine. 
An example jobscript for executing $S^3$ on the *surfaceMountedCube* simulation may look like:

    #!/bin/bash
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=72
    #SBATCH --time=08:00:00
    #SBATCH --job-name=s_cube
    
    # load python
    module load release/23.04  GCCcore/10.2.0
    module load Python/3.8.6
    
    # activate venv
    source s_cube_venv/bin/activate
    
    # add the path to s_cube
    . source_path
    
    # path to the python script
    cd examples/
    
    python3 s3_for_surfaceMountedCube_large_hpc.py &> "log.main"

An [example jobscript](https://github.com/JanisGeise/sparseSpatialSampling/blob/main/example_jobscript) for the
[Barnard](https://compendium.hpc.tu-dresden.de/jobs_and_resources/barnard/) HPC of TU Dresden is provided.

### Performing an SVD
Once the grid is generated and a field is interpolated, an SVD from this field can be computed:

    # instantiate DataLoader
    loader = DataLoader()

    # load the data matrix from the HDF5 file
    loader.load_data(path_to_the_hdf5_file, file_name, field_name)

    # compute the SVD
    loader.compute_svd()

    # write the results of the SVD to HDF5 & XDMF
    loader.write_data(save_path, save_name)

The modes, singular values and mode coefficients are saved in an extra HDF5 and XDMF file. The singular values and mode 
coefficients are not referenced in the XDMF file. The singular values as well as the mode coefficients are saved in full
whereas the modes are only saved up to the optimal rank (if not specified otherwise). Prior to performing the SVD,
the fields are weighted with the cell areas to improve the accuracy and comparability.

## General notes
### Memory requirements
The RAM needs to be large enough to hold at least:
- a single snapshot of the original grid
- the original grid
- the interpolated grid (size depends on the specified target metric)
- the levels of the interpolated grid (size depends on the specified target metric)
- a snapshot of the interpolated field (size depends on the specified target metric)

The required memory can be estimated based on the original grid and the target metric. Consider the example of a single 
snapshot having a size of 30 MB (double precision) and the original grid of 10 MB. The target metric is set to *75%*, 
leading to an approximate max. size of *7.5* MB for the generated grid and cell levels, and *22.5* MB for a single 
snapshot of the interpolated field. Consequently, interpolation and export of a single snapshot requires at least *~80* 
MB of additional RAM. Note that this is just an estimation, the actual grid size and consequently required RAM size 
highly depends on the chosen metric. In most cases, the number of cells will scale much more favorable.

**Note:** When performing an SVD, the complete data matrix (all snapshots) of the interpolated field need to be loaded. 
The available RAM has to be large enough to hold all snapshots of the interpolated field as well as additional memory to
perform the SVD.

### Reaching the specified target metric
- if the target metric is not reached with sufficient accuracy, the parameter `n_cells_iter_start` and
`n_cells_iter_end` have to be decreased. If none provided, they are automatically set to:  
  
&emsp;&emsp; `n_cells_iter_start` = 1% of original grid size  
&emsp;&emsp; `n_cells_iter_end` = 5% of `n_cells_iter_start` 


- the refinement of the grid near geometries requires approximately the same amount of time as the adaptive refinement, 
so unless a high resolution of geometries is required, it is recommended to set `_refine_geometry = False`
- if the error between the original fields and the interpolated ones is still too large (despite `_refine_geometry = True`), 
the following steps can be performed for improvement:
  - the refinement level of the geometries can be increased by setting `_min_level_geometry` to a larger value. By default,
all geometries are refined with the max. cell level present at the geometry after the adaptive refinement. When providing a
value for the refinement level, all geometry objects will be refined with this level
  - activate the delta level constraint by setting `_max_delta_level = True`
  - additionally, a second metric can be added, increasing the weight of areas near geometries (e.g., adding the influence
  of the shear stress to the existing metric)

### Projection of the coordinates for 2D in x-y-plane
- for 2D cases, the coordinates of the generated grid are always exported in the *x-y-* plane, independently of the 
orientation of the original CFD data 

## Unit tests
- tests can be executed with `pytest` inside the `tests` directory
- `pytest` can be installed via: `pip install pytest`

## Issues
If you have any questions or something is not working as expected, fell free to open up a new 
[issue](https://github.com/JanisGeise/sparseSpatialSampling/issues). There are some known issues, which are listed below.

### Known issues
#### Visibility of internal nodes in ParaView
- for 3D cases, the internal nodes are not or only partially displayed in Paraview, although they are present in the 
HDF5 file
- the fields and all points are present, each node and each center has a value, which is displayed correctly

This seems to be a rendering issue in Paraview resulting from the sorting of the nodes. However, this issue
should not be affecting any computations or operations done in ParaView or with the interpolated data in general.

When exporting a grid from OpenFoam to HDF5 using the `flowtorch.data.FOAM2HDF5` converter, 
the internal nodes are also not displayed in Paraview. This supports the assumption that this is just a rendering issue.

#### Messed up grid nodes in Paraview
When using single precision, the grid nodes may be messed up in the x-y-plane when imported into Paraview in some parts of the domain. 
This issue was fixed by exporting everything in double precision, so it is recommended to use double precision 
throughout all computations in Paraview. Why this happens only in the x-y-plane is unknown.

Although it wasn't observed so far, for very fine grids this may even be happening with double precision. 
However, the cell centered values should not be affected by this (in case this happens).

The fields as well as the SVD are still performed in single precision to reduce the memory requirements.

## References
- Existing version of the $S^3$ algorithm can be found under: 
  - **D. Fernex, A. Weiner, B. R. Noack and R. Semaan.** *Sparse Spatial Sampling: A mesh sampling algorithm for efficient 
  processing of big simulation data*, DOI: https://doi.org/10.2514/6.2021-1484 (January, 2021).
- Idea & 1D implementation of the current version taken from [Andre Weiner](https://github.com/AndreWeiner)
