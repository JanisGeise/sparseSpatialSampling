"""
    test the correct assignment of the neighbors
    TODO: since parallelized test isn't passed anymore due to division by zero error -> check
"""
import pytest
import torch as pt

from ..s_cube import SamplingTree


def test_assignment_nb_uniform_grid_2d():
    # create test data (just randomly distributed points in space)
    xy = pt.randint(0, 11, (25, 2))
    metric = pt.ones(xy.size(0))

    # instantiate sampling tree
    sampling = SamplingTree(xy, metric, uniform_level=2, geometry_obj=[])

    # do two uniform refinement cycles, so we get a 4x4 uniform grid, cell 0 is the initial cell, cell 1-4 are its
    # child cells and cells 5-20 are the child cells of the child cells
    sampling._refine_uniform()

    """
    the grid now looks like this:
    
    1. uniform refinement cycle
         ______         ___ ___
        | idx  |       | 2 | 3 |
        |  = 0 |  ->   | 1 | 4 |
         ------         --- ---
         
    2. uniform refinement cycle
                        
                         ---- ---- ---- ----
         ___ ___        | 10 | 11 | 14 | 15 |
        | 2 | 3 |       |  9 | 12 | 13 | 16 |
        | 1 | 4 |   ->  |  6 |  7 | 18 | 19 |
         --- ---        |  5 |  8 | 17 | 20 |
                         ---- ---- ---- ----
    """

    # take a few different cells and check if their nb's are assigned correct
    # cell in the lower left corner
    cell_idx5 = sampling._cells[list(sampling._leaf_cells)[0]]
    assert cell_idx5.index == 5

    # since we are at the domain boundaries, there are no left and upper left neighbors
    assert cell_idx5.nb[0] is None
    assert cell_idx5.nb[1] is None

    # then the next 3 nb cells should be the remaining child cells of child cell 1
    assert cell_idx5.nb[2].index == 6
    assert cell_idx5.nb[3].index == 7
    assert cell_idx5.nb[4].index == 8

    # the last 3 nb should again be None, because we are at the domain boundary
    assert cell_idx5.nb[5] is None
    assert cell_idx5.nb[6] is None
    assert cell_idx5.nb[7] is None

    # do the same for cell number 7 (in the lower left middle of the grid)
    cell_idx7 = sampling._cells[list(sampling._leaf_cells)[2]]
    assert cell_idx7.index == 7
    assert cell_idx7.nb[0].index == 6
    assert cell_idx7.nb[1].index == 9
    assert cell_idx7.nb[2].index == 12
    assert cell_idx7.nb[3].index == 13
    assert cell_idx7.nb[4].index == 18
    assert cell_idx7.nb[5].index == 17
    assert cell_idx7.nb[6].index == 8
    assert cell_idx7.nb[7].index == 5

    # again for cell 13 (in the upper right middle of the grid)
    cell_idx13 = sampling._cells[list(sampling._leaf_cells)[8]]
    assert cell_idx13.index == 13
    assert cell_idx13.nb[0].index == 12
    assert cell_idx13.nb[1].index == 11
    assert cell_idx13.nb[2].index == 14
    assert cell_idx13.nb[3].index == 15
    assert cell_idx13.nb[4].index == 16
    assert cell_idx13.nb[5].index == 19
    assert cell_idx13.nb[6].index == 18
    assert cell_idx13.nb[7].index == 7

    # now for the remaining corner cells 10, 15 & 20, if they all pass, then we can conclude that all nb are assigned
    # correctly without testing the remaining cells
    cell_idx10 = sampling._cells[list(sampling._leaf_cells)[5]]
    assert cell_idx10.index == 10
    assert cell_idx10.nb[0] is None
    assert cell_idx10.nb[1] is None
    assert cell_idx10.nb[2] is None
    assert cell_idx10.nb[3] is None
    assert cell_idx10.nb[4].index == 11
    assert cell_idx10.nb[5].index == 12
    assert cell_idx10.nb[6].index == 9
    assert cell_idx10.nb[7] is None

    cell_idx15 = sampling._cells[list(sampling._leaf_cells)[10]]
    assert cell_idx15.index == 15
    assert cell_idx15.nb[0].index == 14
    assert cell_idx15.nb[1] is None
    assert cell_idx15.nb[2] is None
    assert cell_idx15.nb[3] is None
    assert cell_idx15.nb[4] is None
    assert cell_idx15.nb[5] is None
    assert cell_idx15.nb[6].index == 16
    assert cell_idx15.nb[7].index == 13

    cell_idx20 = sampling._cells[list(sampling._leaf_cells)[-1]]
    assert cell_idx20.index == 20
    assert cell_idx20.nb[0].index == 17
    assert cell_idx20.nb[1].index == 18
    assert cell_idx20.nb[2].index == 19
    assert cell_idx20.nb[3] is None
    assert cell_idx20.nb[4] is None
    assert cell_idx20.nb[5] is None
    assert cell_idx20.nb[6] is None
    assert cell_idx20.nb[7] is None


def test_assignment_nb_uniform_grid_3d():
    # create test data (just randomly distributed points in space)
    xy = pt.randint(0, 11, (50, 3))
    metric = pt.ones(xy.size(0))

    # instantiate sampling tree
    sampling = SamplingTree(xy, metric, uniform_level=2, geometry_obj=[])

    # do two uniform refinement cycles, so we get a 4x4x4 uniform grid
    sampling._refine_uniform()

    # if the test for the 2D grid passed, then we only need to check the assignment of NB in the 3rd dimension since the
    # assignment of NB in the same plane is the same for both 2D and 3D
    # start with the upper-left corner (south-west-upper)
    cell_idx9 = sampling._cells[list(sampling._leaf_cells)[0]]
    assert cell_idx9.index == 9

    # check the nb on the plane under the cell
    assert cell_idx9.nb[8] is None
    assert cell_idx9.nb[9] is None
    assert cell_idx9.nb[10].index == 14
    assert cell_idx9.nb[11].index == 15
    assert cell_idx9.nb[12].index == 16
    assert cell_idx9.nb[13] is None
    assert cell_idx9.nb[14] is None
    assert cell_idx9.nb[15] is None
    assert cell_idx9.nb[16].index == 13

    # check the nb on the plane above the cell
    assert cell_idx9.nb[17] is None
    assert cell_idx9.nb[18] is None
    assert cell_idx9.nb[19] is None
    assert cell_idx9.nb[20] is None
    assert cell_idx9.nb[21] is None
    assert cell_idx9.nb[22] is None
    assert cell_idx9.nb[23] is None
    assert cell_idx9.nb[24] is None
    assert cell_idx9.nb[25] is None

    # now check a cell near the center of the grid
    cell_idx43 = sampling._cells[list(sampling._leaf_cells)[34]]
    assert cell_idx43.index == 43

    # check the nb on the plane under the cell
    assert cell_idx43.nb[8].index == 46
    assert cell_idx43.nb[9].index == 53
    assert cell_idx43.nb[10].index == 56
    assert cell_idx43.nb[11].index == 61
    assert cell_idx43.nb[12].index == 70
    assert cell_idx43.nb[13].index == 69
    assert cell_idx43.nb[14].index == 48
    assert cell_idx43.nb[15].index == 45
    assert cell_idx43.nb[16].index == 47

    # check the nb on the plane above the cell
    assert cell_idx43.nb[17].index == 14
    assert cell_idx43.nb[18].index == 21
    assert cell_idx43.nb[19].index == 24
    assert cell_idx43.nb[20].index == 29
    assert cell_idx43.nb[21].index == 38
    assert cell_idx43.nb[22].index == 37
    assert cell_idx43.nb[23].index == 16
    assert cell_idx43.nb[24].index == 13
    assert cell_idx43.nb[25].index == 15

    # lastly, check a cell in the middle on the right side of the grid
    cell_idx32 = sampling._cells[list(sampling._leaf_cells)[23]]
    assert cell_idx32.index == 32

    # check the nb on the plane under the cell
    assert cell_idx32.nb[8].index == 57
    assert cell_idx32.nb[9].index == 58
    assert cell_idx32.nb[10].index == 59
    assert cell_idx32.nb[11] is None
    assert cell_idx32.nb[12] is None
    assert cell_idx32.nb[13] is None
    assert cell_idx32.nb[14].index == 67
    assert cell_idx32.nb[15].index == 66
    assert cell_idx32.nb[16].index == 60

    # check the nb on the plane above the cell
    assert cell_idx32.nb[17].index == 25
    assert cell_idx32.nb[18].index == 26
    assert cell_idx32.nb[19].index == 27
    assert cell_idx32.nb[20] is None
    assert cell_idx32.nb[21] is None
    assert cell_idx32.nb[22] is None
    assert cell_idx32.nb[23].index == 35
    assert cell_idx32.nb[24].index == 34
    assert cell_idx32.nb[25].index == 28


if __name__ == "__main__":
    pass
