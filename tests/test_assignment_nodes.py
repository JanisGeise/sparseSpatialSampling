"""
    test the correct assignment of the node indices
"""
import pytest
import torch as pt

from sparseSpatialSampling.s_cube import SamplingTree


def test_assignment_nodes_uniform_grid_2d():
    # create test data (just randomly distributed points in space)
    xy = pt.randint(0, 11, (25, 2))
    metric = pt.ones(xy.size(0))

    # instantiate sampling tree
    sampling = SamplingTree(xy, metric, level_bounds=(2, 2), geometry_obj=[])

    # do two uniform refinement cycles, so we get a 4x4 uniform grid, cell 0 is the initial cell, cell 1-4 are its
    # child cells and cells 5-20 are the child cells of the child cells
    sampling._refine_uniform()

    """
    the grid now looks like this:

    1. uniform refinement cycle, nodes idx = 0 are [0, 1, 2, 3]
         ______         ___ ___
        | idx  |       | 2 | 3 |
        |  = 0 |  ->   | 1 | 4 |
         ------         --- ---
         
         the nodes are then added for each cell, meaning cell 1 adds nodes [4, 5, 6]
         cell 2 adds only node 7, because 4 & 5 are already present and so on

    2. uniform refinement cycle

                         ---- ---- ---- ----
         ___ ___        | 10 | 11 | 14 | 15 |
        | 2 | 3 |       |  9 | 12 | 13 | 16 |
        | 1 | 4 |   ->  |  6 |  7 | 18 | 19 |
         --- ---        |  5 |  8 | 17 | 20 |
                         ---- ---- ---- ----
    """

    # take a few different cells and check if their nodes are assigned correctly
    # cell in the lower left corner
    cell_idx5 = sampling._cells[sampling._leaf_cells[0]]
    assert cell_idx5.index == 5
    assert cell_idx5.node_idx[0] == 0
    assert cell_idx5.node_idx[1] == 9
    assert cell_idx5.node_idx[2] == 10
    assert cell_idx5.node_idx[3] == 11

    # cell in the lower left middle
    cell_idx7 = sampling._cells[sampling._leaf_cells[2]]
    assert cell_idx7.index == 7
    assert cell_idx7.node_idx[0] == 10
    assert cell_idx7.node_idx[1] == 12
    assert cell_idx7.node_idx[2] == 5
    assert cell_idx7.node_idx[3] == 13

    # cell in the upper-left middle
    cell_idx12 = sampling._cells[sampling._leaf_cells[7]]
    assert cell_idx12.index == 12
    assert cell_idx12.node_idx[0] == 12
    assert cell_idx12.node_idx[1] == 15
    assert cell_idx12.node_idx[2] == 17
    assert cell_idx12.node_idx[3] == 5

    # cell in the upper-right middle
    cell_idx13 = sampling._cells[sampling._leaf_cells[8]]
    assert cell_idx13.index == 13
    assert cell_idx13.node_idx[0] == 5
    assert cell_idx13.node_idx[1] == 17
    assert cell_idx13.node_idx[2] == 18
    assert cell_idx13.node_idx[3] == 19

    # cell in the upper-right corner
    cell_idx15 = sampling._cells[sampling._leaf_cells[10]]
    assert cell_idx15.index == 15
    assert cell_idx15.node_idx[0] == 18
    assert cell_idx15.node_idx[1] == 20
    assert cell_idx15.node_idx[2] == 2
    assert cell_idx15.node_idx[3] == 21


def test_assignment_nodes_uniform_grid_3d_single_level():
    # create test data (just randomly distributed points in space)
    xy = pt.randint(0, 11, (50, 3))
    metric = pt.ones(xy.size(0))

    # instantiate sampling tree, test for one level
    sampling = SamplingTree(xy, metric, level_bounds=(1, 1), geometry_obj=[])

    # do two uniform refinement cycles, so we get a 2x2x2 uniform grid with 8 cells
    sampling._refine_uniform()

    # make sure we have 27 nodes in total
    assert len(sampling.all_nodes) == 27

    # for each cell, check the nodes (nodes. no. 0-7 are the nodes of the initial cell)
    # cell no. 0 (= 1st child cell of initial cell)
    cell_idx1 = sampling._cells[sampling._leaf_cells[0]]
    assert cell_idx1.index == 1
    assert cell_idx1.node_idx[0] == 0
    assert cell_idx1.node_idx[1] == 8
    assert cell_idx1.node_idx[2] == 9
    assert cell_idx1.node_idx[3] == 10
    assert cell_idx1.node_idx[4] == 11
    assert cell_idx1.node_idx[5] == 12
    assert cell_idx1.node_idx[6] == 13
    assert cell_idx1.node_idx[7] == 14

    # cell no. 1 (= 2nd child cell of initial cell)
    cell_idx2 = sampling._cells[sampling._leaf_cells[1]]
    assert cell_idx2.index == 2
    assert cell_idx2.node_idx[0] == 8
    assert cell_idx2.node_idx[1] == 1
    assert cell_idx2.node_idx[2] == 15
    assert cell_idx2.node_idx[3] == 9
    assert cell_idx2.node_idx[4] == 12
    assert cell_idx2.node_idx[5] == 16
    assert cell_idx2.node_idx[6] == 17
    assert cell_idx2.node_idx[7] == 13

    # cell no. 2 (= 3rd child cell of initial cell)
    cell_idx3 = sampling._cells[sampling._leaf_cells[2]]
    assert cell_idx3.index == 3
    assert cell_idx3.node_idx[0] == 9
    assert cell_idx3.node_idx[1] == 15
    assert cell_idx3.node_idx[2] == 2
    assert cell_idx3.node_idx[3] == 18
    assert cell_idx3.node_idx[4] == 13
    assert cell_idx3.node_idx[5] == 17
    assert cell_idx3.node_idx[6] == 19
    assert cell_idx3.node_idx[7] == 20

    # cell no. 3 (= 4th child cell of initial cell)
    cell_idx4 = sampling._cells[sampling._leaf_cells[3]]
    assert cell_idx4.index == 4
    assert cell_idx4.node_idx[0] == 10
    assert cell_idx4.node_idx[1] == 9
    assert cell_idx4.node_idx[2] == 18
    assert cell_idx4.node_idx[3] == 3
    assert cell_idx4.node_idx[4] == 14
    assert cell_idx4.node_idx[5] == 13
    assert cell_idx4.node_idx[6] == 20
    assert cell_idx4.node_idx[7] == 21

    # cell no. 4 (= 5th child cell of initial cell)
    cell_idx5 = sampling._cells[sampling._leaf_cells[4]]
    assert cell_idx5.index == 5
    assert cell_idx5.node_idx[0] == 11
    assert cell_idx5.node_idx[1] == 12
    assert cell_idx5.node_idx[2] == 13
    assert cell_idx5.node_idx[3] == 14
    assert cell_idx5.node_idx[4] == 4
    assert cell_idx5.node_idx[5] == 22
    assert cell_idx5.node_idx[6] == 23
    assert cell_idx5.node_idx[7] == 24

    # cell no. 5 (= 6th child cell of initial cell)
    cell_idx6 = sampling._cells[sampling._leaf_cells[5]]
    assert cell_idx6.index == 6
    assert cell_idx6.node_idx[0] == 12
    assert cell_idx6.node_idx[1] == 16
    assert cell_idx6.node_idx[2] == 17
    assert cell_idx6.node_idx[3] == 13
    assert cell_idx6.node_idx[4] == 22
    assert cell_idx6.node_idx[5] == 5
    assert cell_idx6.node_idx[6] == 25
    assert cell_idx6.node_idx[7] == 23

    # cell no. 6 (= 7th child cell of initial cell)
    cell_idx7 = sampling._cells[sampling._leaf_cells[6]]
    assert cell_idx7.index == 7
    assert cell_idx7.node_idx[0] == 13
    assert cell_idx7.node_idx[1] == 17
    assert cell_idx7.node_idx[2] == 19
    assert cell_idx7.node_idx[3] == 20
    assert cell_idx7.node_idx[4] == 23
    assert cell_idx7.node_idx[5] == 25
    assert cell_idx7.node_idx[6] == 6
    assert cell_idx7.node_idx[7] == 26

    # cell no. 7 (= 8th child cell of initial cell)
    cell_idx8 = sampling._cells[sampling._leaf_cells[7]]
    assert cell_idx8.index == 8
    assert cell_idx8.node_idx[0] == 14
    assert cell_idx8.node_idx[1] == 13
    assert cell_idx8.node_idx[2] == 20
    assert cell_idx8.node_idx[3] == 21
    assert cell_idx8.node_idx[4] == 24
    assert cell_idx8.node_idx[5] == 23
    assert cell_idx8.node_idx[6] == 26
    assert cell_idx8.node_idx[7] == 7
