"""
    implementation of the sparse spatial sampling algorithm (S^3) for 2D & 3D CFD data
"""
import numpy as np
import torch as pt

from time import time
from numba import njit
from sklearn.neighbors import KNeighborsRegressor

"""
Note:
        each cell has 8 neighbors (nb) in 2D case: 1 at each side, 1 at each corner of the cell. In 3D, there
        are additionally 9 neighbors of the plane above and below (in z-direction) present. The neighbors are
        assigned in clockwise direction starting at the left neighbor. For 3D 1st neighbors in same plane are
        assigned, then neighbors in lower plane and finally neighbors in upper plane. The indices of the
        neighbors list refer to the corresponding neighbors as:

        2D:
            0 = left nb, 1 = upper left nb, 2 = upper nb, 3 = upper right nb, 4 = right nb, 5 = lower right nb,
            6 = lower nb, 7 = lower left nb

        additionally for 3D:

            8 = left nb lower plane, 9 = upper left nb lower plane,  10 = upper nb lower plane,
            11 = upper right nb lower plane, 12 = right nb lower plane, 13 = lower right nb lower plane,
            14 = lower nb lower plane, 15 = lower left nb lower plane, 16 = center nb lower plane

            17 = left nb upper plane, 18 = upper left nb upper plane, 19 = upper nb upper plane,
            20 = upper right nb upper plane, 21 = right nb upper plane, 22 = lower right nb upper plane,
            23 = lower nb upper plane, 24 = lower left nb upper plane, 25 = center nb upper plane
           
        for readability, actual positions (indices) are replaced with cardinal direction as:
            n = north, e = east, s = south, w = west, l = lower plane, u = upper plane, c = center

        e.g. 'swu' = south-west neighbor cell in the plane above the current cell
"""
# possible positions of neighbors relative to the current cell
NB = {
    "w": 0, "nw": 1, "n": 2, "ne": 3, "e": 4, "se": 5, "s": 6, "sw": 7,
    "wl": 8, "nwl": 9, "nl": 10, "nel": 11, "el": 12, "sel": 13, "sl": 14, "swl": 15, "cl": 16,
    "wu": 17, "nwu": 18, "nu": 19, "neu": 20, "eu": 21, "seu": 22, "su": 23, "swu": 24, "cu": 25
}

# positions of the children or nodes of a cell (nodes are numbered in the same way as children of a parent cell)
CH = {"swu": 0, "nwu": 1, "neu": 2, "seu": 3, "swl": 4, "nwl": 5, "nel": 6, "sel": 7}


class Cell(object):
    """
    implements a cell for the KNN regressor
    """

    def __init__(self, index: int, parent, nb: list, center: pt.Tensor, level: int, children=None, metric=None,
                 gain=None, dimensions: int = 2, idx: list = None):
        """
        each cell has the following attributes:

        :param index: index of the cell (cell number)
        :param parent: the superior cell, from which this cell was created
        :param nb: list containing all neighbor of the cell
        :param center: coordinates of the cell center
        :param level: refinement level (how often do we need to divide an initial cell to get to this cell)
        :param children: is this cell a parent cell of other cells (contains this cell other, smaller cells inside it)
        :param metric: the prediction made by the KNN based on the cell centers
        :param gain: value indicating the benefit arising from refining this cell
        :param dimensions: number of physical dimensions (2D / 3D)
        """
        self.index = index
        self.parent = parent
        self.level = level
        self.center = center
        self.children = children
        self.metric = metric
        self.gain = gain
        self._n_dims = dimensions
        self.nb = nb
        self.node_idx = idx

    def leaf_cell(self):
        return self.children is None


class SamplingTree(object):
    """
    class implementing the SamplingTree, which is creates the grid (tree structure)
    """

    def __init__(self, vertices, target, n_cells: int = None, level_bounds=(2, 25), _smooth_geometry: bool = True):
        """
        initialize the KNNand settings, create an initial cell, which can be refined iteratively in the 'refine'-methods

        :param vertices: coordinates of the nodes of the original mesh (CFD)
        :param target: the metric based on which the grid should be created, e.g. std. deviation of pressure wrt time
        :param n_cells: max. number of cell, if 'None', then the refinement process stopps automatically
        :param level_bounds: min. and max. number of levels of the final grid
        :param _smooth_geometry: flag for final refinement of the mesh around geometries and domain boundaries
        """
        self._vertices = vertices
        self._target = target
        self._n_cells = 0
        self._n_cells_max = 1e9 if n_cells is None else n_cells
        self._min_level = level_bounds[0]
        self._max_level = level_bounds[1]
        self._current_min_level = 0
        self._current_max_level = 0
        self._cells_per_iter_start = int(0.01 * vertices.size()[0])  # starting value = 1% of original grid size
        self._cells_per_iter_end = int(0.01 * self._cells_per_iter_start)  # end value = 1% of start value
        self._cells_per_iter = self._cells_per_iter_start
        self._width = None
        self._n_dimensions = self._vertices.size()[-1]
        self._knn = KNeighborsRegressor(n_neighbors=8 if self._n_dimensions == 2 else 26, weights="distance")
        self._knn.fit(self._vertices, self._target)
        self._cells = None
        self._leaf_cells = None
        self._geometry = []
        self._smooth_geometry = _smooth_geometry
        self._geometry_refinement_cycles = 1
        self._global_gain = []
        self._stop_thr = 1e-3
        self.all_nodes = []
        self.all_centers = []
        self.face_ids = None

        # offset matrix, used for computing the cell centers
        if self._n_dimensions == 2:
            self._directions = pt.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        else:
            # same order as for 2D
            self._directions = pt.tensor([[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
                                          [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]])

        # create initial cell and compute its gain
        self._create_first_cell()
        self._compute_global_gain()

        # remove the vertices and target to free up memory since they are only required for fitting the KNN and
        # computing the dominant width of the domain
        self._remove_original_cfd_data()

    def _create_first_cell(self) -> None:
        """
        creates a single cell based on the dominant dimension of the numerical domain, used as a starting point for the
        refinement process

        :return: None
        """
        self._width = max([self._vertices[:, i].max() - self._vertices[:, i].min() for i in
                           range(self._n_dimensions)]).item()

        # compute the node locations, we can't use '_compute_cell_centers()', because we don't have any cells yet
        centers_ = pt.zeros((pow(2, self._n_dimensions) + 1, self._n_dimensions))

        for node in range(pow(2, self._n_dimensions) + 1):
            for d in range(self._n_dimensions):
                centers_[node, d] = (pt.max(self._vertices[:, d]).item() + pt.min(self._vertices[:, d]).item()) / 2.0

        # correct cell centers of children with offset -> in each direction +- 0.25 of cell width
        nodes = centers_[1:, :] + self._directions * 0.5 * self._width
        centers_[1:, :] += self._directions * 0.25 * self._width

        # make prediction
        metric = self._knn.predict(centers_.numpy()).squeeze()

        # gain of 1st cell
        sum_distances = sum([abs(metric[0] - metric[i]) for i in range(1, len(metric))])
        gain = pow(self._width / 2, 2) * sum_distances

        # add the first cell
        self._n_cells += 1

        # add the node coordinates
        for n in range(nodes.size()[0]):
            self.all_nodes.append(nodes[n, :])
        self.all_centers.append(centers_[0, :])

        # we have max. 8 neighbors for each cell in 2D case (4 at sides + 4 at corners), for 3D we get 9 more per level
        # -> 26 neighbors in total
        self._cells = [Cell(0, None, self._knn.n_neighbors * [None], centers_[0, :], 0, None, metric[0], gain,
                            dimensions=self._n_dimensions, idx=list(range(nodes.size()[0])))]
        self._update_leaf_cells()

    def _remove_original_cfd_data(self) -> None:
        """
        delete the original coordinates (CFD) and metric since the are not used anymore

        :return: None
        """
        del self._vertices
        del self._target

    def _find_neighbors(self, cell) -> list:
        """
        find all the neighbors of the current cell

        :param cell: the current cell
        :return: list with all neighbors of the cell, if geometry or domain bound is neighbor, the entry is set to'None'
        """
        n_neighbors = self._knn.n_neighbors
        if cell.level < 2:
            return n_neighbors * [None]
        else:
            # iterate over all possible neighbor cells and assign them if they are present
            nb_tmp = n_neighbors * [None]
            for n in range(len(nb_tmp)):
                if cell.parent.nb[n] is not None:
                    nb_tmp[n] = cell.parent.nb[n]

            return nb_tmp

    def _refine_uniform(self) -> None:
        """
        create uniform background mesh to save runtime, since the min. level of the generated grid is likely to be > 1
        (uniform refinement is significantly faster than adaptive refinement)

        :return: None
        """
        for _ in range(self._min_level):
            new_cells = []
            new_index = len(self._cells)
            for i in self._leaf_cells:
                cell = self._cells[i]

                # find the neighbors of current cell
                neighbors = self._find_neighbors(cell)

                # compute cell centers
                loc_center = self._compute_cell_centers(i, keep_parent_center_=False)

                # assign the neighbors of current cell and add all new cells as children
                cell.children = tuple(self._assign_neighbors(loc_center, cell, neighbors, new_index))

                # assign idx for the newly created nodes
                self._assign_indices(cell.children)

                new_cells.extend(cell.children)
                self._n_cells += (pow(2, self._n_dimensions) - 1)
                new_index += pow(2, self._n_dimensions)

            self._cells.extend(new_cells)
            self._update_leaf_cells()
            self._update_gain()
            self._current_min_level += 1
            self._current_max_level += 1

            # delete cells which are outside the domain or inside a geometry (here we update all cells every iteration)
            self.remove_invalid_cells([c.index for c in new_cells])

    def _update_gain(self) -> None:
        """
        update the gain of all (new) leaf cells

        :return: None
        """
        indices, centers = ([], [])
        for i in self._leaf_cells:
            if self._cells[i].gain is None or self._cells[i].metric is None:
                indices.append(i)

                # compute cell centers
                centers.append(self._compute_cell_centers(i))

        if len(indices) > 0:
            # create tensor with all centers of all leaf cells and their potential children
            all_centers = pt.cat(centers, dim=0)

            # predict the std. deviation of the target function at these centers
            metric = self._knn.predict(all_centers.numpy())
            metric = pt.from_numpy(metric).reshape(int(metric.shape[0] / centers[0].size()[0]), centers[0].size()[0])
            for i, index in enumerate(indices):
                cell = self._cells[index]

                # cell center of original cell
                cell.metric = metric[i, 0]

                # sum of the delta(std. dev (target function)) -> in which direction (left, right, ...) child changes
                # the target function the most relative to the original cell center to each newly created cell center
                sum_delta_metric = sum([abs(metric[i, 0] - metric[i, j]) for j in range(1, metric.size()[1])])

                cell.gain = 1 / pow(2, self._n_dimensions) * pow(self._width / pow(2, cell.level), self._n_dimensions) \
                            * sum_delta_metric

    def _update_leaf_cells(self) -> None:
        self._leaf_cells = [cell.index for cell in self._cells if cell.leaf_cell()]

    def _update_min_ref_level(self) -> None:
        min_level = min([self._cells[index].level for index in self._leaf_cells])
        self._current_min_level = max(self._current_min_level, min_level)

    def leaf_cells(self) -> list:
        return [self._cells[i] for i in self._leaf_cells]

    def refine(self) -> None:
        """
        implements the generation of the grid based on the original grid and a metric

        :return: None
        """
        print("Starting refinement:")
        start_time = time()
        if self._min_level > 0:
            self._refine_uniform()
            print("Finished uniform refinement.")
        else:
            self._update_leaf_cells()
        end_time_uniform = time()
        self._compute_global_gain()
        iteration_count = 0

        while abs(self._global_gain[-2] - self._global_gain[-1]) >= self._stop_thr and self._n_cells <= self._n_cells_max:
            print(f"\r\tStarting iteration no. {iteration_count}", end="", flush=True)

            # update _n_cells_per_iter based on the gain difference and threshold for stopping
            if len(self._global_gain) > 3:
                # predict the iterations left until stopping criteria is met (linearly), '_stop_thr' may be neglectable
                # due to its small value. This is just an eq. for a linear function, with an intersection at _stop_thr
                # and y-intercept at _global_gain[-3]
                pred = (self._stop_thr - self._global_gain[-3]) / (self._global_gain[-1] - self._global_gain[-2])

                # set the new n_cells_per_iter based on the distance to this iteration and its boundaries
                self._cells_per_iter = int(self._cells_per_iter_start / pred + self._cells_per_iter_end)

            self._update_gain()
            self._leaf_cells.sort(key=lambda x: self._cells[x].gain, reverse=True)
            to_refine = set()
            for i in self._leaf_cells[:min(self._cells_per_iter, self._n_cells)]:
                cell = self._cells[i]
                to_refine.add(cell.index)
                if self._current_min_level > 1:
                    # same plane (3D) or if 2D. We have 4 children in 2D & 8 in 3D, in this plane we need to check for
                    # children 0...3, so counter = 0, start = 0 and stop = 4
                    to_refine.update(self._check_constraint(i, 0))

                    # for 3D case, we additionally need to check upper and lower plane
                    if self._n_dimensions == 3:
                        # child cells 4...7 (same nb as children 0...3, but we need to check just in case)
                        to_refine.update(self._check_constraint(i, 0, 4, 8))

                        # then check nb of lower plane (plane under the current cell). Counter = 8, because cell 8
                        # corresponds to left nb lower plane of children 4...7; start = 4 because we are looking at
                        # children 4...7
                        to_refine.update(self._check_constraint(i, 8, 4, 8))

                        # now do the same for the upper plane (children 0...3), idx = 17 corresponds to left nb upper
                        # plane (= counter), start = 0 because we check children 0...4
                        to_refine.update(self._check_constraint(i, 17, 0, 4))

            new_cells = []
            new_index = len(self._cells)
            for i in to_refine:
                cell = self._cells[i]

                # find the neighbors of current cell
                neighbors = self._find_neighbors(cell)

                # compute cell centers
                loc_center = self._compute_cell_centers(i, keep_parent_center_=False)

                # assign the neighbors of current cell and add all new cells as children
                cell.children = tuple(self._assign_neighbors(loc_center, cell, neighbors, new_index))

                # assign idx for the newly created nodes
                self._assign_indices(cell.children)

                new_cells.extend(cell.children)
                self._n_cells += (pow(2, self._n_dimensions) - 1)
                self._current_max_level = max(self._current_max_level, cell.level + 1)

                # for each cell in to_refine, we added 4 cells in 2D (2 cells in 1D), 8 cells in 3D
                new_index += pow(2, self._n_dimensions)

            # add the newly generated cell to the list of all existing cells and update everything
            self._cells.extend(new_cells)
            self._update_leaf_cells()
            self._update_gain()
            self._update_min_ref_level()

            # check the newly generated cells if they are outside the domain or inside a geometry, if so delete them
            self.remove_invalid_cells([c.index for c in new_cells])

            # compute global gain after refinement to check if we can stop the refinement
            self._compute_global_gain()
            iteration_count += 1

        # refine the grid near geometry objects is specified
        print("\nFinished adaptive refinement.")
        if self._smooth_geometry:
            t_start_geometry = time()
            self._refine_geometry()
            t_end_geometry = time()

        # assemble the final grid
        print("Starting renumbering final mesh.")
        t_start_renumber = time()
        self._resort_nodes_and_indices_of_grid()
        end_time = time()

        # print timings and size of final mesh
        print("Finished refinement in {:2.4f} s ({:d} iterations).".format(end_time - start_time, iteration_count))
        print("Time for uniform refinement: {:2.4f} s".format(end_time_uniform - start_time))
        if self._smooth_geometry:
            print("Time for adaptive refinement: {:2.4f} s".format(t_start_geometry - end_time_uniform))
            print("Time for geometry refinement: {:2.4f} s".format(t_end_geometry - t_start_geometry))
        else:
            print("Time for adaptive refinement: {:2.4f} s".format(t_start_renumber - end_time_uniform))
        print("Time for renumbering the final mesh: {:2.4f} s".format(end_time - t_start_renumber))
        print(self)

    def remove_invalid_cells(self, _refined_cells, _refine_geometry: bool = False) -> None or list:
        """
        check if any of the generated cells are located inside a geometry or outside a domain. If so, they are removed.

        :param _refined_cells: indices of the cells which are newly added
        :param _refine_geometry: flag if we want to refine the grid near geometry objects
        :return: None if we removed all invalid cells, if '_refine_geometry = True' returns list with indices of the
                 cells which are neighbors of geometries / domain boundaries
        """
        # check for each cell if it is located outside the domain or inside a geometry
        cells_invalid, idx = set(), set()
        for cell in _refined_cells:
            # compute the node locations of current cell
            nodes = self._compute_cell_centers(cell, factor_=0.5, keep_parent_center_=False)

            # check for each geometry object if the cell is inside the geometry or outside the domain
            invalid = [g.check_geometry(nodes, _refine_geometry) for g in self._geometry]

            # save the cell and corresponding index, set the gain to zero and make sure this cell is not changed to
            # leaf cell in future iterations resulting from delta level
            if any(invalid):
                cells_invalid.add(self._cells[cell])
                idx.add(cell)
                self._cells[cell].children, self._cells[cell].gain = [], 0

        # if we didn't find any invalid cells, we are done here. Else we need to reset the nb of the cells affected by
        # the removal of the masked cells
        if cells_invalid == set():
            return
        elif _refine_geometry:
            return idx
        else:
            # loop over all cells and check all neighbors for each cell, if invalid replace with None
            for cell in self._leaf_cells:
                self._cells[cell].nb = [None if n in cells_invalid else n for n in self._cells[cell].nb]

            # remove all invalid cells as leaf cells if we have any
            self._leaf_cells = [i for i in self._leaf_cells if i not in idx]

            # update n_cells
            self._n_cells = len(self._leaf_cells)

    def _resort_nodes_and_indices_of_grid(self) -> None:
        """
        remove all invalid and parent cells from the mesh. Sort the cell centers and vertices of the cells of the final
        grid with respect to their corresponding index and re-number all nodes.

        :return: None
        """
        _all_idx = pt.tensor([cell.node_idx for i, cell in enumerate(self._cells) if cell.leaf_cell()]).int()
        _unique_idx = _all_idx.flatten().unique()
        _all_available_idx = pt.arange(_all_idx.min().item(), _all_idx.max().item()+1)

        # get all node indices which are not used by all cells any more
        _unused_idx = _all_available_idx[~pt.isin(_all_available_idx, _unique_idx)].unique().int().numpy()
        del _unique_idx, _all_available_idx

        # re-index using numba -> faster than python, and this step is computationally quite expensive
        _unique_node_coord, _all_idx = renumber_node_indices(_all_idx.numpy(), pt.stack(self.all_nodes).numpy(),
                                                             _unused_idx, self._n_dimensions)

        # update node ID's and their coordinates
        self.face_ids = pt.from_numpy(_all_idx)
        self.all_nodes = pt.from_numpy(_unique_node_coord)
        self.all_centers = pt.stack([self._cells[cell].center for cell in self._leaf_cells])

    def _compute_cell_centers(self, idx_: int = None, factor_: float = 0.25, keep_parent_center_: bool = True,
                              cell_: Cell = None) -> pt.Tensor:
        """
        computes either the cell centers of the child cells for a given parent cell ('factor_ = 0.25) or the nodes of
        a given cell (factor_ = 0.5)

        Note:   although this method is called 'compute_nodes_final_mesh()', it computes the (corner) nodes of a cell if
                'factor_=0.5', not the cell center. If 'factor_=0.25', it computes the cell centers of all child cells,
                 hence the name for this method.

        :param idx_: index of the cell
        :param factor_: the factor (0.5 = half distance between two adjacent cell centers = node; 0.25 = cell center of
                        child cells relative to the parent cell center)
        :param keep_parent_center_: if the cell center of the parent cell should be deleted from the tensor prior return
        :return: either cell centers of child cells (factor_=0.25) or nodes of current cell (factor_=0.5)
        """
        center_ = self._cells[idx_].center if cell_ is None else cell_.center
        level_ = self._cells[idx_].level if cell_ is None else cell_.level

        # empty tensor with size of (parent_cell + n_children, n_dims)
        coord_ = pt.zeros((pow(2, self._n_dimensions) + 1, self._n_dimensions))

        # fill with cell centers of parent & child cells, the first row corresponds to parent cell
        coord_[0, :] = center_

        # compute the cell centers of the children relative to the  parent cell center location
        # -> offset in each direction is +- 0.25 * cell_width (in case the cell center of each cell should be computed
        # it is 0.5 * cell_width)
        coord_[1:, :] = center_ + self._directions * factor_ * self._width / pow(2, level_)

        # remove parent cell center if flag is set (no parent cell center if we just want to compute the cell centers of
        # each cell, but we need the parent cell center e.g. for computing the gain)
        return coord_[1:, :] if not keep_parent_center_ else coord_

    def _check_constraint(self, cell_no_, counter_, start_=0, stop_=4) -> list:
        """
        check if the level difference between a cell and its neighbors is max. one, if not then we need to refine all
        cells for which the constraint is violated

        :param cell_no_: current cell index for which we want to check the constraint
        :param counter_: current position of the nb we already checked based on the lasst cell
        :param start_: min. idx which a nb can have based on the location of the current cell
        :param stop_: max. idx which a nb can have based on the location of the current cell
        :return: list with indices of nb cell we need to refine, because delta level is > 1
        """
        # check if the level difference of current cell is larger than one wrt nb cells -> if so, then refine the nb
        # cell in order to avoid too large level differences between adjacent cells
        idx_list = []
        for child_ in range(start_, stop_):
            # if we have the correct child of the parent cell (aka current cell), then check its parent nb cell
            if self._cells[cell_no_] == self._cells[cell_no_].parent.children[child_]:
                # each child has 3 nb cells cornering it (the remaining cells are other child cells), so go around the
                # child cell check its nb (3 nb per level in 3D)
                for i in range(3):
                    # in same plane (or 2D), nb(0) = nb(8), but we only have 7 nb, so set (counter_ + i) to zero
                    # -> nb(0) = left nb of child 3 (i = 2, because left nb of child 3, counter_ = 6, because last cell)
                    # we already added this cell, but otherwise this would lead to an index error
                    if counter_ + i == 8 and stop_ == 4:
                        counter_ = -6

                    # if the nb parent cell of current parent cell is a leaf cell, and we are not at the domain boundary
                    # then add this cell to refine, because if we refine current parent cell then delta level would be 2
                    if self._cells[cell_no_].parent.nb[counter_ + i] is not None:
                        if self._cells[cell_no_].parent.nb[counter_ + i].leaf_cell():
                            idx_list.append(self._cells[cell_no_].parent.nb[counter_ + i].index)
            # move idx by 2 (current idx + 2 idx = 3 nb cells)
            counter_ += 2

        return idx_list

    def _refine_geometry(self) -> None:
        """
        stripped down version of the 'refine()' method for refinement of the final grid near geometry objects or domain
        boundaries. The documentation of this method is equivalent to the 'refine()' method
        """
        for _ in range(self._geometry_refinement_cycles):
            print("Starting geometry refinement.")
            self._update_leaf_cells()
            self._update_gain()
            to_refine = set()

            for i in set(self.remove_invalid_cells(self._leaf_cells, _refine_geometry=True)):
                cell = self._cells[i]
                to_refine.add(cell.index)
                if self._current_min_level > 1:
                    to_refine.update(self._check_constraint(i, 0))

                    if self._n_dimensions == 3:
                        to_refine.update(self._check_constraint(i, 0, 4, 8))
                        to_refine.update(self._check_constraint(i, 8, 4, 8))
                        to_refine.update(self._check_constraint(i, 17, 0, 4))

            new_cells = []
            new_index = len(self._cells)
            for i in to_refine:
                cell = self._cells[i]
                neighbors = self._find_neighbors(cell)
                loc_center = self._compute_cell_centers(i, keep_parent_center_=False)
                cell.children = tuple(self._assign_neighbors(loc_center, cell, neighbors, new_index))
                self._assign_indices(cell.children)
                new_cells.extend(cell.children)
                self._n_cells += (pow(2, self._n_dimensions) - 1)
                self._current_max_level = max(self._current_max_level, cell.level + 1)
                new_index += pow(2, self._n_dimensions)

            self._cells.extend(new_cells)
            self._update_leaf_cells()
            self._update_gain()
            self._update_min_ref_level()
            self.remove_invalid_cells([c.index for c in new_cells])
            print("Finished geometry refinement.")

    def _compute_global_gain(self) -> None:
        """
        compute a normalized gain of all cells in order to determine if we can stop the refinement. The gain is summed
        up over all leaf cells and then scaled by the area (2D) or volume (3D) of each cell and the number of leaf cells
        in total.

        :return: None
        """
        normalized_gain = []
        for c in self._leaf_cells:
            area = 1 / pow(2, self._n_dimensions) * pow(self._width / pow(2, self._cells[c].level), self._n_dimensions)
            normalized_gain.append(self._cells[c].gain / area)
        self._global_gain.append(sum(normalized_gain).item() / len(self._leaf_cells))

    def __len__(self):
        return self._n_cells

    def __str__(self):
        message = """
                        Number of cells: {:d}
                        Minimum ref. level: {:d}
                        Maximum ref. level: {:d}
                  """.format(self._n_cells, self._current_min_level, self._current_max_level)
        return message

    @property
    def n_dimensions(self):
        return self._n_dimensions

    @property
    def width(self):
        return self._width

    @property
    def geometry(self):
        return self._geometry

    def _assign_neighbors(self, loc_center: pt.Tensor, cell: Cell, neighbors: list, new_idx: int) -> list:
        """
        create a child cell from a given parent cell, assign its neighbors correctly to each child cell

        :param loc_center: center coordinates of the cell and its sub-cells
        :param cell: current cell
        :param neighbors: list containing all neighbors of current cell
        :param new_idx: new index of the cell
        :return: the child cells with correctly assigned neighbors
        """
        # each cell gets new index within all cells
        cell_tmp = [Cell(new_idx + idx, cell, len(neighbors) * [None], loc, cell.level + 1,
                         dimensions=self._n_dimensions) for idx, loc in enumerate(loc_center)]

        # add the neighbors for each of the child cells
        # neighbors for new lower left cell, we need to check for children, because we only assigned the parent cell,
        # which is ambiguous for the non-cornering neighbors. Further, we need to make sure that we have exactly N
        # children (if cells is removed due to geometry issues, then the children are empty list)
        check = [n is not None and n.children is not None and len(n.children) == pow(2, self._n_dimensions) for n in
                 neighbors]

        if check[NB["w"]]:
            cell_tmp[CH["swu"]].nb[NB["w"]] = neighbors[NB["w"]].children[CH["seu"]]
            cell_tmp[CH["swu"]].nb[NB["nw"]] = neighbors[NB["w"]].children[CH["neu"]]
            cell_tmp[CH["nwu"]].nb[NB["w"]] = neighbors[NB["w"]].children[CH["neu"]]
            cell_tmp[CH["nwu"]].nb[NB["sw"]] = neighbors[NB["w"]].children[CH["seu"]]
        else:
            cell_tmp[CH["swu"]].nb[NB["w"]] = neighbors[NB["w"]]
            cell_tmp[CH["swu"]].nb[NB["nw"]] = neighbors[NB["w"]]
            cell_tmp[CH["nwu"]].nb[NB["w"]] = neighbors[NB["w"]]
            cell_tmp[CH["nwu"]].nb[NB["sw"]] = neighbors[NB["w"]]

        if check[NB["nw"]]:
            cell_tmp[CH["nwu"]].nb[NB["nw"]] = neighbors[NB["nw"]].children[CH["seu"]]
        else:
            cell_tmp[CH["nwu"]].nb[NB["nw"]] = neighbors[NB["nw"]]

        if check[NB["n"]]:
            cell_tmp[CH["nwu"]].nb[NB["n"]] = neighbors[NB["n"]].children[CH["swu"]]
            cell_tmp[CH["nwu"]].nb[NB["ne"]] = neighbors[NB["n"]].children[CH["seu"]]
            cell_tmp[CH["nwu"]].nb[NB["nw"]] = neighbors[NB["n"]].children[CH["swu"]]
            cell_tmp[CH["nwu"]].nb[NB["n"]] = neighbors[NB["n"]].children[CH["seu"]]
        else:
            cell_tmp[CH["nwu"]].nb[NB["n"]] = neighbors[NB["n"]]
            cell_tmp[CH["nwu"]].nb[NB["ne"]] = neighbors[NB["n"]]
            cell_tmp[CH["neu"]].nb[NB["nw"]] = neighbors[NB["n"]]
            cell_tmp[CH["neu"]].nb[NB["n"]] = neighbors[NB["n"]]

        if check[NB["ne"]]:
            cell_tmp[CH["neu"]].nb[NB["ne"]] = neighbors[NB["ne"]].children[CH["swu"]]
        else:
            cell_tmp[CH["neu"]].nb[NB["ne"]] = neighbors[NB["ne"]]

        if check[NB["e"]]:
            cell_tmp[CH["neu"]].nb[NB["e"]] = neighbors[NB["e"]].children[CH["nwu"]]
            cell_tmp[CH["neu"]].nb[NB["se"]] = neighbors[NB["e"]].children[CH["swu"]]
            cell_tmp[CH["seu"]].nb[NB["ne"]] = neighbors[NB["e"]].children[CH["nwu"]]
            cell_tmp[CH["seu"]].nb[NB["e"]] = neighbors[NB["e"]].children[CH["swu"]]
        else:
            cell_tmp[CH["neu"]].nb[NB["e"]] = neighbors[NB["e"]]
            cell_tmp[CH["neu"]].nb[NB["se"]] = neighbors[NB["se"]]
            cell_tmp[CH["seu"]].nb[NB["ne"]] = neighbors[NB["e"]]
            cell_tmp[CH["seu"]].nb[NB["e"]] = neighbors[NB["e"]]

        if check[NB["se"]]:
            cell_tmp[CH["seu"]].nb[NB["se"]] = neighbors[NB["se"]].children[CH["nwu"]]
        else:
            cell_tmp[CH["seu"]].nb[NB["se"]] = neighbors[NB["se"]]

        if check[NB["s"]]:
            cell_tmp[CH["swu"]].nb[NB["se"]] = neighbors[NB["s"]].children[CH["neu"]]
            cell_tmp[CH["swu"]].nb[NB["s"]] = neighbors[NB["s"]].children[CH["nwu"]]
            cell_tmp[CH["seu"]].nb[NB["s"]] = neighbors[NB["s"]].children[CH["neu"]]
            cell_tmp[CH["seu"]].nb[NB["sw"]] = neighbors[NB["s"]].children[CH["nwu"]]
        else:
            cell_tmp[CH["swu"]].nb[NB["se"]] = neighbors[NB["s"]]
            cell_tmp[CH["swu"]].nb[NB["s"]] = neighbors[NB["s"]]
            cell_tmp[CH["seu"]].nb[NB["s"]] = neighbors[NB["s"]]
            cell_tmp[CH["seu"]].nb[NB["sw"]] = neighbors[NB["s"]]

        if check[NB["sw"]]:
            cell_tmp[CH["swu"]].nb[NB["sw"]] = neighbors[NB["sw"]].children[CH["neu"]]
        else:
            cell_tmp[CH["swu"]].nb[NB["sw"]] = neighbors[NB["sw"]]

        # remaining nb
        cell_tmp[CH["swu"]].nb[NB["n"]] = cell_tmp[CH["nwu"]]
        cell_tmp[CH["swu"]].nb[NB["ne"]] = cell_tmp[CH["neu"]]
        cell_tmp[CH["swu"]].nb[NB["e"]] = cell_tmp[CH["seu"]]
        cell_tmp[CH["nwu"]].nb[NB["e"]] = cell_tmp[CH["neu"]]
        cell_tmp[CH["nwu"]].nb[NB["se"]] = cell_tmp[CH["seu"]]
        cell_tmp[CH["nwu"]].nb[NB["s"]] = cell_tmp[CH["swu"]]
        cell_tmp[CH["neu"]].nb[NB["w"]] = cell_tmp[CH["nwu"]]
        cell_tmp[CH["neu"]].nb[NB["s"]] = cell_tmp[CH["seu"]]
        cell_tmp[CH["neu"]].nb[NB["sw"]] = cell_tmp[CH["swu"]]
        cell_tmp[CH["seu"]].nb[NB["w"]] = cell_tmp[CH["swu"]]
        cell_tmp[CH["seu"]].nb[NB["nw"]] = cell_tmp[CH["nwu"]]
        cell_tmp[CH["seu"]].nb[NB["n"]] = cell_tmp[CH["neu"]]

        # if 2D, then we are done but for 3D, we need to add neighbors of upper and lower plane
        # same plane as current cell is always the same as for 2D
        if self._n_dimensions == 3:
            if check[NB["w"]]:
                cell_tmp[CH["swu"]].nb[NB["wl"]] = neighbors[NB["w"]].children[CH["sel"]]
                cell_tmp[CH["swu"]].nb[NB["nwl"]] = neighbors[NB["w"]].children[CH["nel"]]
                cell_tmp[CH["nwu"]].nb[NB["wl"]] = neighbors[NB["w"]].children[CH["nel"]]
                cell_tmp[CH["nwu"]].nb[NB["swl"]] = neighbors[NB["w"]].children[CH["sel"]]
                cell_tmp[CH["swl"]].nb[NB["wu"]] = neighbors[NB["w"]].children[CH["seu"]]
                cell_tmp[CH["swl"]].nb[NB["nwu"]] = neighbors[NB["w"]].children[CH["neu"]]
                cell_tmp[CH["nwl"]].nb[NB["wu"]] = neighbors[NB["w"]].children[CH["neu"]]
                cell_tmp[CH["nwl"]].nb[NB["swu"]] = neighbors[NB["w"]].children[CH["seu"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["wl"]] = neighbors[NB["w"]]
                cell_tmp[CH["swu"]].nb[NB["nwl"]] = neighbors[NB["w"]]
                cell_tmp[CH["nwu"]].nb[NB["wl"]] = neighbors[NB["w"]]
                cell_tmp[CH["nwu"]].nb[NB["swl"]] = neighbors[NB["w"]]
                cell_tmp[CH["swl"]].nb[NB["wu"]] = neighbors[NB["w"]]
                cell_tmp[CH["swl"]].nb[NB["nwu"]] = neighbors[NB["w"]]
                cell_tmp[CH["nwl"]].nb[NB["wu"]] = neighbors[NB["w"]]
                cell_tmp[CH["nwl"]].nb[NB["swu"]] = neighbors[NB["w"]]

            if check[NB["nw"]]:
                cell_tmp[CH["nwu"]].nb[NB["nwl"]] = neighbors[NB["nw"]].children[CH["sel"]]
                cell_tmp[CH["nwl"]].nb[NB["nwu"]] = neighbors[NB["nw"]].children[CH["seu"]]
            else:
                cell_tmp[CH["nwu"]].nb[NB["nwl"]] = neighbors[NB["nw"]]
                cell_tmp[CH["nwl"]].nb[NB["nwu"]] = neighbors[NB["nw"]]

            if check[NB["n"]]:
                cell_tmp[CH["nwu"]].nb[NB["nl"]] = neighbors[NB["n"]].children[CH["swl"]]
                cell_tmp[CH["nwu"]].nb[NB["nel"]] = neighbors[NB["n"]].children[CH["sel"]]
                cell_tmp[CH["neu"]].nb[NB["nwl"]] = neighbors[NB["n"]].children[CH["swl"]]
                cell_tmp[CH["neu"]].nb[NB["nl"]] = neighbors[NB["n"]].children[CH["sel"]]
                cell_tmp[CH["nwl"]].nb[NB["nu"]] = neighbors[NB["n"]].children[CH["swu"]]
                cell_tmp[CH["nwl"]].nb[NB["neu"]] = neighbors[NB["n"]].children[CH["seu"]]
                cell_tmp[CH["nel"]].nb[NB["nwu"]] = neighbors[NB["n"]].children[CH["swu"]]
                cell_tmp[CH["nel"]].nb[NB["nu"]] = neighbors[NB["n"]].children[CH["seu"]]
            else:
                cell_tmp[CH["nwu"]].nb[NB["nl"]] = neighbors[NB["n"]]
                cell_tmp[CH["nwu"]].nb[NB["nel"]] = neighbors[NB["n"]]
                cell_tmp[CH["neu"]].nb[NB["nwl"]] = neighbors[NB["n"]]
                cell_tmp[CH["neu"]].nb[NB["nl"]] = neighbors[NB["n"]]
                cell_tmp[CH["nwl"]].nb[NB["nu"]] = neighbors[NB["n"]]
                cell_tmp[CH["nwl"]].nb[NB["neu"]] = neighbors[NB["n"]]
                cell_tmp[CH["nel"]].nb[NB["nwu"]] = neighbors[NB["n"]]
                cell_tmp[CH["nel"]].nb[NB["nu"]] = neighbors[NB["n"]]

            if check[NB["ne"]]:
                cell_tmp[CH["neu"]].nb[NB["nel"]] = neighbors[NB["ne"]].children[CH["swl"]]
                cell_tmp[CH["nel"]].nb[NB["neu"]] = neighbors[NB["ne"]].children[CH["swu"]]
            else:
                cell_tmp[CH["neu"]].nb[NB["nel"]] = neighbors[NB["ne"]]
                cell_tmp[CH["nel"]].nb[NB["neu"]] = neighbors[NB["ne"]]

            if check[NB["e"]]:
                cell_tmp[CH["neu"]].nb[NB["el"]] = neighbors[NB["e"]].children[CH["nwl"]]
                cell_tmp[CH["neu"]].nb[NB["sel"]] = neighbors[NB["e"]].children[CH["swl"]]
                cell_tmp[CH["seu"]].nb[NB["nel"]] = neighbors[NB["e"]].children[CH["nwl"]]
                cell_tmp[CH["seu"]].nb[NB["el"]] = neighbors[NB["e"]].children[CH["swl"]]
                cell_tmp[CH["nel"]].nb[NB["eu"]] = neighbors[NB["e"]].children[CH["nwu"]]
                cell_tmp[CH["nel"]].nb[NB["seu"]] = neighbors[NB["e"]].children[CH["swu"]]
                cell_tmp[CH["sel"]].nb[NB["neu"]] = neighbors[NB["e"]].children[CH["nwu"]]
                cell_tmp[CH["sel"]].nb[NB["eu"]] = neighbors[NB["e"]].children[CH["swu"]]
            else:
                cell_tmp[CH["neu"]].nb[NB["el"]] = neighbors[NB["e"]]
                cell_tmp[CH["neu"]].nb[NB["sel"]] = neighbors[NB["e"]]
                cell_tmp[CH["seu"]].nb[NB["nel"]] = neighbors[NB["e"]]
                cell_tmp[CH["seu"]].nb[NB["el"]] = neighbors[NB["e"]]
                cell_tmp[CH["nel"]].nb[NB["eu"]] = neighbors[NB["e"]]
                cell_tmp[CH["nel"]].nb[NB["seu"]] = neighbors[NB["e"]]
                cell_tmp[CH["sel"]].nb[NB["neu"]] = neighbors[NB["e"]]
                cell_tmp[CH["sel"]].nb[NB["eu"]] = neighbors[NB["e"]]

            if check[NB["se"]]:
                cell_tmp[CH["seu"]].nb[NB["sel"]] = neighbors[NB["se"]].children[CH["nwl"]]
                cell_tmp[CH["sel"]].nb[NB["seu"]] = neighbors[NB["se"]].children[CH["nwu"]]
            else:
                cell_tmp[CH["seu"]].nb[NB["sel"]] = neighbors[NB["se"]]
                cell_tmp[CH["sel"]].nb[NB["seu"]] = neighbors[NB["se"]]

            if check[NB["s"]]:
                cell_tmp[CH["swu"]].nb[NB["sel"]] = neighbors[NB["s"]].children[CH["nel"]]
                cell_tmp[CH["swu"]].nb[NB["sl"]] = neighbors[NB["s"]].children[CH["nwl"]]
                cell_tmp[CH["seu"]].nb[NB["sl"]] = neighbors[NB["s"]].children[CH["nel"]]
                cell_tmp[CH["seu"]].nb[NB["swl"]] = neighbors[NB["s"]].children[CH["nwl"]]
                cell_tmp[CH["swl"]].nb[NB["seu"]] = neighbors[NB["s"]].children[CH["neu"]]
                cell_tmp[CH["swl"]].nb[NB["su"]] = neighbors[NB["s"]].children[CH["nwu"]]
                cell_tmp[CH["sel"]].nb[NB["su"]] = neighbors[NB["s"]].children[CH["neu"]]
                cell_tmp[CH["sel"]].nb[NB["swu"]] = neighbors[NB["s"]].children[CH["nwu"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["sel"]] = neighbors[NB["s"]]
                cell_tmp[CH["swu"]].nb[NB["sl"]] = neighbors[NB["s"]]
                cell_tmp[CH["seu"]].nb[NB["sl"]] = neighbors[NB["s"]]
                cell_tmp[CH["seu"]].nb[NB["swl"]] = neighbors[NB["s"]]
                cell_tmp[CH["swl"]].nb[NB["seu"]] = neighbors[NB["s"]]
                cell_tmp[CH["swl"]].nb[NB["su"]] = neighbors[NB["s"]]
                cell_tmp[CH["sel"]].nb[NB["su"]] = neighbors[NB["s"]]
                cell_tmp[CH["sel"]].nb[NB["swu"]] = neighbors[NB["s"]]

            if check[NB["sw"]]:
                cell_tmp[CH["swu"]].nb[NB["swl"]] = neighbors[NB["sw"]].children[CH["nel"]]
                cell_tmp[CH["swl"]].nb[NB["swu"]] = neighbors[NB["sw"]].children[CH["neu"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["swl"]] = neighbors[NB["sw"]]
                cell_tmp[CH["swl"]].nb[NB["swu"]] = neighbors[NB["sw"]]

            if check[NB["wl"]]:
                cell_tmp[CH["swl"]].nb[NB["wl"]] = neighbors[NB["wl"]].children[CH["seu"]]
                cell_tmp[CH["swl"]].nb[NB["nwl"]] = neighbors[NB["wl"]].children[CH["neu"]]
                cell_tmp[CH["nwl"]].nb[NB["wl"]] = neighbors[NB["wl"]].children[CH["neu"]]
                cell_tmp[CH["nwl"]].nb[NB["swl"]] = neighbors[NB["wl"]].children[CH["seu"]]
            else:
                cell_tmp[CH["swl"]].nb[NB["wl"]] = neighbors[NB["wl"]]
                cell_tmp[CH["swl"]].nb[NB["nwl"]] = neighbors[NB["wl"]]
                cell_tmp[CH["nwl"]].nb[NB["wl"]] = neighbors[NB["wl"]]
                cell_tmp[CH["nwl"]].nb[NB["swl"]] = neighbors[NB["wl"]]

            if check[NB["nwl"]]:
                cell_tmp[CH["nwl"]].nb[NB["nwl"]] = neighbors[NB["nwl"]].children[CH["seu"]]
            else:
                cell_tmp[CH["nwl"]].nb[NB["nwl"]] = neighbors[NB["nwl"]]

            if check[NB["nl"]]:
                cell_tmp[CH["nwl"]].nb[NB["nl"]] = neighbors[NB["nl"]].children[CH["swu"]]
                cell_tmp[CH["nwl"]].nb[NB["nel"]] = neighbors[NB["nl"]].children[CH["seu"]]
                cell_tmp[CH["nel"]].nb[NB["nwl"]] = neighbors[NB["nl"]].children[CH["swu"]]
                cell_tmp[CH["nel"]].nb[NB["nl"]] = neighbors[NB["nl"]].children[CH["seu"]]
            else:
                cell_tmp[CH["nwl"]].nb[NB["nl"]] = neighbors[NB["nl"]]
                cell_tmp[CH["nwl"]].nb[NB["nel"]] = neighbors[NB["nl"]]
                cell_tmp[CH["nel"]].nb[NB["nwl"]] = neighbors[NB["nl"]]
                cell_tmp[CH["nel"]].nb[NB["nl"]] = neighbors[NB["nl"]]

            if check[NB["nel"]]:
                cell_tmp[CH["nel"]].nb[NB["nel"]] = neighbors[NB["nel"]].children[CH["swu"]]
            else:
                cell_tmp[CH["nel"]].nb[NB["nel"]] = neighbors[NB["nel"]]

            if check[NB["el"]]:
                cell_tmp[CH["nel"]].nb[NB["el"]] = neighbors[NB["el"]].children[CH["nwu"]]
                cell_tmp[CH["nel"]].nb[NB["sel"]] = neighbors[NB["el"]].children[CH["swu"]]
                cell_tmp[CH["sel"]].nb[NB["nel"]] = neighbors[NB["el"]].children[CH["nwu"]]
                cell_tmp[CH["sel"]].nb[NB["el"]] = neighbors[NB["el"]].children[CH["swu"]]
            else:
                cell_tmp[CH["nel"]].nb[NB["el"]] = neighbors[NB["el"]]
                cell_tmp[CH["nel"]].nb[NB["sel"]] = neighbors[NB["el"]]
                cell_tmp[CH["sel"]].nb[NB["nel"]] = neighbors[NB["el"]]
                cell_tmp[CH["sel"]].nb[NB["el"]] = neighbors[NB["el"]]

            if check[NB["sel"]]:
                cell_tmp[CH["sel"]].nb[NB["sel"]] = neighbors[NB["sel"]].children[CH["nwu"]]
            else:
                cell_tmp[CH["sel"]].nb[NB["sel"]] = neighbors[NB["sel"]]

            if check[NB["sl"]]:
                cell_tmp[CH["swl"]].nb[NB["sel"]] = neighbors[NB["sl"]].children[CH["neu"]]
                cell_tmp[CH["swl"]].nb[NB["sl"]] = neighbors[NB["sl"]].children[CH["nwu"]]
                cell_tmp[CH["sel"]].nb[NB["sl"]] = neighbors[NB["sl"]].children[CH["neu"]]
                cell_tmp[CH["sel"]].nb[NB["swl"]] = neighbors[NB["sl"]].children[CH["nwu"]]
            else:
                cell_tmp[CH["swl"]].nb[NB["sel"]] = neighbors[NB["sl"]]
                cell_tmp[CH["swl"]].nb[NB["sl"]] = neighbors[NB["sl"]]
                cell_tmp[CH["sel"]].nb[NB["sl"]] = neighbors[NB["sl"]]
                cell_tmp[CH["sel"]].nb[NB["swl"]] = neighbors[NB["sl"]]

            if check[NB["swl"]]:
                cell_tmp[CH["swl"]].nb[NB["swl"]] = neighbors[NB["swl"]].children[CH["neu"]]
            else:
                cell_tmp[CH["swl"]].nb[NB["swl"]] = neighbors[NB["swl"]]

            if check[NB["cl"]]:
                cell_tmp[CH["swl"]].nb[NB["nl"]] = neighbors[NB["cl"]].children[CH["nwu"]]
                cell_tmp[CH["swl"]].nb[NB["nel"]] = neighbors[NB["cl"]].children[CH["neu"]]
                cell_tmp[CH["swl"]].nb[NB["el"]] = neighbors[NB["cl"]].children[CH["seu"]]
                cell_tmp[CH["swl"]].nb[NB["cl"]] = neighbors[NB["cl"]].children[CH["swu"]]
                cell_tmp[CH["nwl"]].nb[NB["el"]] = neighbors[NB["cl"]].children[CH["neu"]]
                cell_tmp[CH["nwl"]].nb[NB["sel"]] = neighbors[NB["cl"]].children[CH["seu"]]
                cell_tmp[CH["nwl"]].nb[NB["sl"]] = neighbors[NB["cl"]].children[CH["swu"]]
                cell_tmp[CH["nwl"]].nb[NB["cl"]] = neighbors[NB["cl"]].children[CH["nwu"]]
                cell_tmp[CH["nel"]].nb[NB["wl"]] = neighbors[NB["cl"]].children[CH["nwu"]]
                cell_tmp[CH["nel"]].nb[NB["sl"]] = neighbors[NB["cl"]].children[CH["seu"]]
                cell_tmp[CH["nel"]].nb[NB["swl"]] = neighbors[NB["cl"]].children[CH["swu"]]
                cell_tmp[CH["nel"]].nb[NB["cl"]] = neighbors[NB["cl"]].children[CH["neu"]]
                cell_tmp[CH["sel"]].nb[NB["wl"]] = neighbors[NB["cl"]].children[CH["swu"]]
                cell_tmp[CH["sel"]].nb[NB["nwl"]] = neighbors[NB["cl"]].children[CH["nwu"]]
                cell_tmp[CH["sel"]].nb[NB["nl"]] = neighbors[NB["cl"]].children[CH["neu"]]
                cell_tmp[CH["sel"]].nb[NB["cl"]] = neighbors[NB["cl"]].children[CH["seu"]]
            else:
                cell_tmp[CH["swl"]].nb[NB["nl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["swl"]].nb[NB["nel"]] = neighbors[NB["cl"]]
                cell_tmp[CH["swl"]].nb[NB["el"]] = neighbors[NB["cl"]]
                cell_tmp[CH["swl"]].nb[NB["cl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nwl"]].nb[NB["el"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nwl"]].nb[NB["sel"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nwl"]].nb[NB["sl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nwl"]].nb[NB["cl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nel"]].nb[NB["wl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nel"]].nb[NB["sl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nel"]].nb[NB["swl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["nel"]].nb[NB["cl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["sel"]].nb[NB["wl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["sel"]].nb[NB["nwl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["sel"]].nb[NB["nl"]] = neighbors[NB["cl"]]
                cell_tmp[CH["sel"]].nb[NB["cl"]] = neighbors[NB["cl"]]

            if check[NB["wu"]]:
                cell_tmp[CH["swu"]].nb[NB["wu"]] = neighbors[NB["wu"]].children[CH["sel"]]
                cell_tmp[CH["swu"]].nb[NB["nwu"]] = neighbors[NB["wu"]].children[CH["nel"]]
                cell_tmp[CH["nwu"]].nb[NB["wu"]] = neighbors[NB["wu"]].children[CH["nel"]]
                cell_tmp[CH["nwu"]].nb[NB["swu"]] = neighbors[NB["wu"]].children[CH["sel"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["wu"]] = neighbors[NB["wu"]]
                cell_tmp[CH["swu"]].nb[NB["nwu"]] = neighbors[NB["wu"]]
                cell_tmp[CH["nwu"]].nb[NB["wu"]] = neighbors[NB["wu"]]
                cell_tmp[CH["nwu"]].nb[NB["swu"]] = neighbors[NB["wu"]]

            if check[NB["nwu"]]:
                cell_tmp[CH["nwu"]].nb[NB["nwu"]] = neighbors[NB["nwu"]].children[CH["sel"]]
            else:
                cell_tmp[CH["nwu"]].nb[NB["nwu"]] = neighbors[NB["nwu"]]

            if check[NB["nu"]]:
                cell_tmp[CH["nwu"]].nb[NB["nu"]] = neighbors[NB["nu"]].children[CH["swl"]]
                cell_tmp[CH["nwu"]].nb[NB["neu"]] = neighbors[NB["nu"]].children[CH["nwl"]]
                cell_tmp[CH["neu"]].nb[NB["nwu"]] = neighbors[NB["nu"]].children[CH["swl"]]
                cell_tmp[CH["neu"]].nb[NB["nu"]] = neighbors[NB["nu"]].children[CH["sel"]]
            else:
                cell_tmp[CH["nwu"]].nb[NB["nu"]] = neighbors[NB["nu"]]
                cell_tmp[CH["nwu"]].nb[NB["neu"]] = neighbors[NB["nu"]]
                cell_tmp[CH["neu"]].nb[NB["nwu"]] = neighbors[NB["nu"]]
                cell_tmp[CH["neu"]].nb[NB["nu"]] = neighbors[NB["nu"]]

            if check[NB["neu"]]:
                cell_tmp[CH["neu"]].nb[NB["neu"]] = neighbors[NB["neu"]].children[CH["swl"]]
            else:
                cell_tmp[CH["neu"]].nb[NB["neu"]] = neighbors[NB["neu"]]

            if check[NB["eu"]]:
                cell_tmp[CH["neu"]].nb[NB["eu"]] = neighbors[NB["eu"]].children[CH["nwl"]]
                cell_tmp[CH["neu"]].nb[NB["seu"]] = neighbors[NB["eu"]].children[CH["swl"]]
                cell_tmp[CH["seu"]].nb[NB["neu"]] = neighbors[NB["eu"]].children[CH["nwl"]]
                cell_tmp[CH["seu"]].nb[NB["eu"]] = neighbors[NB["eu"]].children[CH["swl"]]
            else:
                cell_tmp[CH["neu"]].nb[NB["eu"]] = neighbors[NB["eu"]]
                cell_tmp[CH["neu"]].nb[NB["seu"]] = neighbors[NB["eu"]]
                cell_tmp[CH["seu"]].nb[NB["neu"]] = neighbors[NB["eu"]]
                cell_tmp[CH["seu"]].nb[NB["eu"]] = neighbors[NB["eu"]]

            if check[NB["seu"]]:
                cell_tmp[CH["seu"]].nb[NB["seu"]] = neighbors[NB["seu"]].children[CH["nwl"]]
            else:
                cell_tmp[CH["seu"]].nb[NB["seu"]] = neighbors[NB["seu"]]

            if check[NB["su"]]:
                cell_tmp[CH["swu"]].nb[NB["seu"]] = neighbors[NB["su"]].children[CH["nel"]]
                cell_tmp[CH["swu"]].nb[NB["su"]] = neighbors[NB["su"]].children[CH["nwl"]]
                cell_tmp[CH["seu"]].nb[NB["su"]] = neighbors[NB["su"]].children[CH["nel"]]
                cell_tmp[CH["seu"]].nb[NB["swu"]] = neighbors[NB["su"]].children[CH["nwl"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["seu"]] = neighbors[NB["su"]]
                cell_tmp[CH["swu"]].nb[NB["su"]] = neighbors[NB["su"]]
                cell_tmp[CH["seu"]].nb[NB["su"]] = neighbors[NB["su"]]
                cell_tmp[CH["seu"]].nb[NB["swu"]] = neighbors[NB["su"]]

            if check[NB["swu"]]:
                cell_tmp[CH["swu"]].nb[NB["swu"]] = neighbors[NB["swu"]].children[CH["nel"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["swu"]] = neighbors[NB["swu"]]

            if check[NB["cu"]]:
                cell_tmp[CH["swu"]].nb[NB["nu"]] = neighbors[NB["cu"]].children[CH["nwl"]]
                cell_tmp[CH["swu"]].nb[NB["neu"]] = neighbors[NB["cu"]].children[CH["nel"]]
                cell_tmp[CH["swu"]].nb[NB["eu"]] = neighbors[NB["cu"]].children[CH["sel"]]
                cell_tmp[CH["swu"]].nb[NB["cu"]] = neighbors[NB["cu"]].children[CH["swl"]]
                cell_tmp[CH["nwu"]].nb[NB["eu"]] = neighbors[NB["cu"]].children[CH["nel"]]
                cell_tmp[CH["nwu"]].nb[NB["seu"]] = neighbors[NB["cu"]].children[CH["sel"]]
                cell_tmp[CH["nwu"]].nb[NB["su"]] = neighbors[NB["cu"]].children[CH["swl"]]
                cell_tmp[CH["nwu"]].nb[NB["cu"]] = neighbors[NB["cu"]].children[CH["nwl"]]
                cell_tmp[CH["neu"]].nb[NB["wu"]] = neighbors[NB["cu"]].children[CH["nwl"]]
                cell_tmp[CH["neu"]].nb[NB["su"]] = neighbors[NB["cu"]].children[CH["sel"]]
                cell_tmp[CH["neu"]].nb[NB["swu"]] = neighbors[NB["cu"]].children[CH["swl"]]
                cell_tmp[CH["neu"]].nb[NB["cu"]] = neighbors[NB["cu"]].children[CH["nel"]]
                cell_tmp[CH["seu"]].nb[NB["wu"]] = neighbors[NB["cu"]].children[CH["swl"]]
                cell_tmp[CH["seu"]].nb[NB["nwu"]] = neighbors[NB["cu"]].children[CH["nwl"]]
                cell_tmp[CH["seu"]].nb[NB["nu"]] = neighbors[NB["cu"]].children[CH["nel"]]
                cell_tmp[CH["seu"]].nb[NB["cu"]] = neighbors[NB["cu"]].children[CH["sel"]]
            else:
                cell_tmp[CH["swu"]].nb[NB["nu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["swu"]].nb[NB["neu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["swu"]].nb[NB["eu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["swu"]].nb[NB["cu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["nwu"]].nb[NB["eu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["nwu"]].nb[NB["seu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["nwu"]].nb[NB["su"]] = neighbors[NB["cu"]]
                cell_tmp[CH["nwu"]].nb[NB["cu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["neu"]].nb[NB["wu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["neu"]].nb[NB["su"]] = neighbors[NB["cu"]]
                cell_tmp[CH["neu"]].nb[NB["swu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["neu"]].nb[NB["cu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["seu"]].nb[NB["wu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["seu"]].nb[NB["nwu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["seu"]].nb[NB["nu"]] = neighbors[NB["cu"]]
                cell_tmp[CH["seu"]].nb[NB["cu"]] = neighbors[NB["cu"]]

            # add the remaining nb
            cell_tmp[CH["swu"]].nb[NB["nl"]] = cell_tmp[CH["nwl"]]
            cell_tmp[CH["swu"]].nb[NB["nel"]] = cell_tmp[CH["nel"]]
            cell_tmp[CH["swu"]].nb[NB["el"]] = cell_tmp[CH["sel"]]
            cell_tmp[CH["swu"]].nb[NB["cl"]] = cell_tmp[CH["swl"]]
            cell_tmp[CH["nwu"]].nb[NB["el"]] = cell_tmp[CH["nel"]]
            cell_tmp[CH["nwu"]].nb[NB["sel"]] = cell_tmp[CH["sel"]]
            cell_tmp[CH["nwu"]].nb[NB["sl"]] = cell_tmp[CH["swu"]]
            cell_tmp[CH["nwu"]].nb[NB["cl"]] = cell_tmp[CH["nwl"]]
            cell_tmp[CH["neu"]].nb[NB["wl"]] = cell_tmp[CH["nwl"]]
            cell_tmp[CH["neu"]].nb[NB["sl"]] = cell_tmp[CH["sel"]]
            cell_tmp[CH["neu"]].nb[NB["swl"]] = cell_tmp[CH["swl"]]
            cell_tmp[CH["neu"]].nb[NB["cl"]] = cell_tmp[CH["nel"]]
            cell_tmp[CH["seu"]].nb[NB["wl"]] = cell_tmp[CH["swl"]]
            cell_tmp[CH["seu"]].nb[NB["nwl"]] = cell_tmp[CH["nwl"]]
            cell_tmp[CH["seu"]].nb[NB["nl"]] = cell_tmp[CH["nel"]]
            cell_tmp[CH["seu"]].nb[NB["cl"]] = cell_tmp[CH["sel"]]
            cell_tmp[CH["swl"]].nb[NB["nu"]] = cell_tmp[CH["nwu"]]
            cell_tmp[CH["swl"]].nb[NB["neu"]] = cell_tmp[CH["neu"]]
            cell_tmp[CH["swl"]].nb[NB["eu"]] = cell_tmp[CH["seu"]]
            cell_tmp[CH["swl"]].nb[NB["cu"]] = cell_tmp[CH["swu"]]
            cell_tmp[CH["nwl"]].nb[NB["eu"]] = cell_tmp[CH["neu"]]
            cell_tmp[CH["nwl"]].nb[NB["seu"]] = cell_tmp[CH["seu"]]
            cell_tmp[CH["nwl"]].nb[NB["su"]] = cell_tmp[CH["swu"]]
            cell_tmp[CH["nwl"]].nb[NB["cu"]] = cell_tmp[CH["nwu"]]
            cell_tmp[CH["nel"]].nb[NB["wu"]] = cell_tmp[CH["nwu"]]
            cell_tmp[CH["nel"]].nb[NB["su"]] = cell_tmp[CH["seu"]]
            cell_tmp[CH["nel"]].nb[NB["swu"]] = cell_tmp[CH["swu"]]
            cell_tmp[CH["nel"]].nb[NB["cu"]] = cell_tmp[CH["neu"]]
            cell_tmp[CH["sel"]].nb[NB["wu"]] = cell_tmp[CH["swu"]]
            cell_tmp[CH["sel"]].nb[NB["nwu"]] = cell_tmp[CH["nwu"]]
            cell_tmp[CH["sel"]].nb[NB["nu"]] = cell_tmp[CH["neu"]]
            cell_tmp[CH["sel"]].nb[NB["cu"]] = cell_tmp[CH["seu"]]

        return cell_tmp

    def _assign_indices(self, cells):
        """
        due to round-off errors, accumulation of errors etc. even with double precision and is_close(), we are
        interpreting the same node shared by adjacent cells as different node, so we need to generate the node indices
        without any coordinate information

        :param cells: Tuple containing the child cells
        :return: None
        """
        for i in range(len(cells)):
            # add the cell center to the set containing all centers -> ensuring that order of nodes and centers is
            # consistent
            self.all_centers.append(cells[i].center)

            # initialize empty list
            cells[i].node_idx = [0] * pow(2, self._n_dimensions)

            # compute the node locations of current cell
            nodes = self._compute_cell_centers(factor_=0.5, keep_parent_center_=False, cell_=cells[i])

            # the node of the parent cell, for child cell 0: node 0 == node 0 of parent cell and so on
            cells[i].node_idx[i] = cells[i].parent.node_idx[i]

            # at the moment we treat 2D & 3D separately, if it works then we may make this more efficient by combining
            # in find_nb, we are assigning the parent nb, so we need to go to parent, and then to children
            if self._n_dimensions == 2:
                if i == 0:
                    # left nb
                    if self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].parent.nb[NB["w"]].children[CH["seu"]].node_idx[CH["neu"]]
                    else:
                        self.all_nodes.append(nodes[1, :])
                        cells[i].node_idx[CH["nwu"]] = len(self.all_nodes) - 1

                    # lower nb
                    if self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["s"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["s"]].children[CH["nwu"]].node_idx[CH["neu"]]
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1

                    # the remaining node in the center off all children
                    self.all_nodes.append(nodes[2, :])
                    cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1

                elif i == 1:
                    # upper nb
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["n"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].parent.nb[NB["n"]].children[CH["swu"]].node_idx[CH["seu"]]
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1
                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["nwu"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["swu"]].node_idx[CH["neu"]]

                elif i == 2:
                    # right nb
                    if self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["e"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["e"]].children[CH["nwu"]].node_idx[CH["swu"]]
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1
                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["neu"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["nwu"]].node_idx[CH["neu"]]

                elif i == 3:
                    # all new nodes are already introduced
                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["seu"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["swu"]].node_idx[CH["neu"]]
                    cells[i].node_idx[CH["neu"]] = cells[CH["neu"]].node_idx[CH["seu"]]

            else:
                # child no. 0: new nodes 1, 2, 3, 4, 5, 6, 7 remain to add
                if i == 0:
                    # check left nb (same plane) if node 1 is present
                    if self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].parent.nb[NB["w"]].children[CH["seu"]].node_idx[CH["neu"]]
                    # check left nb (upper plane) if node 1 is present
                    elif self.check_nb_node(cells[i], child_no=CH["sel"], nb_no=NB["wu"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].parent.nb[NB["wu"]].children[CH["sel"]].node_idx[CH["nel"]]
                    # check nb above current cell (upper plane) if node 1 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].parent.nb[NB["cu"]].children[CH["swl"]].node_idx[CH["nwl"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[1, :])
                        cells[i].node_idx[CH["nwu"]] = len(self.all_nodes) - 1

                    # check nb above current cell (upper plane) if node 2 is present
                    if self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].parent.nb[NB["cu"]].children[CH["swl"]].node_idx[CH["nel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 3 is present
                    if self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["s"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["s"]].children[CH["nwu"]].node_idx[CH["neu"]]
                    # check lower nb (upper plane) if node 3 is present
                    elif self.check_nb_node(cells[i], child_no=CH["nwl"], nb_no=NB["su"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["su"]].children[CH["nwl"]].node_idx[CH["nel"]]
                    # check nb above current cell (upper plane) if node 2 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["cu"]].children[CH["swl"]].node_idx[CH["sel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1

                    # check left nb (same plane) if node 4 is present
                    if self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["w"]):
                        cells[i].node_idx[CH["swl"]] = cells[i].parent.nb[NB["w"]].children[CH["seu"]].node_idx[CH["sel"]]
                    # check lower left corner nb (same plane) if node 4 is present
                    elif self.check_nb_node(cells[i], child_no=CH["neu"], nb_no=NB["sw"]):
                        cells[i].node_idx[CH["swl"]] = cells[i].parent.nb[NB["sw"]].children[CH["neu"]].node_idx[CH["nel"]]
                    # check lower nb (same plane) if node 4 is present
                    elif self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["s"]):
                        cells[i].node_idx[CH["swl"]] = cells[i].parent.nb[NB["s"]].children[CH["nwu"]].node_idx[CH["nwl"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[4, :])
                        cells[i].node_idx[CH["swl"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 5 is present
                    if self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["w"]].children[CH["seu"]].node_idx[CH["nel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[CH["nwl"]] = len(self.all_nodes) - 1

                    # node no. 6 is in center of all child cells, this node can't exist in other nb
                    self.all_nodes.append(nodes[6, :])
                    cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["s"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["s"]].children[CH["nwu"]].node_idx[CH["nel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[CH["sel"]] = len(self.all_nodes) - 1

                elif i == 1:
                    # child no. 1: new nodes 2, 5, 6 remain to add
                    # check upper nb (same plane) if node 2 is present
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["n"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].parent.nb[NB["n"]].children[CH["swu"]].node_idx[CH["seu"]]
                    # check upper nb (upper plane) if node 2 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["nu"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].parent.nb[NB["nu"]].children[CH["swl"]].node_idx[CH["sel"]]
                    # check nb above current cell (upper plane) if node 2 is present
                    elif self.check_nb_node(cells[i], child_no=CH["nwl"], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].parent.nb[NB["cu"]].children[CH["nwl"]].node_idx[CH["nel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1

                    # check left nb (same plane) if node 5 is present
                    if self.check_nb_node(cells[i], child_no=CH["neu"], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["w"]].children[CH["neu"]].node_idx[CH["nel"]]
                    # check upper left corner nb (same plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["nw"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["nw"]].children[CH["seu"]].node_idx[CH["sel"]]
                    # check upper nb (same plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["n"]].children[CH["swu"]].node_idx[CH["swl"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[CH["nwl"]] = len(self.all_nodes) - 1

                    # check upper nb (same plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["n"]].children[CH["swu"]].node_idx[CH["sel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["nwu"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["swu"]].node_idx[CH["neu"]]
                    cells[i].node_idx[CH["swl"]] = cells[CH["swu"]].node_idx[CH["nwl"]]
                    cells[i].node_idx[CH["sel"]] = cells[CH["swu"]].node_idx[CH["nel"]]
                elif i == 2:
                    # child no. 2: new nodes 3, 6, 7 remain to add
                    # check right nb (same plane) if node 3 is present
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["e"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["e"]].children[CH["swu"]].node_idx[CH["nwu"]]
                    # check right nb (upper plane) if node 3 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["eu"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["eu"]].children[CH["swl"]].node_idx[CH["nwl"]]
                    # check nb above current cell (upper plane) if node 3 is present
                    elif self.check_nb_node(cells[i], child_no=CH["sel"], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].parent.nb[NB["cu"]].children[CH["sel"]].node_idx[CH["nel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1

                    # check right nb (same plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["e"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["e"]].children[CH["nwu"]].node_idx[CH["nwl"]]
                    # check upper right corner nb (same plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["ne"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["ne"]].children[CH["swu"]].node_idx[CH["swl"]]
                    # check upper nb (same plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["n"]].children[CH["seu"]].node_idx[CH["sel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    # check right nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["e"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["e"]].children[CH["swu"]].node_idx[CH["nwl"]]
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[CH["sel"]] = len(self.all_nodes) - 1

                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["neu"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["nwu"]].node_idx[CH["neu"]]
                    cells[i].node_idx[CH["swl"]] = cells[CH["swu"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["nwl"]] = cells[CH["nwu"]].node_idx[CH["nel"]]
                elif i == 3:
                    # child no. 3: new nodes 7 remain to add
                    # check right nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["e"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["e"]].children[CH["swu"]].node_idx[CH["swl"]]
                    # check lower right corner nb (same plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["se"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["se"]].children[CH["nwu"]].node_idx[CH["nwl"]]
                    # check lower nb (same plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=CH["neu"], nb_no=NB["s"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["s"]].children[CH["neu"]].node_idx[CH["nel"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[CH["sel"]] = len(self.all_nodes) - 1

                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["seu"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["swu"]].node_idx[CH["neu"]]
                    cells[i].node_idx[CH["neu"]] = cells[CH["neu"]].node_idx[CH["seu"]]
                    cells[i].node_idx[CH["swl"]] = cells[CH["swu"]].node_idx[CH["sel"]]
                    cells[i].node_idx[CH["nwl"]] = cells[CH["swu"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["nel"]] = cells[CH["neu"]].node_idx[CH["sel"]]
                elif i == 4:
                    # child no. 4: new nodes 5, 6, 7 remain to add
                    # check left nb (same plane) if node 5 is present
                    if self.check_nb_node(cells[i], child_no=CH["sel"], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["w"]].children[CH["sel"]].node_idx[CH["nel"]]
                    # check left nb (lower plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=CH["seu"], nb_no=NB["wl"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["wl"]].children[CH["seu"]].node_idx[CH["neu"]]
                    # check nb below current cell (lower plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].parent.nb[NB["cl"]].children[CH["swu"]].node_idx[CH["nwu"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[CH["nwl"]] = len(self.all_nodes) - 1

                    # check nb below current cell (lower plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["cl"]].children[CH["swu"]].node_idx[CH["neu"]]
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=CH["nwl"], nb_no=NB["s"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["s"]].children[CH["nwl"]].node_idx[CH["nel"]]
                    # check lower corner nb (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["sl"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["sl"]].children[CH["nwu"]].node_idx[CH["neu"]]
                    # check nb below current cell (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["cl"]].children[CH["swu"]].node_idx[CH["seu"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[CH["sel"]] = len(self.all_nodes) - 1

                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["swl"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["swu"]].node_idx[CH["nwl"]]
                    cells[i].node_idx[CH["neu"]] = cells[CH["swu"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["swu"]].node_idx[CH["sel"]]
                elif i == 5:
                    # child no. 5: new nodes 6 remain to add
                    # check upper nb (same plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["n"]].children[CH["swl"]].node_idx[CH["sel"]]
                    # check upper corner nb (lower plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["nl"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["nl"]].children[CH["swu"]].node_idx[CH["seu"]]
                    # check nb below current cell (lower plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=CH["nwu"], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].parent.nb[NB["cl"]].children[CH["nwu"]].node_idx[CH["neu"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    cells[i].node_idx[CH["swu"]] = cells[CH["nwu"]].node_idx[CH["swl"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["nwu"]].node_idx[CH["nwl"]]
                    cells[i].node_idx[CH["neu"]] = cells[CH["nwu"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["nwu"]].node_idx[CH["sel"]]
                    cells[i].node_idx[CH["swl"]] = cells[CH["swl"]].node_idx[CH["nwl"]]
                    cells[i].node_idx[CH["sel"]] = cells[CH["swl"]].node_idx[CH["nel"]]
                elif i == 6:
                    # child no. 6: new nodes 7 to add
                    # check right nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=CH["swl"], nb_no=NB["e"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["e"]].children[CH["swl"]].node_idx[CH["nwl"]]
                    # check right corner nb (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=CH["swu"], nb_no=NB["el"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["el"]].children[CH["swu"]].node_idx[CH["nwu"]]
                    # check nb below current cell (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=CH["neu"], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].parent.nb[NB["cl"]].children[CH["neu"]].node_idx[CH["seu"]]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[CH["sel"]] = len(self.all_nodes) - 1

                    cells[i].node_idx[CH["swu"]] = cells[CH["neu"]].node_idx[CH["swl"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["neu"]].node_idx[CH["nwl"]]
                    cells[i].node_idx[CH["neu"]] = cells[CH["neu"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["neu"]].node_idx[CH["sel"]]
                    cells[i].node_idx[CH["swl"]] = cells[CH["nwl"]].node_idx[CH["sel"]]
                    cells[i].node_idx[CH["nwl"]] = cells[CH["nwl"]].node_idx[CH["nel"]]
                elif i == 7:
                    # child no. 7 no new nodes anymore, only assign the existing nodes
                    cells[i].node_idx[CH["swu"]] = cells[CH["seu"]].node_idx[CH["swl"]]
                    cells[i].node_idx[CH["nwu"]] = cells[CH["seu"]].node_idx[CH["nwl"]]
                    cells[i].node_idx[CH["neu"]] = cells[CH["seu"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["seu"]].node_idx[CH["sel"]]
                    cells[i].node_idx[CH["swl"]] = cells[CH["swl"]].node_idx[CH["sel"]]
                    cells[i].node_idx[CH["nwl"]] = cells[CH["swl"]].node_idx[CH["nel"]]
                    cells[i].node_idx[CH["nel"]] = cells[CH["nel"]].node_idx[CH["sel"]]

    def check_nb_node(self, _cell, child_no, nb_no):
        return _cell.parent is not None and _cell.parent.nb[nb_no] is not None and \
               _cell.parent.nb[nb_no].children is not None and \
               len(_cell.parent.nb[nb_no].children) == pow(2, self._n_dimensions) and \
               _cell.parent.nb[nb_no].children[child_no] is not None and \
               _cell.parent.nb[nb_no].children[child_no].level == _cell.level


@njit(fastmath=True)
def renumber_node_indices(all_idx: np.ndarray, all_nodes: np.ndarray, _unused_idx: np.ndarray, dims: int):
    """
    some things regarding numba:
        - 'i in _unused_idx' or (i == _unused_idx).any() takes longer than the loop for assigning 'check'
        - set() is not supported by numba, so we need to convert '_unused_idx' into a numpy array
        - np.isin() is not supported by numba
        - using an explizit loop (in 'if check') significantly speeds up the execution compared to advanced indexing
        'all_idx[all_idx > i - _visited] -= 1'
    """
    _unique_node_coord = np.zeros((all_nodes.shape[0] - _unused_idx.shape[0], dims))
    _counter, _visited = 0, 0

    # numba does not allow advanced indexing on more than 1 dimension, so we need to flatten it first and reshape at
    # the end, we can't run this loop in parallel, because it messes up the idx numbers
    orig_shape = all_idx.shape
    all_idx = all_idx.flatten()
    for i in range(all_nodes.shape[0]):
        left = 0
        right = len(_unused_idx) - 1
        check = False
        while left <= right:
            mid = (left + right) // 2
            if _unused_idx[mid] == i:
                check = True
                break
            elif _unused_idx[mid] < i:
                left = mid + 1
            else:
                right = mid - 1

        if check:
            # decrement all idx which are > current index by 1, since we are deleting this node. but since we are
            # overwriting the all_idx tensor, we need to account for the entries we already deleted
            _visited += 1
            for j in range(all_idx.shape[0]):
                if all_idx[j] > i - _visited:
                    all_idx[j] -= 1
        else:
            _unique_node_coord[_counter, :] = all_nodes[i, :]
            _counter += 1

    return _unique_node_coord, all_idx.reshape(orig_shape)


if __name__ == "__main__":
    pass
