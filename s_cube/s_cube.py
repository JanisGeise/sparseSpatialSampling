"""
    implementation of the sparse spatial sampling algorithm (S^3) for 2D & 3D CFD data
"""
import numpy as np
import torch as pt

from time import time

from numba import njit, prange, typed
from sklearn.neighbors import KNeighborsRegressor


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
                    23 = lower nb upper plane, 24 = lower left nb upper plane, 25 = center nb upper plane,
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
                for n in range(len(self._cells[cell].nb)):
                    if self._cells[cell].nb[n] in cells_invalid:
                        self._cells[cell].nb[n] = None

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
        t1 = time()
        _all_idx = pt.tensor([cell.node_idx for i, cell in enumerate(self._cells) if cell.leaf_cell()]).int()
        print("[DEBUG] dt1 = ", round(time() - t1, 6), "s")

        t2 = time()
        _unique_idx = _all_idx.flatten().unique()
        _all_available_idx = pt.arange(_all_idx.min().item(), _all_idx.max().item()+1)

        # get all node indices which are not used anymore, convert to set(), because search in set is faster
        _unused_idx = set(_all_available_idx[~pt.isin(_all_available_idx, _unique_idx)].int().tolist())
        del _unique_idx, _all_available_idx
        print("[DEBUG] dt2 = ", round(time() - t2, 6), "s")

        # test
        t3 = time()
        # _unique_node_coord, _all_idx = renumber_node_indices(_all_idx.numpy(), pt.stack(self.all_nodes).numpy(),
        #                                                      typed.List(_unused_idx), self._n_dimensions)

        # """
        _unique_node_coord = pt.zeros((len(self.all_nodes) - len(_unused_idx), self._n_dimensions))
        _counter, _visited = 0, 0
        for i in range(len(self.all_nodes)):
            if i in _unused_idx:
                # decrement all idx which are > current index by 1, since we are deleting this node. but since we are
                # overwriting the all_idx tensor, we need to account for the entries we already deleted
                _visited += 1
                _all_idx[_all_idx > i - _visited] -= 1
            else:
                _unique_node_coord[_counter, :] = self.all_nodes[i]
                _counter += 1
        # """
        print("[DEBUG] dt3 = ", round(time() - t3, 6), "s")
        t4 = time()

        # update node ID's and their coordinates
        # self.face_ids = pt.from_numpy(_all_idx)
        # self.all_nodes = pt.from_numpy(_unique_node_coord)
        self.face_ids = _all_idx
        self.all_nodes = _unique_node_coord
        self.all_centers = pt.stack([self._cells[cell].center for cell in self._leaf_cells])
        print("[DEBUG] dt4 =", round(time() - t4, 6), "s")
        print("[DEBUG] dt total =", round(time() - t1, 6), "s")

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
        TODO: exploitation of patterns (if any) to make this more readable & efficient

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

        if check[0]:
            cell_tmp[0].nb[0] = neighbors[0].children[3]
            cell_tmp[0].nb[1] = neighbors[0].children[2]
            cell_tmp[1].nb[0] = neighbors[0].children[2]
            cell_tmp[1].nb[7] = neighbors[0].children[3]
        else:
            cell_tmp[0].nb[0] = neighbors[0]
            cell_tmp[0].nb[1] = neighbors[0]
            cell_tmp[1].nb[0] = neighbors[0]
            cell_tmp[1].nb[7] = neighbors[0]

        if check[1]:
            cell_tmp[1].nb[1] = neighbors[1].children[3]
        else:
            cell_tmp[1].nb[1] = neighbors[1]

        if check[2]:
            cell_tmp[1].nb[2] = neighbors[2].children[0]
            cell_tmp[1].nb[3] = neighbors[2].children[3]
            cell_tmp[1].nb[1] = neighbors[2].children[0]
            cell_tmp[1].nb[2] = neighbors[2].children[3]
        else:
            cell_tmp[1].nb[2] = neighbors[2]
            cell_tmp[1].nb[3] = neighbors[2]
            cell_tmp[2].nb[1] = neighbors[2]
            cell_tmp[2].nb[2] = neighbors[2]

        if check[3]:
            cell_tmp[2].nb[3] = neighbors[3].children[0]
        else:
            cell_tmp[2].nb[3] = neighbors[3]

        if check[4]:
            cell_tmp[2].nb[4] = neighbors[4].children[1]
            cell_tmp[2].nb[5] = neighbors[4].children[0]
            cell_tmp[3].nb[3] = neighbors[4].children[1]
            cell_tmp[3].nb[4] = neighbors[4].children[0]
        else:
            cell_tmp[2].nb[4] = neighbors[4]
            cell_tmp[2].nb[5] = neighbors[5]
            cell_tmp[3].nb[3] = neighbors[4]
            cell_tmp[3].nb[4] = neighbors[4]

        if check[5]:
            cell_tmp[3].nb[5] = neighbors[5].children[1]
        else:
            cell_tmp[3].nb[5] = neighbors[5]

        if check[6]:
            cell_tmp[0].nb[5] = neighbors[6].children[2]
            cell_tmp[0].nb[6] = neighbors[6].children[1]
            cell_tmp[3].nb[6] = neighbors[6].children[2]
            cell_tmp[3].nb[7] = neighbors[6].children[1]
        else:
            cell_tmp[0].nb[5] = neighbors[6]
            cell_tmp[0].nb[6] = neighbors[6]
            cell_tmp[3].nb[6] = neighbors[6]
            cell_tmp[3].nb[7] = neighbors[6]

        if check[7]:
            cell_tmp[0].nb[7] = neighbors[7].children[2]
        else:
            cell_tmp[0].nb[7] = neighbors[7]

        # remaining nb
        cell_tmp[0].nb[2] = cell_tmp[1]
        cell_tmp[0].nb[3] = cell_tmp[2]
        cell_tmp[0].nb[4] = cell_tmp[3]
        cell_tmp[1].nb[4] = cell_tmp[2]
        cell_tmp[1].nb[5] = cell_tmp[3]
        cell_tmp[1].nb[6] = cell_tmp[0]
        cell_tmp[2].nb[0] = cell_tmp[1]
        cell_tmp[2].nb[6] = cell_tmp[3]
        cell_tmp[2].nb[7] = cell_tmp[0]
        cell_tmp[3].nb[0] = cell_tmp[0]
        cell_tmp[3].nb[1] = cell_tmp[1]
        cell_tmp[3].nb[2] = cell_tmp[2]

        # if 2D, then we are done but for 3D, we need to add neighbors of upper and lower plane
        # same plane as current cell is always the same as for 2D
        if self._n_dimensions == 3:
            # ----------------  lower left cell center, upper plane  ----------------
            # plane under the current cell
            if check[0]:
                cell_tmp[0].nb[8] = neighbors[0].children[7]
                cell_tmp[0].nb[9] = neighbors[0].children[6]
                cell_tmp[1].nb[8] = neighbors[0].children[6]
                cell_tmp[1].nb[15] = neighbors[0].children[7]
                cell_tmp[4].nb[17] = neighbors[0].children[3]
                cell_tmp[4].nb[18] = neighbors[0].children[2]
                cell_tmp[5].nb[17] = neighbors[0].children[2]
                cell_tmp[5].nb[24] = neighbors[0].children[3]
            else:
                cell_tmp[0].nb[8] = neighbors[0]
                cell_tmp[0].nb[9] = neighbors[0]
                cell_tmp[1].nb[8] = neighbors[0]
                cell_tmp[1].nb[15] = neighbors[0]
                cell_tmp[4].nb[17] = neighbors[0]
                cell_tmp[4].nb[18] = neighbors[0]
                cell_tmp[5].nb[17] = neighbors[0]
                cell_tmp[5].nb[24] = neighbors[0]

            if check[1]:
                cell_tmp[1].nb[9] = neighbors[1].children[7]
                cell_tmp[5].nb[18] = neighbors[1].children[3]
            else:
                cell_tmp[1].nb[9] = neighbors[1]
                cell_tmp[5].nb[18] = neighbors[1]

            if check[2]:
                cell_tmp[1].nb[10] = neighbors[2].children[4]
                cell_tmp[1].nb[11] = neighbors[2].children[7]
                cell_tmp[2].nb[9] = neighbors[2].children[4]
                cell_tmp[2].nb[10] = neighbors[2].children[7]
                cell_tmp[5].nb[19] = neighbors[2].children[0]
                cell_tmp[5].nb[20] = neighbors[2].children[3]
                cell_tmp[6].nb[18] = neighbors[2].children[0]
                cell_tmp[6].nb[19] = neighbors[2].children[3]
            else:
                cell_tmp[1].nb[10] = neighbors[2]
                cell_tmp[1].nb[11] = neighbors[2]
                cell_tmp[2].nb[9] = neighbors[2]
                cell_tmp[2].nb[10] = neighbors[2]
                cell_tmp[5].nb[19] = neighbors[2]
                cell_tmp[5].nb[20] = neighbors[2]
                cell_tmp[6].nb[18] = neighbors[2]
                cell_tmp[6].nb[19] = neighbors[2]

            if check[3]:
                cell_tmp[2].nb[11] = neighbors[3].children[4]
                cell_tmp[6].nb[20] = neighbors[3].children[0]
            else:
                cell_tmp[2].nb[11] = neighbors[3]
                cell_tmp[6].nb[20] = neighbors[3]

            if check[4]:
                cell_tmp[2].nb[12] = neighbors[4].children[5]
                cell_tmp[2].nb[13] = neighbors[4].children[4]
                cell_tmp[3].nb[11] = neighbors[4].children[5]
                cell_tmp[3].nb[12] = neighbors[4].children[4]
                cell_tmp[6].nb[21] = neighbors[4].children[1]
                cell_tmp[6].nb[22] = neighbors[4].children[0]
                cell_tmp[7].nb[20] = neighbors[4].children[1]
                cell_tmp[7].nb[21] = neighbors[4].children[0]
            else:
                cell_tmp[2].nb[12] = neighbors[4]
                cell_tmp[2].nb[13] = neighbors[4]
                cell_tmp[3].nb[11] = neighbors[4]
                cell_tmp[3].nb[12] = neighbors[4]
                cell_tmp[6].nb[21] = neighbors[4]
                cell_tmp[6].nb[22] = neighbors[4]
                cell_tmp[7].nb[20] = neighbors[4]
                cell_tmp[7].nb[21] = neighbors[4]

            if check[5]:
                cell_tmp[3].nb[13] = neighbors[5].children[5]
                cell_tmp[7].nb[22] = neighbors[5].children[1]
            else:
                cell_tmp[3].nb[13] = neighbors[5]
                cell_tmp[7].nb[22] = neighbors[5]

            if check[6]:
                cell_tmp[0].nb[13] = neighbors[6].children[6]
                cell_tmp[0].nb[14] = neighbors[6].children[5]
                cell_tmp[3].nb[14] = neighbors[6].children[6]
                cell_tmp[3].nb[15] = neighbors[6].children[5]
                cell_tmp[4].nb[22] = neighbors[6].children[2]
                cell_tmp[4].nb[23] = neighbors[6].children[1]
                cell_tmp[7].nb[23] = neighbors[6].children[2]
                cell_tmp[7].nb[24] = neighbors[6].children[1]
            else:
                cell_tmp[0].nb[13] = neighbors[6]
                cell_tmp[0].nb[14] = neighbors[6]
                cell_tmp[3].nb[14] = neighbors[6]
                cell_tmp[3].nb[15] = neighbors[6]
                cell_tmp[4].nb[22] = neighbors[6]
                cell_tmp[4].nb[23] = neighbors[6]
                cell_tmp[7].nb[23] = neighbors[6]
                cell_tmp[7].nb[24] = neighbors[6]

            if check[7]:
                cell_tmp[0].nb[15] = neighbors[7].children[6]
                cell_tmp[4].nb[24] = neighbors[7].children[2]
            else:
                cell_tmp[0].nb[15] = neighbors[7]
                cell_tmp[4].nb[24] = neighbors[7]

            if check[8]:
                cell_tmp[4].nb[8] = neighbors[8].children[3]
                cell_tmp[4].nb[9] = neighbors[8].children[2]
                cell_tmp[5].nb[8] = neighbors[8].children[2]
                cell_tmp[5].nb[15] = neighbors[8].children[3]
            else:
                cell_tmp[4].nb[8] = neighbors[8]
                cell_tmp[4].nb[9] = neighbors[8]
                cell_tmp[5].nb[8] = neighbors[8]
                cell_tmp[5].nb[15] = neighbors[8]

            if check[9]:
                cell_tmp[5].nb[9] = neighbors[9].children[3]
            else:
                cell_tmp[5].nb[9] = neighbors[9]

            if check[10]:
                cell_tmp[5].nb[10] = neighbors[10].children[0]
                cell_tmp[5].nb[11] = neighbors[10].children[3]
                cell_tmp[6].nb[9] = neighbors[10].children[0]
                cell_tmp[6].nb[10] = neighbors[10].children[3]
            else:
                cell_tmp[5].nb[10] = neighbors[10]
                cell_tmp[5].nb[11] = neighbors[10]
                cell_tmp[6].nb[9] = neighbors[10]
                cell_tmp[6].nb[10] = neighbors[10]

            if check[11]:
                cell_tmp[6].nb[11] = neighbors[11].children[0]
            else:
                cell_tmp[6].nb[11] = neighbors[11]

            if check[12]:
                cell_tmp[6].nb[12] = neighbors[12].children[1]
                cell_tmp[6].nb[13] = neighbors[12].children[0]
                cell_tmp[7].nb[11] = neighbors[12].children[1]
                cell_tmp[7].nb[12] = neighbors[12].children[0]
            else:
                cell_tmp[6].nb[12] = neighbors[12]
                cell_tmp[6].nb[13] = neighbors[12]
                cell_tmp[7].nb[11] = neighbors[12]
                cell_tmp[7].nb[12] = neighbors[12]

            if check[13]:
                cell_tmp[7].nb[13] = neighbors[13].children[1]
            else:
                cell_tmp[7].nb[13] = neighbors[13]

            if check[14]:
                cell_tmp[4].nb[13] = neighbors[14].children[2]
                cell_tmp[4].nb[14] = neighbors[14].children[1]
                cell_tmp[7].nb[14] = neighbors[14].children[2]
                cell_tmp[7].nb[15] = neighbors[14].children[1]
            else:
                cell_tmp[4].nb[13] = neighbors[14]
                cell_tmp[4].nb[14] = neighbors[14]
                cell_tmp[7].nb[14] = neighbors[14]
                cell_tmp[7].nb[15] = neighbors[14]

            if check[15]:
                cell_tmp[4].nb[15] = neighbors[15].children[2]
            else:
                cell_tmp[4].nb[15] = neighbors[15]

            if check[16]:
                cell_tmp[4].nb[10] = neighbors[16].children[1]
                cell_tmp[4].nb[11] = neighbors[16].children[2]
                cell_tmp[4].nb[12] = neighbors[16].children[3]
                cell_tmp[4].nb[16] = neighbors[16].children[0]
                cell_tmp[5].nb[12] = neighbors[16].children[2]
                cell_tmp[5].nb[13] = neighbors[16].children[3]
                cell_tmp[5].nb[14] = neighbors[16].children[0]
                cell_tmp[5].nb[16] = neighbors[16].children[1]
                cell_tmp[6].nb[8] = neighbors[16].children[1]
                cell_tmp[6].nb[14] = neighbors[16].children[3]
                cell_tmp[6].nb[15] = neighbors[16].children[0]
                cell_tmp[6].nb[16] = neighbors[16].children[2]
                cell_tmp[7].nb[8] = neighbors[16].children[0]
                cell_tmp[7].nb[9] = neighbors[16].children[1]
                cell_tmp[7].nb[10] = neighbors[16].children[2]
                cell_tmp[7].nb[16] = neighbors[16].children[3]
            else:
                cell_tmp[4].nb[10] = neighbors[16]
                cell_tmp[4].nb[11] = neighbors[16]
                cell_tmp[4].nb[12] = neighbors[16]
                cell_tmp[4].nb[16] = neighbors[16]
                cell_tmp[5].nb[12] = neighbors[16]
                cell_tmp[5].nb[13] = neighbors[16]
                cell_tmp[5].nb[14] = neighbors[16]
                cell_tmp[5].nb[16] = neighbors[16]
                cell_tmp[6].nb[8] = neighbors[16]
                cell_tmp[6].nb[14] = neighbors[16]
                cell_tmp[6].nb[15] = neighbors[16]
                cell_tmp[6].nb[16] = neighbors[16]
                cell_tmp[7].nb[8] = neighbors[16]
                cell_tmp[7].nb[9] = neighbors[16]
                cell_tmp[7].nb[10] = neighbors[16]
                cell_tmp[7].nb[16] = neighbors[16]

            if check[17]:
                cell_tmp[0].nb[17] = neighbors[17].children[7]
                cell_tmp[0].nb[18] = neighbors[17].children[6]
                cell_tmp[1].nb[17] = neighbors[17].children[6]
                cell_tmp[1].nb[24] = neighbors[17].children[7]
            else:
                cell_tmp[0].nb[17] = neighbors[17]
                cell_tmp[0].nb[18] = neighbors[17]
                cell_tmp[1].nb[17] = neighbors[17]
                cell_tmp[1].nb[24] = neighbors[17]

            if check[18]:
                cell_tmp[1].nb[18] = neighbors[18].children[7]
            else:
                cell_tmp[1].nb[18] = neighbors[18]

            if check[19]:
                cell_tmp[1].nb[19] = neighbors[19].children[4]
                cell_tmp[1].nb[20] = neighbors[19].children[5]
                cell_tmp[2].nb[18] = neighbors[19].children[4]
                cell_tmp[2].nb[19] = neighbors[19].children[7]
            else:
                cell_tmp[1].nb[19] = neighbors[19]
                cell_tmp[1].nb[20] = neighbors[19]
                cell_tmp[2].nb[18] = neighbors[19]
                cell_tmp[2].nb[19] = neighbors[19]

            if check[20]:
                cell_tmp[2].nb[20] = neighbors[20].children[4]
            else:
                cell_tmp[2].nb[20] = neighbors[20]

            if check[21]:
                cell_tmp[2].nb[21] = neighbors[21].children[5]
                cell_tmp[2].nb[22] = neighbors[21].children[4]
                cell_tmp[3].nb[20] = neighbors[21].children[5]
                cell_tmp[3].nb[21] = neighbors[21].children[4]
            else:
                cell_tmp[2].nb[21] = neighbors[21]
                cell_tmp[2].nb[22] = neighbors[21]
                cell_tmp[3].nb[20] = neighbors[21]
                cell_tmp[3].nb[21] = neighbors[21]

            if check[22]:
                cell_tmp[3].nb[22] = neighbors[22].children[5]
            else:
                cell_tmp[3].nb[22] = neighbors[22]

            if check[23]:
                cell_tmp[0].nb[22] = neighbors[23].children[6]
                cell_tmp[0].nb[23] = neighbors[23].children[5]
                cell_tmp[3].nb[23] = neighbors[23].children[6]
                cell_tmp[3].nb[24] = neighbors[23].children[5]
            else:
                cell_tmp[0].nb[22] = neighbors[23]
                cell_tmp[0].nb[23] = neighbors[23]
                cell_tmp[3].nb[23] = neighbors[23]
                cell_tmp[3].nb[24] = neighbors[23]

            if check[24]:
                cell_tmp[0].nb[24] = neighbors[24].children[6]
            else:
                cell_tmp[0].nb[24] = neighbors[24]

            if check[25]:
                cell_tmp[0].nb[19] = neighbors[25].children[5]
                cell_tmp[0].nb[20] = neighbors[25].children[6]
                cell_tmp[0].nb[21] = neighbors[25].children[7]
                cell_tmp[0].nb[25] = neighbors[25].children[4]
                cell_tmp[1].nb[21] = neighbors[25].children[6]
                cell_tmp[1].nb[22] = neighbors[25].children[7]
                cell_tmp[1].nb[23] = neighbors[25].children[4]
                cell_tmp[1].nb[25] = neighbors[25].children[5]
                cell_tmp[2].nb[17] = neighbors[25].children[5]
                cell_tmp[2].nb[23] = neighbors[25].children[7]
                cell_tmp[2].nb[24] = neighbors[25].children[4]
                cell_tmp[2].nb[25] = neighbors[25].children[6]
                cell_tmp[3].nb[17] = neighbors[25].children[4]
                cell_tmp[3].nb[18] = neighbors[25].children[5]
                cell_tmp[3].nb[19] = neighbors[25].children[6]
                cell_tmp[3].nb[25] = neighbors[25].children[7]
            else:
                cell_tmp[0].nb[19] = neighbors[25]
                cell_tmp[0].nb[20] = neighbors[25]
                cell_tmp[0].nb[21] = neighbors[25]
                cell_tmp[0].nb[25] = neighbors[25]
                cell_tmp[1].nb[21] = neighbors[25]
                cell_tmp[1].nb[22] = neighbors[25]
                cell_tmp[1].nb[23] = neighbors[25]
                cell_tmp[1].nb[25] = neighbors[25]
                cell_tmp[2].nb[17] = neighbors[25]
                cell_tmp[2].nb[23] = neighbors[25]
                cell_tmp[2].nb[24] = neighbors[25]
                cell_tmp[2].nb[25] = neighbors[25]
                cell_tmp[3].nb[17] = neighbors[25]
                cell_tmp[3].nb[18] = neighbors[25]
                cell_tmp[3].nb[19] = neighbors[25]
                cell_tmp[3].nb[25] = neighbors[25]

            # add the remaining nb
            cell_tmp[0].nb[10] = cell_tmp[5]
            cell_tmp[0].nb[11] = cell_tmp[6]
            cell_tmp[0].nb[12] = cell_tmp[7]
            cell_tmp[0].nb[16] = cell_tmp[4]
            cell_tmp[1].nb[12] = cell_tmp[6]
            cell_tmp[1].nb[13] = cell_tmp[7]
            cell_tmp[1].nb[14] = cell_tmp[0]
            cell_tmp[1].nb[16] = cell_tmp[5]
            cell_tmp[2].nb[8] = cell_tmp[5]
            cell_tmp[2].nb[14] = cell_tmp[7]
            cell_tmp[2].nb[15] = cell_tmp[4]
            cell_tmp[2].nb[16] = cell_tmp[6]
            cell_tmp[3].nb[8] = cell_tmp[4]
            cell_tmp[3].nb[9] = cell_tmp[5]
            cell_tmp[3].nb[10] = cell_tmp[6]
            cell_tmp[3].nb[16] = cell_tmp[7]
            cell_tmp[4].nb[19] = cell_tmp[1]
            cell_tmp[4].nb[20] = cell_tmp[2]
            cell_tmp[4].nb[21] = cell_tmp[3]
            cell_tmp[4].nb[25] = cell_tmp[0]
            cell_tmp[5].nb[21] = cell_tmp[2]
            cell_tmp[5].nb[22] = cell_tmp[3]
            cell_tmp[5].nb[23] = cell_tmp[0]
            cell_tmp[5].nb[25] = cell_tmp[1]
            cell_tmp[6].nb[17] = cell_tmp[1]
            cell_tmp[6].nb[23] = cell_tmp[3]
            cell_tmp[6].nb[24] = cell_tmp[0]
            cell_tmp[6].nb[25] = cell_tmp[2]
            cell_tmp[7].nb[17] = cell_tmp[0]
            cell_tmp[7].nb[18] = cell_tmp[1]
            cell_tmp[7].nb[19] = cell_tmp[2]
            cell_tmp[7].nb[25] = cell_tmp[3]

        return cell_tmp

    def _assign_indices(self, cells):
        """
        due to round-off errors, accumulation of errors etc. even with double precision and is_close(), we are
        interpreting the same node shared by adjacent cells as different node, so we need to generate the node indices
        without any coordinate information

        :param cells: Tuple containing the child cells
        :return: None
        """
        # TODO: documentation of this method & make more efficient

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
                    if self.check_nb_node(cells[i], child_no=3, nb_no=0):
                        cells[i].node_idx[1] = cells[i].parent.nb[0].children[3].node_idx[2]
                    else:
                        self.all_nodes.append(nodes[1, :])
                        cells[i].node_idx[1] = len(self.all_nodes) - 1

                    # lower nb
                    if self.check_nb_node(cells[i], child_no=1, nb_no=6):
                        cells[i].node_idx[3] = cells[i].parent.nb[6].children[1].node_idx[2]
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[3] = len(self.all_nodes) - 1

                    # the remaining node in the center off all children
                    self.all_nodes.append(nodes[2, :])
                    cells[i].node_idx[2] = len(self.all_nodes) - 1

                elif i == 1:
                    # upper nb
                    if self.check_nb_node(cells[i], child_no=0, nb_no=2):
                        cells[i].node_idx[2] = cells[i].parent.nb[2].children[0].node_idx[3]
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[2] = len(self.all_nodes) - 1
                    cells[i].node_idx[0] = cells[0].node_idx[1]
                    cells[i].node_idx[3] = cells[0].node_idx[2]

                elif i == 2:
                    # right nb
                    if self.check_nb_node(cells[i], child_no=1, nb_no=4):
                        cells[i].node_idx[3] = cells[i].parent.nb[4].children[1].node_idx[0]
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[3] = len(self.all_nodes) - 1
                    cells[i].node_idx[0] = cells[0].node_idx[2]
                    cells[i].node_idx[1] = cells[1].node_idx[2]

                elif i == 3:
                    # all new nodes are already introduced
                    cells[i].node_idx[0] = cells[0].node_idx[3]
                    cells[i].node_idx[1] = cells[0].node_idx[2]
                    cells[i].node_idx[2] = cells[2].node_idx[3]

            else:
                # child no. 0: new nodes 1, 2, 3, 4, 5, 6, 7 remain to add
                if i == 0:
                    # check left nb (same plane) if node 1 is present
                    if self.check_nb_node(cells[i], child_no=3, nb_no=0):
                        cells[i].node_idx[1] = cells[i].parent.nb[0].children[3].node_idx[2]
                    # check left nb (upper plane) if node 1 is present
                    elif self.check_nb_node(cells[i], child_no=7, nb_no=17):
                        cells[i].node_idx[1] = cells[i].parent.nb[17].children[7].node_idx[6]
                    # check nb above current cell (upper plane) if node 1 is present
                    elif self.check_nb_node(cells[i], child_no=4, nb_no=25):
                        cells[i].node_idx[1] = cells[i].parent.nb[25].children[4].node_idx[5]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[1, :])
                        cells[i].node_idx[1] = len(self.all_nodes) - 1

                    # check nb above current cell (upper plane) if node 2 is present
                    if self.check_nb_node(cells[i], child_no=4, nb_no=25):
                        cells[i].node_idx[2] = cells[i].parent.nb[25].children[4].node_idx[6]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[2] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 3 is present
                    if self.check_nb_node(cells[i], child_no=1, nb_no=6):
                        cells[i].node_idx[3] = cells[i].parent.nb[6].children[1].node_idx[2]
                    # check lower nb (upper plane) if node 3 is present
                    elif self.check_nb_node(cells[i], child_no=5, nb_no=23):
                        cells[i].node_idx[3] = cells[i].parent.nb[23].children[5].node_idx[6]
                    # check nb above current cell (upper plane) if node 2 is present
                    elif self.check_nb_node(cells[i], child_no=4, nb_no=25):
                        cells[i].node_idx[3] = cells[i].parent.nb[25].children[4].node_idx[7]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[3] = len(self.all_nodes) - 1

                    # check left nb (same plane) if node 4 is present
                    if self.check_nb_node(cells[i], child_no=3, nb_no=0):
                        cells[i].node_idx[4] = cells[i].parent.nb[0].children[3].node_idx[7]
                    # check lower left corner nb (same plane) if node 4 is present
                    elif self.check_nb_node(cells[i], child_no=2, nb_no=7):
                        cells[i].node_idx[4] = cells[i].parent.nb[7].children[2].node_idx[6]
                    # check lower nb (same plane) if node 4 is present
                    elif self.check_nb_node(cells[i], child_no=1, nb_no=6):
                        cells[i].node_idx[4] = cells[i].parent.nb[6].children[1].node_idx[5]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[4, :])
                        cells[i].node_idx[4] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 5 is present
                    if self.check_nb_node(cells[i], child_no=3, nb_no=0):
                        cells[i].node_idx[5] = cells[i].parent.nb[0].children[3].node_idx[6]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[5] = len(self.all_nodes) - 1

                    # node no. 6 is in center of all child cells, this node can't exist in other nb
                    self.all_nodes.append(nodes[6, :])
                    cells[i].node_idx[6] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=1, nb_no=6):
                        cells[i].node_idx[7] = cells[i].parent.nb[6].children[1].node_idx[6]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[7] = len(self.all_nodes) - 1

                elif i == 1:
                    # child no. 1: new nodes 2, 5, 6 remain to add
                    # check upper nb (same plane) if node 2 is present
                    if self.check_nb_node(cells[i], child_no=0, nb_no=2):
                        cells[i].node_idx[2] = cells[i].parent.nb[2].children[0].node_idx[3]
                    # check upper nb (upper plane) if node 2 is present
                    elif self.check_nb_node(cells[i], child_no=4, nb_no=19):
                        cells[i].node_idx[2] = cells[i].parent.nb[19].children[4].node_idx[7]
                    # check nb above current cell (upper plane) if node 2 is present
                    elif self.check_nb_node(cells[i], child_no=5, nb_no=25):
                        cells[i].node_idx[2] = cells[i].parent.nb[25].children[5].node_idx[6]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[2] = len(self.all_nodes) - 1

                    # check left nb (same plane) if node 5 is present
                    if self.check_nb_node(cells[i], child_no=2, nb_no=0):
                        cells[i].node_idx[5] = cells[i].parent.nb[0].children[2].node_idx[6]
                    # check upper left corner nb (same plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=3, nb_no=1):
                        cells[i].node_idx[5] = cells[i].parent.nb[1].children[3].node_idx[7]
                    # check upper nb (same plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=0, nb_no=2):
                        cells[i].node_idx[5] = cells[i].parent.nb[2].children[0].node_idx[4]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[5] = len(self.all_nodes) - 1

                    # check upper nb (same plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=0, nb_no=2):
                        cells[i].node_idx[6] = cells[i].parent.nb[2].children[0].node_idx[7]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[6] = len(self.all_nodes) - 1

                    cells[i].node_idx[0] = cells[0].node_idx[1]
                    cells[i].node_idx[3] = cells[0].node_idx[2]
                    cells[i].node_idx[4] = cells[0].node_idx[5]
                    cells[i].node_idx[7] = cells[0].node_idx[6]
                elif i == 2:
                    # child no. 2: new nodes 3, 6, 7 remain to add
                    # check right nb (same plane) if node 3 is present
                    if self.check_nb_node(cells[i], child_no=0, nb_no=4):
                        cells[i].node_idx[3] = cells[i].parent.nb[4].children[0].node_idx[1]
                    # check right nb (upper plane) if node 3 is present
                    elif self.check_nb_node(cells[i], child_no=4, nb_no=21):
                        cells[i].node_idx[3] = cells[i].parent.nb[21].children[4].node_idx[5]
                    # check nb above current cell (upper plane) if node 3 is present
                    elif self.check_nb_node(cells[i], child_no=7, nb_no=25):
                        cells[i].node_idx[3] = cells[i].parent.nb[25].children[7].node_idx[6]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[3] = len(self.all_nodes) - 1

                    # check right nb (same plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=1, nb_no=4):
                        cells[i].node_idx[6] = cells[i].parent.nb[4].children[1].node_idx[5]
                    # check upper right corner nb (same plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=0, nb_no=3):
                        cells[i].node_idx[6] = cells[i].parent.nb[3].children[0].node_idx[4]
                    # check upper nb (same plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=3, nb_no=2):
                        cells[i].node_idx[6] = cells[i].parent.nb[2].children[3].node_idx[7]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[6] = len(self.all_nodes) - 1

                    # check right nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=0, nb_no=4):
                        cells[i].node_idx[7] = cells[i].parent.nb[4].children[0].node_idx[5]
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[7] = len(self.all_nodes) - 1

                    cells[i].node_idx[0] = cells[0].node_idx[2]
                    cells[i].node_idx[1] = cells[1].node_idx[2]
                    cells[i].node_idx[4] = cells[0].node_idx[6]
                    cells[i].node_idx[5] = cells[1].node_idx[6]
                elif i == 3:
                    # child no. 3: new nodes 7 remain to add
                    # check right nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=0, nb_no=4):
                        cells[i].node_idx[7] = cells[i].parent.nb[4].children[0].node_idx[4]
                    # check lower right corner nb (same plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=1, nb_no=5):
                        cells[i].node_idx[7] = cells[i].parent.nb[5].children[1].node_idx[5]
                    # check lower nb (same plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=2, nb_no=6):
                        cells[i].node_idx[7] = cells[i].parent.nb[6].children[2].node_idx[6]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[7] = len(self.all_nodes) - 1

                    cells[i].node_idx[0] = cells[0].node_idx[3]
                    cells[i].node_idx[1] = cells[0].node_idx[2]
                    cells[i].node_idx[2] = cells[2].node_idx[3]
                    cells[i].node_idx[4] = cells[0].node_idx[7]
                    cells[i].node_idx[5] = cells[0].node_idx[6]
                    cells[i].node_idx[6] = cells[2].node_idx[7]
                elif i == 4:
                    # child no. 4: new nodes 5, 6, 7 remain to add
                    # check left nb (same plane) if node 5 is present
                    if self.check_nb_node(cells[i], child_no=7, nb_no=0):
                        cells[i].node_idx[5] = cells[i].parent.nb[0].children[7].node_idx[6]
                    # check left nb (lower plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=3, nb_no=8):
                        cells[i].node_idx[5] = cells[i].parent.nb[8].children[3].node_idx[2]
                    # check nb below current cell (lower plane) if node 5 is present
                    elif self.check_nb_node(cells[i], child_no=0, nb_no=16):
                        cells[i].node_idx[5] = cells[i].parent.nb[16].children[0].node_idx[1]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[5] = len(self.all_nodes) - 1

                    # check nb below current cell (lower plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=0, nb_no=16):
                        cells[i].node_idx[6] = cells[i].parent.nb[16].children[0].node_idx[2]
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[6] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=5, nb_no=6):
                        cells[i].node_idx[7] = cells[i].parent.nb[6].children[5].node_idx[6]
                    # check lower corner nb (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=1, nb_no=14):
                        cells[i].node_idx[7] = cells[i].parent.nb[14].children[1].node_idx[2]
                    # check nb below current cell (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=0, nb_no=16):
                        cells[i].node_idx[7] = cells[i].parent.nb[16].children[0].node_idx[3]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[7] = len(self.all_nodes) - 1

                    cells[i].node_idx[0] = cells[0].node_idx[4]
                    cells[i].node_idx[1] = cells[0].node_idx[5]
                    cells[i].node_idx[2] = cells[0].node_idx[6]
                    cells[i].node_idx[3] = cells[0].node_idx[7]
                elif i == 5:
                    # child no. 5: new nodes 6 remain to add
                    # check upper nb (same plane) if node 6 is present
                    if self.check_nb_node(cells[i], child_no=4, nb_no=2):
                        cells[i].node_idx[6] = cells[i].parent.nb[2].children[4].node_idx[7]
                    # check upper corner nb (lower plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=0, nb_no=10):
                        cells[i].node_idx[6] = cells[i].parent.nb[10].children[0].node_idx[3]
                    # check nb below current cell (lower plane) if node 6 is present
                    elif self.check_nb_node(cells[i], child_no=1, nb_no=16):
                        cells[i].node_idx[6] = cells[i].parent.nb[16].children[1].node_idx[2]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[6] = len(self.all_nodes) - 1

                    cells[i].node_idx[0] = cells[1].node_idx[4]
                    cells[i].node_idx[1] = cells[1].node_idx[5]
                    cells[i].node_idx[2] = cells[1].node_idx[6]
                    cells[i].node_idx[3] = cells[1].node_idx[7]
                    cells[i].node_idx[4] = cells[4].node_idx[5]
                    cells[i].node_idx[7] = cells[4].node_idx[6]
                elif i == 6:
                    # child no. 6: new nodes 7 to add
                    # check right nb (same plane) if node 7 is present
                    if self.check_nb_node(cells[i], child_no=4, nb_no=4):
                        cells[i].node_idx[7] = cells[i].parent.nb[4].children[4].node_idx[5]
                    # check right corner nb (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=0, nb_no=12):
                        cells[i].node_idx[7] = cells[i].parent.nb[12].children[0].node_idx[1]
                    # check nb below current cell (lower plane) if node 7 is present
                    elif self.check_nb_node(cells[i], child_no=2, nb_no=16):
                        cells[i].node_idx[7] = cells[i].parent.nb[16].children[2].node_idx[3]
                    # otherwise this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[7] = len(self.all_nodes) - 1

                    cells[i].node_idx[0] = cells[2].node_idx[4]
                    cells[i].node_idx[1] = cells[2].node_idx[5]
                    cells[i].node_idx[2] = cells[2].node_idx[6]
                    cells[i].node_idx[3] = cells[2].node_idx[7]
                    cells[i].node_idx[4] = cells[5].node_idx[7]
                    cells[i].node_idx[5] = cells[5].node_idx[6]
                elif i == 7:
                    # child no. 7 no new nodes anymore, only assign the existing nodes
                    cells[i].node_idx[0] = cells[3].node_idx[4]
                    cells[i].node_idx[1] = cells[3].node_idx[5]
                    cells[i].node_idx[2] = cells[3].node_idx[6]
                    cells[i].node_idx[3] = cells[3].node_idx[7]
                    cells[i].node_idx[4] = cells[4].node_idx[7]
                    cells[i].node_idx[5] = cells[4].node_idx[6]
                    cells[i].node_idx[6] = cells[6].node_idx[7]

    def check_nb_node(self, _cell, child_no, nb_no):
        return _cell.parent is not None and _cell.parent.nb[nb_no] is not None and \
               _cell.parent.nb[nb_no].children is not None and \
               len(_cell.parent.nb[nb_no].children) == pow(2, self._n_dimensions) and \
               _cell.parent.nb[nb_no].children[child_no] is not None and \
               _cell.parent.nb[nb_no].children[child_no].level == _cell.level


# @njit(fastmath=True, parallel=True)
def renumber_node_indices(all_idx: np.ndarray, all_nodes: np.ndarray, _unused_idx: list, dims: int):
    _unique_node_coord = np.zeros((all_nodes.shape[0] - len(_unused_idx), dims))
    _counter, _visited = 0, 0
    for i in prange(all_nodes.shape[0]):
        if i in _unused_idx:
            # decrement all idx which are > current index by 1, since we are deleting this node. but since we are
            # overwriting the all_idx tensor, we need to account for the entries we already deleted
            _visited += 1
            # TODO: these 2 lines not the same as 'all_idx[all_idx > i - _visited] -= 1' -> issue!,
            #  l. 1450 not woking with numba
            # indices_to_decrement = np.where(all_idx > i - _visited)[0]
            # all_idx[indices_to_decrement] -= 1
            all_idx[all_idx > i - _visited] -= 1
        else:
            _unique_node_coord[_counter, :] = all_nodes[i, :]
            _counter += 1
    return _unique_node_coord, all_idx


if __name__ == "__main__":
    pass
