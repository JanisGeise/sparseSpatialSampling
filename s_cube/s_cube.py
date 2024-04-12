"""
    implementation of the sparse spatial sampling algorithm (S^3) for 2D & 3D CFD data
"""
import torch as pt

from time import time
from sklearn.neighbors import KNeighborsRegressor


class Cell(object):
    """
    implements a cell for the KNN regressor
    """
    def __init__(self, index: int, parent, nb: list, center: pt.Tensor, level: int, children=None,
                 metric=None, gain=None, dimensions: int = 2):
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
        self._cells_per_iter_start = int(0.01 * vertices.size()[0])       # starting value = 1% of original grid size
        self._cells_per_iter_end = int(0.01 * self._cells_per_iter_start)        # end value = 1% of start value
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

        # offset matrix, used for computing the cell centers
        if self._n_dimensions == 2:
            self._directions = pt.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        else:
            self._directions = pt.tensor([[-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1],
                                          [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]])

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
        centers_[1:, :] += self._directions * 0.25 * self._width

        # make prediction
        metric = self._knn.predict(centers_.numpy()).squeeze()

        # gain of 1st cell
        sum_distances = sum([abs(metric[0] - metric[i]) for i in range(1, len(metric))])
        gain = pow(self._width / 2, 2) * sum_distances

        # add the first cell
        self._n_cells += 1

        # we have max. 8 neighbors for each cell in 2D case (4 at sides + 4 at corners), for 3D we get 9 more per level
        # -> 26 neighbors in total
        self._cells = [Cell(0, None, self._knn.n_neighbors * [None], centers_[0, :], 0, None, metric[0], gain,
                            dimensions=self._n_dimensions)]
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
                cell.children = tuple(assign_neighbors(loc_center, cell, neighbors, new_index, self._n_dimensions))

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
                cell.children = tuple(assign_neighbors(loc_center, cell, neighbors, new_index, self._n_dimensions))

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
        if self._smooth_geometry:
            self._refine_geometry()

        end_time = time()
        print("Finished refinement in {:2.4f} s ({:d} iterations).".format(end_time - start_time, iteration_count))
        print("Time for uniform refinement: {:2.4f} s".format(end_time_uniform - start_time))
        print("Time for adaptive refinement: {:2.4f} s".format(end_time - end_time_uniform))
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

    def compute_nodes_final_mesh(self) -> None:
        """
        compute the cell centers and vertices of the cells of the final grid

        :return: None
        """
        # loop over all leaf cells
        for cell in self._leaf_cells:
            # compute cell centers and store the nodes and faces for writing HDF5 file later
            self._cells[cell].nodes = self._compute_cell_centers(cell, factor_=0.5, keep_parent_center_=False)

    def _compute_cell_centers(self, idx_: int, factor_: float = 0.25, keep_parent_center_: bool = True) -> pt.Tensor:
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
        # empty tensor with size of (parent_cell + n_children, n_dims)
        coord_ = pt.zeros((pow(2, self._n_dimensions) + 1, self._n_dimensions))

        # fill with cell centers of parent & child cells, the first row corresponds to parent cell
        coord_[0, :] = self._cells[idx_].center

        # compute the cell centers of the children relative to the  parent cell center location
        # -> offset in each direction is +- 0.25 * cell_width (in case the cell center of each cell should be computed
        # it is 0.5 * cell_width)
        coord_[1:, :] = self._cells[idx_].center + self._directions * factor_ * self._width / pow(2, self._cells[idx_].level)

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
        stripped down version of the refine() method for refinement of the final grid near geometry objects or domain
        boundaries. The documentation of this method is equivalent to the refine() method
        """
        for _ in range(self._geometry_refinement_cycles):
            print("\nStarting geometry refinement:")
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
                cell.children = tuple(assign_neighbors(loc_center, cell, neighbors, new_index, self._n_dimensions))
                new_cells.extend(cell.children)
                self._n_cells += (pow(2, self._n_dimensions) - 1)
                self._current_max_level = max(self._current_max_level, cell.level + 1)
                new_index += pow(2, self._n_dimensions)

            self._cells.extend(new_cells)
            self._update_leaf_cells()
            self._update_gain()
            self._update_min_ref_level()
            self.remove_invalid_cells([c.index for c in new_cells])

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


def assign_neighbors(loc_center: pt.Tensor, cell: Cell, neighbors: list, new_idx: int, dimensions: int) -> list:
    """
    create a child cell from a given parent cell, assign its neighbors correctly to each child cell

    :param loc_center: center coordinates of the cell and its sub-cells
    :param cell: current cell
    :param neighbors: list containing all neighbors of current cell
    :param new_idx: new index of the cell
    :param dimensions: number of physical dimensions
    :return: the child cells with correctly assigned neighbors
    """
    # each cell gets new index within all cells
    cell_tmp = [Cell(new_idx + idx, cell, len(neighbors) * [None], loc, cell.level + 1, dimensions=dimensions) for
                idx, loc in enumerate(loc_center)]

    # add the neighbors for each of the child cells
    # neighbors for new upper left cell
    cell_tmp[0].nb[0] = neighbors[0]
    cell_tmp[0].nb[1] = neighbors[1]
    cell_tmp[0].nb[2] = neighbors[2]
    cell_tmp[0].nb[3] = neighbors[2]
    cell_tmp[0].nb[4] = cell_tmp[1]
    cell_tmp[0].nb[5] = cell_tmp[2]
    cell_tmp[0].nb[6] = cell_tmp[3]
    cell_tmp[0].nb[7] = neighbors[0]

    # neighbors for new upper right cell
    cell_tmp[1].nb[0] = cell_tmp[0]
    cell_tmp[1].nb[1] = neighbors[2]
    cell_tmp[1].nb[2] = neighbors[2]
    cell_tmp[1].nb[3] = neighbors[3]
    cell_tmp[1].nb[4] = neighbors[4]
    cell_tmp[1].nb[5] = neighbors[4]
    cell_tmp[1].nb[6] = cell_tmp[2]
    cell_tmp[1].nb[7] = cell_tmp[3]

    # neighbors for new lower right cell
    cell_tmp[2].nb[0] = cell_tmp[3]
    cell_tmp[2].nb[1] = cell_tmp[0]
    cell_tmp[2].nb[2] = cell_tmp[1]
    cell_tmp[2].nb[3] = neighbors[4]
    cell_tmp[2].nb[4] = neighbors[4]
    cell_tmp[2].nb[5] = neighbors[5]
    cell_tmp[2].nb[6] = neighbors[6]
    cell_tmp[2].nb[7] = neighbors[6]

    # neighbors for new lower left cell
    cell_tmp[3].nb[0] = neighbors[0]
    cell_tmp[3].nb[1] = neighbors[0]
    cell_tmp[3].nb[2] = cell_tmp[0]
    cell_tmp[3].nb[3] = cell_tmp[1]
    cell_tmp[3].nb[4] = cell_tmp[2]
    cell_tmp[3].nb[5] = neighbors[6]
    cell_tmp[3].nb[6] = neighbors[6]
    cell_tmp[3].nb[7] = neighbors[7]

    # if 2D, then we are done but for 3D, we need to add neighbors of upper and lower plane
    # same plane as current cell is always the same as for 2D
    if dimensions == 3:
        # ----------------  upper left cell center, upper plane  ----------------
        # plane under the current cell
        cell_tmp[0].nb[8] = neighbors[0]
        cell_tmp[0].nb[9] = neighbors[1]
        cell_tmp[0].nb[10] = neighbors[2]
        cell_tmp[0].nb[11] = neighbors[2]
        cell_tmp[0].nb[12] = cell_tmp[5]
        cell_tmp[0].nb[13] = cell_tmp[6]
        cell_tmp[0].nb[14] = cell_tmp[7]
        cell_tmp[0].nb[15] = neighbors[0]
        cell_tmp[0].nb[16] = neighbors[4]

        # plane above the current cell
        cell_tmp[0].nb[17] = neighbors[17]
        cell_tmp[0].nb[18] = neighbors[18]
        cell_tmp[0].nb[19] = neighbors[19]
        cell_tmp[0].nb[20] = neighbors[19]
        cell_tmp[0].nb[21] = neighbors[25]
        cell_tmp[0].nb[22] = neighbors[25]
        cell_tmp[0].nb[23] = neighbors[25]
        cell_tmp[0].nb[24] = neighbors[17]
        cell_tmp[0].nb[25] = neighbors[25]

        # ----------------  upper right cell center, upper plane  ----------------
        # plane under the current cell
        cell_tmp[1].nb[8] = cell_tmp[4]
        cell_tmp[1].nb[9] = neighbors[2]
        cell_tmp[1].nb[10] = neighbors[2]
        cell_tmp[1].nb[11] = neighbors[3]
        cell_tmp[1].nb[12] = neighbors[4]
        cell_tmp[1].nb[13] = neighbors[4]
        cell_tmp[1].nb[14] = cell_tmp[6]
        cell_tmp[1].nb[15] = cell_tmp[7]
        cell_tmp[1].nb[16] = cell_tmp[5]

        # plane above the current cell
        cell_tmp[1].nb[17] = neighbors[25]
        cell_tmp[1].nb[18] = neighbors[19]
        cell_tmp[1].nb[19] = neighbors[19]
        cell_tmp[1].nb[20] = neighbors[20]
        cell_tmp[1].nb[21] = neighbors[21]
        cell_tmp[1].nb[22] = neighbors[21]
        cell_tmp[1].nb[23] = neighbors[25]
        cell_tmp[1].nb[24] = neighbors[25]
        cell_tmp[1].nb[25] = neighbors[25]

        # ----------------  lower right cell center, upper plane  ----------------
        # plane under the current cell
        cell_tmp[2].nb[8] = cell_tmp[7]
        cell_tmp[2].nb[9] = cell_tmp[4]
        cell_tmp[2].nb[10] = cell_tmp[5]
        cell_tmp[2].nb[11] = neighbors[4]
        cell_tmp[2].nb[12] = neighbors[4]
        cell_tmp[2].nb[13] = neighbors[5]
        cell_tmp[2].nb[14] = neighbors[6]
        cell_tmp[2].nb[15] = neighbors[6]
        cell_tmp[2].nb[16] = cell_tmp[6]

        # plane above the current cell
        cell_tmp[2].nb[17] = neighbors[25]
        cell_tmp[2].nb[18] = neighbors[25]
        cell_tmp[2].nb[19] = neighbors[25]
        cell_tmp[2].nb[20] = neighbors[21]
        cell_tmp[2].nb[21] = neighbors[21]
        cell_tmp[2].nb[22] = neighbors[22]
        cell_tmp[2].nb[23] = neighbors[23]
        cell_tmp[2].nb[24] = neighbors[23]
        cell_tmp[2].nb[25] = neighbors[25]

        # ----------------  lower left cell center, upper plane  ----------------
        # plane under the current cell
        cell_tmp[3].nb[8] = neighbors[0]
        cell_tmp[3].nb[9] = neighbors[0]
        cell_tmp[3].nb[10] = cell_tmp[4]
        cell_tmp[3].nb[11] = cell_tmp[5]
        cell_tmp[3].nb[12] = cell_tmp[7]
        cell_tmp[3].nb[13] = neighbors[6]
        cell_tmp[3].nb[14] = neighbors[6]
        cell_tmp[3].nb[15] = neighbors[7]
        cell_tmp[3].nb[16] = cell_tmp[7]

        # plane above the current cell
        cell_tmp[3].nb[17] = neighbors[17]
        cell_tmp[3].nb[18] = neighbors[17]
        cell_tmp[3].nb[19] = neighbors[25]
        cell_tmp[3].nb[20] = neighbors[25]
        cell_tmp[3].nb[21] = neighbors[25]
        cell_tmp[3].nb[22] = neighbors[23]
        cell_tmp[3].nb[23] = neighbors[23]
        cell_tmp[3].nb[24] = neighbors[24]
        cell_tmp[3].nb[25] = neighbors[25]

        # ----------------  upper left cell center, lower plane  ----------------
        # plane under the current cell
        cell_tmp[4].nb[8] = neighbors[8]
        cell_tmp[4].nb[9] = neighbors[9]
        cell_tmp[4].nb[10] = neighbors[10]
        cell_tmp[4].nb[11] = neighbors[10]
        cell_tmp[4].nb[12] = neighbors[16]
        cell_tmp[4].nb[13] = neighbors[16]
        cell_tmp[4].nb[14] = neighbors[16]
        cell_tmp[4].nb[15] = neighbors[8]
        cell_tmp[4].nb[16] = neighbors[16]

        # plane above the current cell
        cell_tmp[4].nb[17] = neighbors[0]
        cell_tmp[4].nb[18] = neighbors[1]
        cell_tmp[4].nb[19] = neighbors[2]
        cell_tmp[4].nb[20] = neighbors[2]
        cell_tmp[4].nb[21] = cell_tmp[1]
        cell_tmp[4].nb[22] = cell_tmp[2]
        cell_tmp[4].nb[23] = cell_tmp[3]
        cell_tmp[4].nb[24] = neighbors[0]
        cell_tmp[4].nb[25] = cell_tmp[0]

        # ----------------  upper right cell center, lower plane  ----------------
        # plane under the current cell
        cell_tmp[5].nb[8] = neighbors[16]
        cell_tmp[5].nb[9] = neighbors[10]
        cell_tmp[5].nb[10] = neighbors[10]
        cell_tmp[5].nb[11] = neighbors[11]
        cell_tmp[5].nb[12] = neighbors[12]
        cell_tmp[5].nb[13] = neighbors[12]
        cell_tmp[5].nb[14] = neighbors[16]
        cell_tmp[5].nb[15] = neighbors[16]
        cell_tmp[5].nb[16] = neighbors[16]

        # plane above the current cell
        cell_tmp[5].nb[17] = cell_tmp[0]
        cell_tmp[5].nb[18] = neighbors[2]
        cell_tmp[5].nb[19] = neighbors[2]
        cell_tmp[5].nb[20] = neighbors[3]
        cell_tmp[5].nb[21] = neighbors[4]
        cell_tmp[5].nb[22] = neighbors[4]
        cell_tmp[5].nb[23] = cell_tmp[2]
        cell_tmp[5].nb[24] = cell_tmp[3]
        cell_tmp[5].nb[25] = cell_tmp[1]

        # ----------------  lower right cell center, lower plane  ----------------
        # plane under the current cell
        cell_tmp[6].nb[8] = neighbors[16]
        cell_tmp[6].nb[9] = neighbors[16]
        cell_tmp[6].nb[10] = neighbors[16]
        cell_tmp[6].nb[11] = neighbors[12]
        cell_tmp[6].nb[12] = neighbors[12]
        cell_tmp[6].nb[13] = neighbors[13]
        cell_tmp[6].nb[14] = neighbors[14]
        cell_tmp[6].nb[15] = neighbors[14]
        cell_tmp[6].nb[16] = neighbors[16]

        # plane above the current cell
        cell_tmp[6].nb[17] = cell_tmp[3]
        cell_tmp[6].nb[18] = cell_tmp[0]
        cell_tmp[6].nb[19] = cell_tmp[1]
        cell_tmp[6].nb[20] = neighbors[4]
        cell_tmp[6].nb[21] = neighbors[4]
        cell_tmp[6].nb[22] = neighbors[5]
        cell_tmp[6].nb[23] = neighbors[6]
        cell_tmp[6].nb[24] = neighbors[6]
        cell_tmp[6].nb[25] = cell_tmp[2]

        # ----------------  lower left cell center, lower plane  ----------------
        # plane under the current cell
        cell_tmp[7].nb[8] = neighbors[8]
        cell_tmp[7].nb[9] = neighbors[8]
        cell_tmp[7].nb[10] = neighbors[16]
        cell_tmp[7].nb[11] = neighbors[16]
        cell_tmp[7].nb[12] = neighbors[16]
        cell_tmp[7].nb[13] = neighbors[14]
        cell_tmp[7].nb[14] = neighbors[14]
        cell_tmp[7].nb[15] = neighbors[15]
        cell_tmp[7].nb[16] = neighbors[16]

        # plane above the current cell
        cell_tmp[7].nb[17] = neighbors[0]
        cell_tmp[7].nb[18] = neighbors[0]
        cell_tmp[7].nb[19] = cell_tmp[0]
        cell_tmp[7].nb[20] = cell_tmp[1]
        cell_tmp[7].nb[21] = cell_tmp[2]
        cell_tmp[7].nb[22] = neighbors[6]
        cell_tmp[7].nb[23] = neighbors[6]
        cell_tmp[7].nb[24] = neighbors[7]
        cell_tmp[7].nb[25] = cell_tmp[3]

    return cell_tmp


if __name__ == "__main__":
    pass
