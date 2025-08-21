"""
    implementation of the sparse spatial sampling algorithm (S^3) for 2D and 3D CFD data
"""
import logging
import numpy as np
import torch as pt

from time import time
from numba import njit, prange
from typing import Tuple, Union
from multiprocessing import get_context, cpu_count
from sklearn.neighbors import KNeighborsRegressor

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S',
                    force=True)

pt.set_default_dtype(pt.float64)

"""
Note:
        each cell has 8 neighbors (nb) in 2D case: 1 at each side, 1 at each corner of the cell. In 3D, there
        are additionally 9 neighbors of the plane above and below (in z-direction) present. The neighbors are
        assigned in clockwise direction starting at the left neighbor. For 3D, first the neighbors in the same plane are
        assigned, then the neighbors in lower plane and finally the neighbors in upper plane. The indices of the
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
           
        for readability, actual positions (indices) are replaced with cardinal directions as:
            n = north, e = east, s = south, w = west, l = lower plane, u = upper plane, c = center

        e.g., 'swu' := south-west neighbor cell in the plane above the current cell
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
        note that all cells are rectangular, meaning len_x == len_y = len_z; each cell has the following attributes:

        :param index: index of the cell (cell number)
        :param parent: the superior cell, from which this cell was derived
        :param nb: list containing all neighbor of the cell
        :param center: coordinates of the cell center
        :param level: refinement level (how often do we need to divide an initial cell to get to this cell)
        :param children: is this cell a parent cell of other cells (contains this cell other, smaller cells inside it)
        :param metric: the prediction for the metric made by the KNN based on the cell centers
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

    def leaf_cell(self) -> bool:
        return self.children is None


class SamplingTree(object):
    """
    class implementing the SamplingTree which creates the grid (tree structure) based on a given metric and coordinates
    """

    def __init__(self, vertices: pt.Tensor, target: pt.Tensor, geometry_obj: list, n_cells: int = None,
                 uniform_level: int = 5, min_metric: float = 0.75, max_delta_level: bool = False,
                 n_cells_iter_start: int = None, n_cells_iter_end: int = None, n_jobs: int = 1,
                 relTol: Union[int, float] = 1e-3, reach_at_least: float = 0.75, pre_select: bool = False):
        """
        initialize the KNNand settings, create an initial cell, which can be refined iteratively in the 'refine'-methods

        :param vertices: node coordinates of the original mesh (from CFD)
        :param target: the metric based on which the grid should be created, e.g., std. deviation of pressure wrt time
        :param geometry_obj: a list containing possible geometry objects, such as the numerical domain
        :param n_cells: max. number of cell, if 'None', then the refinement process stopps automatically once the
                        target metric is reached
        :param uniform_level: number of uniform refinement cycles to perform
        :param min_metric: percentage the metric the generated grid should capture (wrt the original grid), if 'None'
                            the max. number of cells will be used as stopping criteria
        :param max_delta_level: flag for setting the constraint that two adjacent cells should have a max. level
                                difference of one (important for computing gradients across cells)
        :param n_cells_iter_start: number of cells to refine per iteration at the beginning
        :param n_cells_iter_end: number of cells to refine per iteration at the end
        :param n_jobs: number of CPUs to use, if None all available CPUs will be used
        :param relTol: min. improvement between two consecutive iterations, defaults to 1e-3
        :param reach_at_least: reach at least per cent of the target metric / number of cells before activating the
                                relTol stopping criterion
        :param pre_select: when dealing with 'GeometrySTL3D' or 'GeometryCoordinates2D' geometry objects, this option
                           can decrease the required runtime significantly if the majority of cells is expected to be
                           created outside a rectangular bounding box around the geometry object,
                           i.e. if the difference between the volume of the geometry and a bounding box are minimal
        """
        # if '_min_metric' is not set, then use 'n_cells' as stopping criteria -> metric of 1 means we capture all
        # the dynamics in the original grid -> we should reach 'n_cells_max' earlier
        self._pre_select = pre_select
        self._n_jobs = n_jobs if n_jobs is not None else cpu_count()
        self._max_delta_level = max_delta_level
        self._vertices = vertices
        self._geometry = geometry_obj
        self._n_cells = 0
        self._min_metric = min_metric
        self._n_cells_max = n_cells
        self._min_level = uniform_level
        self._current_min_level = 0
        self._current_max_level = 0
        # starting value = 0.1% of original grid size
        self._cells_per_iter_start = int(0.001 * vertices.size(0)) if n_cells_iter_start is None else n_cells_iter_start
        # end value = same as start value
        self._cells_per_iter_end = self._cells_per_iter_start if n_cells_iter_end is None else n_cells_iter_end
        self._cells_per_iter = self._cells_per_iter_start
        self._cells_per_iter_last = 1e9
        self._reach_at_least = reach_at_least
        self._width = None
        self._pool = get_context("spawn").Pool(self._n_jobs)
        self._n_dimensions = self._vertices.size(-1)
        self._knn = KNeighborsRegressor(n_neighbors=8 if self._n_dimensions == 2 else 26, weights="distance",
                                        n_jobs=self._n_jobs)
        self._knn.fit(self._vertices, target)
        self._cells = None
        self._leaf_cells = set()
        self._n_cells_after_uniform = None
        self._N_cells_per_iter = []
        self.all_nodes = []
        self.all_centers = []
        self.face_ids = None
        self._metric = []
        self._n_cells_log = []
        self._n_cells_orig = target.size(0)
        self.data_final_mesh = {}
        self._times = initialize_time_dict()

        # relative tolerance for stopping criterion
        if relTol is None:
            # improve the metric by at least 0.1% or add at least 10 cells per iteration
            self._relTol = 1e-3 if n_cells is None else 10
        else:
            self._relTol = relTol

        # print settings to console for logging purposes
        self._print_settings()

        # offset matrix, used for computing the cell centers relative to the cell center of the parent cell
        if self._n_dimensions == 2:
            # 2D
            self._directions = pt.tensor([[-1, -1], [-1, 1], [1, 1], [1, -1]])
        else:
            # 3D, same order as for 2D
            self._directions = pt.tensor([[-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1],
                                          [-1, -1, -1], [-1, 1, -1], [1, 1, -1], [1, -1, -1]])

        # create initial cell and compute its gain
        self._create_first_cell()

        # remove the vertices of the original grid to free up memory since they are only required for fitting the KNN
        # and computing the dominant width of the domain
        del self._vertices

        # overwrite the metric with its L2-Norm, because the metric itself is not needed anymore (Frobenius norm here
        # same as L2, because the metric is always a vector)
        self._target_norm = pt.linalg.norm(target).item()

    def _update_gain(self, new_cells: Union[set, list]) -> None:
        """
        Update the gain of all (new) leaf cells, the gain is composed of the improvement wrt the target metric and the
        cell size.
        Within the first iterations, cells with a high metric are preferred for refinement while later the cell size is
        the main driver

        :param new_cells: Indices of cells for which the gain should be updated
        :return: None
        """
        # compute cell centers of all leaf cells and their potential children
        _centers = self._compute_cell_centers(new_cells)

        # predict the target function (metric) at the cell centers (interpolated based on the original grid)
        _metric = self._knn.predict(_centers.permute(-1, 0, 1).flatten(0, 1).numpy())
        _metric = pt.from_numpy(_metric).reshape(int(_metric.shape[0] / _centers.size(0)), _centers.size(0))

        # sum of the delta (target function) -> in which direction (left, right, ...) child changes
        # the target function the most relative to the original cell center to each newly created cell center
        sum_delta_metric = (_metric[:, [0]] - _metric[:, 1:]).abs().sum(dim=1)

        # args = (level, _n_dims, _width, gain_0, _metric, sum_delta_metric)
        args = [(self._cells[idx].level, self._n_dimensions, self._width, self._cells[0].gain.item(),
                 sum_delta_metric[i].item()) for i, idx in enumerate(new_cells)]
        _res = self._pool.map(_update_gain, args)

        for i, idx in enumerate(new_cells):
            # gain of the cell
            self._cells[idx].gain = _res[i]

            # metric at the cell center of the current cell
            self._cells[idx].metric = _metric[i, 0]

    def _update_leaf_cells(self, idx_parents: set, idx_children: set) -> None:
        """
        Update the leaf cells, make sure all parent cells are removed as leaf cell

        :return: None
        """
        self._leaf_cells -= idx_parents
        self._leaf_cells.update(idx_children)

    def _update_min_ref_level(self) -> None:
        """
        Update the current min. refinement level within the grid

        :return: None
        """
        min_level = min({self._cells[index].level for index in self._leaf_cells})
        self._current_min_level = max(self._current_min_level, min_level)

    def _check_stopping_criteria(self) -> bool:
        """
        Check if the stopping criteria for ending the refinement process is met; either based on the captured metric
        wrt to the original grid or the max. number of cells specified

        :return: bool
        """
        # check the fulfillment of the stopping criterion
        if self._n_cells_max is None:
            # only check stopping if we have captured at least self._reach_at_least % of the target metric (or cells)
            if len(self._metric) > 1 and self._metric[-1] / self._min_metric >= self._reach_at_least:
                return self._metric[-1] < self._min_metric and abs(self._metric[-1] - self._metric[-2]) > self._relTol
        else:
            if len(self._leaf_cells) / self._n_cells_max >= self._reach_at_least:
                _relStop = abs(self._cells_per_iter / self._n_cells_max - self._cells_per_iter_last / self._n_cells_max)
                return len(self._leaf_cells) < self._n_cells_max and _relStop > self._relTol

        # continue refinement if all criteria are not fulfilled
        return True

    def _compute_n_cells_per_iter(self) -> None:
        """
        Update the number of cells which should be refined within the next iteration. The end number is set to one,
        because in the last iteration we only want to refine a single cell to meet the defined metric for stopping as
        close as possible. The relationship is computed as linear approximation, the start and end values as well as the
        current value is known, this approach is either executed for the metric, or the number of cells, depending
        on the specified stopping criteria.

        :return: None
        """
        # if we use the target metric as stopping criteria, compute dx based on the current approximation of the metric
        if self._n_cells_max is None:
            _delta_x = self._min_metric - self._metric[0]
            _current_x = self._metric[-1]

        # else if we use the max. number of cells as stopping criteria, compute dx based on the cells
        else:
            _delta_x = self._n_cells_max - self._n_cells_after_uniform
            _current_x = self._n_cells

        # compute the dy based on the number of cells we defined as start and end
        _delta_y = self._cells_per_iter_start - self._cells_per_iter_end
        _new = self._cells_per_iter_start - (_delta_y / _delta_x) * _current_x

        # avoid negative updates or values of zeros
        self._cells_per_iter_last = self._cells_per_iter
        self._cells_per_iter = int(_new) if _new > 1 else 1

    def _compute_captured_metric(self) -> bool:
        """
        compute the metric at the cell centers captured by the current grid, relative to the metric of the original grid

        :return: bool, indicating if the current captured metric is larger than the min. metric defined as stopping
                 criteria
        """
        # the metric is computed at the cell centers for the original grid from CFD
        _centers = pt.stack([self._cells[cell].center for cell in self._leaf_cells])
        _current_metric = pt.from_numpy(self._knn.predict(_centers))

        # N_leaf_cells != N_cells_orig, so we need to use a norm. Target is saved as L2-Norm once the KNN is fitted, so
        # we don't need to compute it every iteration, since we have a vector, the Frobenius norm (used as default) is
        # the same as L2-norm
        _ratio = pt.linalg.norm(_current_metric) / self._target_norm

        self._metric.append(_ratio.item())
        return _ratio.item() < self._min_metric

    def _create_first_cell(self) -> None:
        """
        Creates a single cell based on the dominant dimension of the numerical domain, used as a starting point for the
        refinement process

        :return: None
        """
        # determine the main dimensions of the domain (cells are all rectangular, so len_x == len_y == len_z)
        self._width = max([(self._vertices[:, i].max() - self._vertices[:, i].min()).item() for i in
                           range(self._n_dimensions)])

        # compute the cell centers of the first cell and its child cells,
        # we can't use '_compute_cell_centers()', because we don't have any cells yet
        centers_ = pt.zeros((2 ** self._n_dimensions + 1, self._n_dimensions))

        # we can't use tensor operations to compute the centers_ tensor all at once, so we need to fill it element-wise
        for node in range(2 ** self._n_dimensions + 1):
            for d in range(self._n_dimensions):
                centers_[node, d] = (pt.max(self._vertices[:, d]).item() + pt.min(self._vertices[:, d]).item()) / 2.0

        # correct cell centers of children with offset -> in each direction +- 0.25 of cell width
        nodes = centers_[1:, :] + self._directions * 0.5 * self._width
        centers_[1:, :] += self._directions * 0.25 * self._width

        # interpolate the metric to the newly created cell centers
        metric = self._knn.predict(centers_.numpy()).squeeze()

        # compute the gain of the first cell
        sum_distances = sum([abs(metric[0] - metric[i]) for i in range(1, len(metric))])
        gain = pow(self._width / 2, 2) * sum_distances

        # add the first cell to the cell counter
        self._n_cells += 1

        # add the node coordinates to the list containing all coordinates
        for n in range(nodes.size()[0]):
            self.all_nodes.append(nodes[n, :])
        self.all_centers.append(centers_[0, :])

        # add the initial cell to the list of created cells
        self._cells = [Cell(0, None, self._knn.n_neighbors * [None], centers_[0, :], 0, None, metric[0], gain,
                            dimensions=self._n_dimensions, idx=list(range(nodes.size()[0])))]

        # update the leaf cells, so the initial cell is now seen as leaf cell
        self._leaf_cells.add(0)

    def _compute_cell_centers(self, _idx: Union[int, list, set] = None, _factor: float = 0.25,
                              _keep_parent_center: bool = True, _cell: Cell = None) -> pt.Tensor:
        """
        Computes either the cell centers of the child cells for a given parent cell ('factor_' = 0.25) or the nodes of
        a given cell ('factor_' = 0.5)

        Note:   although this method is called '_compute_cell_centers()', it computes the vertices of a cell if
                'factor_=0.5', not the cell center.
                If 'factor_=0.25', it computes the cell centers of all child cells, hence the name for this method.

        :param _idx: index of the cell
        :param _factor: the factor (0.5 = half distance between two adjacent cell centers = node; 0.25 = cell center of
                        child cells relative to the parent cell center)
        :param _keep_parent_center: if the cell center of the parent cell should be deleted from the tensor prior return
        :return: either cell centers of child cells (factor_=0.25) or nodes of current cell (factor_=0.5)
        """
        if _cell is None:
            _idx = [_idx] if type(_idx) is int else _idx
            centers = pt.cat(list(map(lambda i: self._cells[i].center.unsqueeze(-1), _idx)), -1)
            levels = pt.tensor(list(map(lambda i: self._cells[i].level, _idx))).int()
        else:
            centers = _cell.center.unsqueeze(-1)
            levels = _cell.level

        # allocate empty tensor with size of (parent_cell + n_children, n_dims, n_cells)
        _coord = pt.zeros(((2 ** self._n_dimensions) + 1, self._n_dimensions, centers.size(-1)))

        # fill with cell centers of parent and child cells, the first row corresponds to parent cell
        _coord[0, :, :] = centers

        # compute the cell centers of the children relative to the parent cell center location
        # -> offset in each direction is +- 0.25 * cell_width;
        # in case the cell center of each cell should be computed, it is 0.5 * cell_width
        _coord[1:, :, :] = centers + self._directions.unsqueeze(-1) * _factor * self._width / (2 ** levels)

        # remove the parent cell center if the flag is set (no parent cell center if we just want to compute the cell
        # centers of each cell, but we need the parent cell center, e.g., for computing the gain)
        return _coord[1:, :, :].squeeze(-1) if not _keep_parent_center else _coord.squeeze(-1)

    def _check_nb(self, _cell_no: int) -> list:
        """
        check if a cell and its neighbors have the same level, if not then we need to refine all cells for which this
        constraint is violated, because after refinement of the current cell, we would end up with a level difference
        of two

        :param _cell_no: current cell index for which we want to check the constraint
        :return: list with indices of nb cell we need to refine, because delta level is >= 1
        """
        # Check if the level of the current cell is the same as the one of all nb cells -> if not, then refine the nb
        # cell to avoid too large level differences between adjacent cells. The same level is required,
        # because at this point the child cells of the current cell are not yet created. If we allow a level difference
        # here, this would lead to a delta level of 2 once the children are created
        return [n.index for n in self._cells[_cell_no].nb if n is not None and n.leaf_cell() and
                n.level < self._cells[_cell_no].level]

    def _check_constraint(self, nb_violating_constraint: set) -> set:
        """
        Check if the constraint is violated when a nb cell of the current cell is refined for nb's of the nb cell. For
        example, consider:

            We want to refine a cell A with level i and the neu-cell B violates the constraint because it has the level
            i-1. However, the nb cell C of cell B has the level i-2 (which is still a delta level of one, relative to
            the nb of B). If we now add B to to_refine (because we want to refine cell A), the constraint between B & C
            would be violated. So we need to check the constraint for cell C as well, and if the constraint is violated
            for any nb of cell C, we need to check these nb as well and so on.

        :param nb_violating_constraint: NB cells of current cell, which violate the delta level constraint
        :return: all nb cells which need to be refined as well (basically all cells that need to be refined in the
                 current iteration)
        """
        new_cells_to_check = True if nb_violating_constraint else False
        while new_cells_to_check:
            # create an empty set for each iteration check
            tmp = set()

            # now go through all cells violating the constraint, which have been added so far and check their nb for
            # constraint violations
            for c in nb_violating_constraint:
                # update nb
                self._cells[c].parent.children = self._assign_neighbors(self._cells[c].parent,
                                                                        children=self._cells[c].parent.children)

                # check constraint
                tmp.update(self._check_nb(c))

            # check if we added new cells or if we checked this cell already if yes then we are done, if not then we
            # need to check the nb cells of the newly added cells as well
            if not tmp or tmp.issubset(nb_violating_constraint):
                new_cells_to_check = False
            else:
                nb_violating_constraint.update(tmp)
        return nb_violating_constraint

    def _refine_uniform(self) -> None:
        """
        Create uniform background mesh to save runtime, since the min. level of the generated grid is likely to be > 1
        (uniform refinement is significantly faster than adaptive refinement)

        :return: None
        """
        self._times["t_start_uniform"] = time()
        for j in range(self._min_level):
            logger.info(f"\r\tStarting iteration no. {j}, N_cells = {len(self._leaf_cells)}")
            new_cells = []
            new_index, all_parents, all_children = len(self._cells), set(), set()

            # compute cell centers
            if j == 0:
                loc_center = self._compute_cell_centers(self._leaf_cells, _keep_parent_center=False).unsqueeze(-1)
            else:
                loc_center = self._compute_cell_centers(self._leaf_cells, _keep_parent_center=False)

            for idx, i in enumerate(self._leaf_cells):
                cell = self._cells[i]

                # assign the neighbors of current cell and add all new cells as children
                cell.children = list(self._assign_neighbors(cell, loc_center[:, :, idx], new_index))
                all_children.update(list(range(new_index, new_index + len(cell.children))))
                all_parents.add(cell.index)

                # assign idx for the newly created nodes
                self._assign_indices(cell.children)

                new_cells.extend(cell.children)
                self._n_cells += 2 ** self._n_dimensions
                new_index += 2 ** self._n_dimensions

            # update all nb
            for cell in range(len(new_cells)):
                new_cells[cell].parent.children = self._assign_neighbors(new_cells[cell].parent,
                                                                         children=new_cells[cell].parent.children)

            self._cells.extend(new_cells)
            self._update_leaf_cells(all_parents, all_children)
            self._update_gain(all_children)
            self._current_min_level += 1
            self._current_max_level += 1

            # delete cells which are outside the domain or inside a geometry (here we update all cells every iteration)
            self._remove_invalid_cells({c.index for c in new_cells})

        logger.info(f"Finished uniform refinement.")
        self._times["t_end_uniform"] = time()

    def refine(self) -> None:
        """
        Implements the generation of the grid based on the original grid and a metric

        :return: None
        """
        logger.info("Starting refinement:")
        # execute uniform refinement, the stopping criteria requires at least one uniform refinement cycle which is
        # checked and set beforehand
        self._refine_uniform()

        # compute the initial metric if selected as a stopping criterion
        iteration_count = 0
        self._n_cells_after_uniform = len(self._leaf_cells)
        if self._n_cells_max is None:
            self._compute_captured_metric()

        # append the current N_cells for logging purposes
        self._n_cells_log.append(len(self._leaf_cells))

        # start the adaptive refinement
        logger.info("Starting adaptive refinement.")
        self._times["t_start_adaptive"] = time()

        while self._check_stopping_criteria():
            if self._n_cells_max is None:
                logger.info(f"\r\tStarting iteration no. {iteration_count}, captured metric: "
                            f"{round(self._metric[-1] * 100, 2)} %, N_cells = {len(self._leaf_cells)}")
            else:
                logger.info(f"\r\tStarting iteration no. {iteration_count}, N_cells = {len(self._leaf_cells)}")

            # update _n_cells_per_iter based on the difference wrt metric or N_cells
            if len(self._metric) >= 2:
                self._compute_n_cells_per_iter()

            _leaf_cells_sorted = sorted(list(self._leaf_cells), key=lambda x: self._cells[x].gain, reverse=True)
            to_refine = set()
            for i in _leaf_cells_sorted[:min(self._cells_per_iter, self._n_cells)]:
                cell = self._cells[i]
                to_refine.add(cell.index)

                # update nb of current cell
                cell.parent.children = self._assign_neighbors(cell.parent, children=cell.parent.children)

                # in case the delta level constraint is active, we need to check if the nb cells have the same level
                if self._max_delta_level:
                    # check if the nb cells have the same level
                    nb_to_refine_as_well = set(self._check_nb(i))

                    # then we need to check each nb of the nb cell, in case the nb was added to to_refine to avoid
                    # constraint violations between a nb and its nb cells
                    to_refine.update(self._check_constraint(nb_to_refine_as_well))

            new_cells, all_parents, all_children = [], set(), set()
            new_index = len(self._cells)

            # compute cell centers
            loc_center = self._compute_cell_centers(to_refine, _keep_parent_center=False)

            for idx, i in enumerate(to_refine):
                cell = self._cells[i]

                # assign the neighbors of current cell and add all new cells as children
                cell.children = tuple(self._assign_neighbors(cell, loc_center[:, :, idx], new_index))
                all_children.update(list(range(new_index, new_index + len(cell.children))))
                all_parents.add(cell.index)

                # assign idx for the newly created nodes
                self._assign_indices(cell.children)

                new_cells.extend(cell.children)
                self._n_cells += 2 ** self._n_dimensions
                self._current_max_level = max(self._current_max_level, cell.level + 1)

                # for each cell in to_refine, we added 4 cells in 2D (2 cells in 1D), 8 cells in 3D
                new_index += 2 ** self._n_dimensions

            # add the newly generated cell to the list of all existing cells and update everything
            self._cells.extend(new_cells)
            self._update_leaf_cells(all_parents, all_children)
            self._update_gain(all_children)

            # check the newly generated cells if they are outside the domain or inside a geometry, if so, delete them
            self._remove_invalid_cells({c.index for c in new_cells})

            # compute global gain after refinement to check if we can stop the refinement
            # if the metric is selected as a stopping criterion
            if self._n_cells_max is None:
                self._compute_captured_metric()
            iteration_count += 1

            # append the current N_cells for logging purposes
            self._n_cells_log.append(len(self._leaf_cells))

        try:
            del _leaf_cells_sorted
        except UnboundLocalError:
            pass

        # if we use N_cells_max as a stopping criterion, then compute the captured metric only once here
        if self._n_cells_max is not None:
            self._compute_captured_metric()

        # refine the grid near geometry objects is specified
        logger.info("Finished adaptive refinement.")

        # refine geometries if specified
        self._refine_geometries()

        # close the multiprocessing pool
        self._pool.close()

        # update the min. refinement level for logging info and grid statistics
        self._update_min_ref_level()

        # assemble the final grid
        self._resort_nodes_and_indices_of_grid()

        # save and print timings and size of final mesh
        self._create_mesh_info(iteration_count)

        # print info
        logger.info(self)

    def _remove_invalid_cells(self, _refined_cells: set, _refine_geometry: bool = False,
                              _geometry_no: Union[int, list] = None) -> Union[None, set]:
        """
        Check if any of the generated cells are located inside a geometry or outside a domain, if so they are removed.

        :param _refined_cells: indices of the cells which are newly added
        :param _refine_geometry: flag if we want to refine the grid near geometry objects
        :param _geometry_no: index of the geometry object to check (or to refine) within the self._geometries
        :return: None if we removed all invalid cells, if '_refine_geometry = True' returns list with indices of the
                 cells which are neighbors of geometries / domain boundaries
        """
        # in case we only have a single geometry, we need to cast it to a list
        if type(_geometry_no) is int:
            _geometry_no = [_geometry_no]

        # determine for which geometries we should check -> important for geometry refinement at the end
        _geometries = [self._geometry[g] for g in _geometry_no] if _geometry_no is not None else self._geometry

        # compute the node locations of the current cell
        nodes = self._compute_cell_centers(_refined_cells, _factor=0.5, _keep_parent_center=False)

        # loop over all new cells and check if they are valid
        args_list = [(cell, nodes[:, :, i], _geometries, _refine_geometry, self._pre_select)
                     for i, cell in enumerate(_refined_cells)]
        result = self._pool.map(_check_cell_validity, args_list)

        _idx = set(filter(None, result))
        del result, args_list

        # if we didn't find any invalid cells, we are done here, else we need to reset the nb of the cells affected by
        # the removal of the masked cells
        if _idx == set():
            return None
        elif _refine_geometry:
            return _idx
        else:
            # for each invalid cell, loop over their neighbors, find the invalid cell and deactivate it as a neighbor
            for cell in _idx:
                self._cells[cell].children = None if _refine_geometry else []
                self._cells[cell].gain = 0

                for nb in self._cells[cell].nb:
                    if nb is not None:
                        self._cells[nb.index].nb = [None if nb.nb[n] is not None and nb.nb[n].index == cell else
                                                    nb.nb[n] for n in range(len(nb.nb))]

                # TODO: can we safely remove the cell center and node idx completely from the tree?
                #       no, because this would require re-indexing of leaf_cells
                #       -> applies for removal of unused cells as well (to save some memory)
                #       -> general steps would be:
                #           1. determination of unused cells
                #           2. complete removal from tree
                #           3. removal of corresponding cell centers and nodes
                #           4. re-indexing of leaf_cells
                #           5. re-indexing of cell faces / face id's (?)

            # TODO: for testing purposes start with removal of invalid cell, if that works try unused parent cells

            # remove all invalid cells from the leaf cells
            self._leaf_cells -= _idx

    def _resort_nodes_and_indices_of_grid(self) -> None:
        """
        Remove all invalid and parent cells from the mesh. Sort the cell centers and vertices of the final grid with
        respect to their corresponding index and re-number all nodes.

        :return: None
        """
        logger.info("Starting renumbering final mesh.")
        self._times["t_start_renumber"] = time()
        dtype = pt.int32 if self._n_cells < pt.iinfo(pt.int32).max else pt.int64

        _all_idx = pt.tensor([cell.node_idx for cell in self._cells if cell.leaf_cell()], dtype=dtype)
        _unique_idx = _all_idx.flatten().unique()

        # the initial cell is not in the list, so add it manually
        _idx_initial_cell = pt.arange(0, 2 ** self._n_dimensions)

        # add the remaining indices to get all available indices present in the current grid
        _all_available_idx = pt.cat([_idx_initial_cell, pt.arange(_all_idx.min().item(), _all_idx.max().item() + 1,
                                                                  dtype=dtype)])

        # get all node indices that are not used by all cells anymore
        _unused_idx = set(_all_available_idx[~pt.isin(_all_available_idx, _unique_idx)].unique().to(dtype=dtype).numpy())
        del _unique_idx, _all_available_idx, _idx_initial_cell

        # re-index using numba -> faster than python, and this step is computationally quite expensive
        _unique_node_coord, _all_idx = renumber_node_indices_parallel(_all_idx.numpy(), pt.stack(self.all_nodes).numpy(),
                                                                      _unused_idx, self._n_dimensions)
        del _unused_idx

        # update node ID's and their coordinates
        self.face_ids = pt.from_numpy(_all_idx)
        self.all_nodes = pt.from_numpy(_unique_node_coord)
        self.all_centers = pt.stack([self._cells[cell].center for cell in self._leaf_cells])
        self.all_levels = pt.tensor([self._cells[cell].level for cell in self._leaf_cells]).unsqueeze(-1)
        self._times["t_end_renumber"] = time()

    def _execute_geometry_refinement(self, _geometries: list = None) -> None:
        """
        an adapted version of the 'refine()' method for refinement of the final grid near geometry objects or
        domain boundaries.
        The documentation of this method is equivalent to the 'refine()' method, but with some extensions:
            - loop over all geometries we want to refine
            - determine the level with which we want to refine the geometry object
            - determine the cells in the vicinity of the geometry
            - refine the geometry analogously to the refine() method

        :param _geometries: list containing the indices of the geometry objects we want to refine
        :return: None
        """
        logger.info("Starting geometry refinement.")

        # To save some time, only go through all leaf cells at the beginning. For later iterations, we will use only the
        # newly created cells
        for g in _geometries:
            logger.info(f"Starting refining geometry {self._geometry[g].name}.")
            _t_start = time()
            try:
                # it may happen that we didn't find any cells near geometries.
                # In that case _remove_invalid_cells will return 'None', so we skip the refinement
                _all_cells = set(self._remove_invalid_cells(self._leaf_cells, _refine_geometry=True, _geometry_no=g))
            except TypeError:
                logger.warning("Could not find any cells to refine. Skipping geometry refinement.")
                logger.info("Finished geometry refinement.")
                return

            _global_min_level = min([self._cells[cell].level for cell in _all_cells])

            # determine the max. refinement level for the geometries:
            if self._geometry[g].min_refinement_level is None:
                _global_max_level = max([self._cells[cell].level for cell in _all_cells])
            else:
                # global max level, bc min. level is the max level we want to reach
                _global_max_level = self._geometry[g].min_refinement_level

            while _global_max_level > _global_min_level:
                to_refine, checked = set(), set()

                for i in _all_cells:
                    # if we already refined this cell due to delta level constraint, then we don't need to refine it
                    # again
                    if i in checked:
                        continue

                    # otherwise refine the cell
                    cell = self._cells[i]

                    # don't refine which have reached the max. level
                    if cell.level < _global_max_level:
                        to_refine.add(cell.index)
                        cell.parent.children = self._assign_neighbors(cell.parent, children=cell.parent.children)

                    # but still check the constraint for the nb cells if specified
                    if self._max_delta_level:
                        # check if the nb cells have the same level
                        nb_to_refine_as_well = set(self._check_nb(i))
                        nb_to_refine_as_well.update(self._check_constraint(nb_to_refine_as_well))
                        to_refine.update(nb_to_refine_as_well)
                        checked.update(nb_to_refine_as_well)

                new_cells, all_parents, all_children = [], set(), set()
                new_index = len(self._cells)
                for i in to_refine:
                    cell = self._cells[i]
                    loc_center = self._compute_cell_centers(i, _keep_parent_center=False)
                    cell.children = tuple(self._assign_neighbors(cell, loc_center, new_index))
                    all_children.update(list(range(new_index, new_index + len(cell.children))))
                    all_parents.add(cell.index)
                    self._assign_indices(cell.children)
                    new_cells.extend(cell.children)
                    self._n_cells += (2 ** self._n_dimensions - 1)
                    new_index += 2 ** self._n_dimensions

                self._cells.extend(new_cells)
                self._update_leaf_cells(all_parents, all_children)
                self._update_gain(all_children)
                _idx_new = {c.index for c in new_cells}
                self._remove_invalid_cells(_idx_new, _geometry_no=g)

                # update '_all_cells' after every iteration, we need to check again which of the refined cells are in
                # the vicinity of the geometry; we use only valid cells because otherwise we would mark the invalid
                # cells as leaf cells (because kwarg _refine_geometry)
                _all_cells = set(self._remove_invalid_cells({i for i in _idx_new if self._cells[i].children is None},
                                                            _refine_geometry=True, _geometry_no=g))

                # update the min. level
                _global_min_level += 1

        # update the global max. level
        self._current_max_level = max({self._cells[cell].level for cell in self._leaf_cells})
        logger.info("Finished geometry refinement.")

    def _assign_neighbors(self, cell: Cell, loc_center: pt.Tensor = None, new_idx: int = None,
                          children: Union[Tuple, list] = None) -> list:
        """
        create a child cell from a given parent cell, assign its neighbors correctly to each child cell

        :param loc_center: center coordinates of the cell and its subcells
        :param cell: current cell
        :param new_idx: new index of the cell
        :return: the child cells with correctly assigned neighbors
        """
        if children is None:
            # each child cell gets a new index within all cells
            children = [Cell(new_idx + i, cell, len(cell.nb) * [None], loc_center[i, :], cell.level + 1,
                             dimensions=self._n_dimensions) for i in range(loc_center.size(0))]

        # Add the neighbors for each of the child cells
        # neighbors for new lower left cell; we need to check for children, because we only assigned the parent cell,
        # which is ambiguous for the non-cornering neighbors. Further, we need to make sure that we have exactly N
        # children (if cells are removed due to geometry issues, then the children are an empty list)
        check = [True if n is not None and n.children is not None and n.children else False for n in cell.nb]

        # lower-left child, same plane (= south-west upper))
        children[CH["swu"]].nb[NB["w"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["seu"])
        children[CH["swu"]].nb[NB["nw"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["neu"])
        children[CH["swu"]].nb[NB["n"]] = children[CH["nwu"]]
        children[CH["swu"]].nb[NB["ne"]] = children[CH["neu"]]
        children[CH["swu"]].nb[NB["e"]] = children[CH["seu"]]
        children[CH["swu"]].nb[NB["se"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["neu"])
        children[CH["swu"]].nb[NB["s"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwu"])
        children[CH["swu"]].nb[NB["sw"]] = parent_or_child(cell.nb, check[NB["sw"]], NB["sw"], CH["neu"])

        # upper-left child, same plane (= north-west upper))
        children[CH["nwu"]].nb[NB["w"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["neu"])
        children[CH["nwu"]].nb[NB["nw"]] = parent_or_child(cell.nb, check[NB["nw"]], NB["nw"], CH["seu"])
        children[CH["nwu"]].nb[NB["n"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swu"])
        children[CH["nwu"]].nb[NB["ne"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["seu"])
        children[CH["nwu"]].nb[NB["e"]] = children[CH["neu"]]
        children[CH["nwu"]].nb[NB["se"]] = children[CH["seu"]]
        children[CH["nwu"]].nb[NB["s"]] = children[CH["swu"]]
        children[CH["nwu"]].nb[NB["sw"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["seu"])

        # upper-right child, same plane (= north-east upper))
        children[CH["neu"]].nb[NB["w"]] = children[CH["nwu"]]
        children[CH["neu"]].nb[NB["nw"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swu"])
        children[CH["neu"]].nb[NB["n"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["seu"])
        children[CH["neu"]].nb[NB["ne"]] = parent_or_child(cell.nb, check[NB["ne"]], NB["ne"], CH["swu"])
        children[CH["neu"]].nb[NB["e"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwu"])
        children[CH["neu"]].nb[NB["se"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swu"])
        children[CH["neu"]].nb[NB["s"]] = children[CH["seu"]]
        children[CH["neu"]].nb[NB["sw"]] = children[CH["swu"]]

        # lower-right child, same plane (= south-east upper))
        children[CH["seu"]].nb[NB["w"]] = children[CH["swu"]]
        children[CH["seu"]].nb[NB["nw"]] = children[CH["nwu"]]
        children[CH["seu"]].nb[NB["n"]] = children[CH["neu"]]
        children[CH["seu"]].nb[NB["ne"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwu"])
        children[CH["seu"]].nb[NB["e"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swu"])
        children[CH["seu"]].nb[NB["se"]] = parent_or_child(cell.nb, check[NB["se"]], NB["se"], CH["nwu"])
        children[CH["seu"]].nb[NB["s"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["neu"])
        children[CH["seu"]].nb[NB["sw"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwu"])

        # if 2D, then we are done, but for 3D, we need to add neighbors of upper and lower plane
        # same plane as current cell is always the same as for 2D
        if self._n_dimensions == 3:
            # lower-left child, lower plane (= south-west lower)
            children[CH["swl"]].nb[NB["w"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["sel"])
            children[CH["swl"]].nb[NB["nw"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["nel"])
            children[CH["swl"]].nb[NB["n"]] = children[CH["nwl"]]
            children[CH["swl"]].nb[NB["ne"]] = children[CH["nel"]]
            children[CH["swl"]].nb[NB["e"]] = children[CH["sel"]]
            children[CH["swl"]].nb[NB["se"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nel"])
            children[CH["swl"]].nb[NB["s"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwl"])
            children[CH["swl"]].nb[NB["sw"]] = parent_or_child(cell.nb, check[NB["sw"]], NB["sw"], CH["nel"])

            children[CH["swl"]].nb[NB["wl"]] = parent_or_child(cell.nb, check[NB["wl"]], NB["wl"], CH["seu"])
            children[CH["swl"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["wl"]], NB["wl"], CH["neu"])
            children[CH["swl"]].nb[NB["nl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["nwu"])
            children[CH["swl"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["neu"])
            children[CH["swl"]].nb[NB["el"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["seu"])
            children[CH["swl"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["sl"]], NB["sl"], CH["neu"])
            children[CH["swl"]].nb[NB["sl"]] = parent_or_child(cell.nb, check[NB["sl"]], NB["sl"], CH["nwu"])
            children[CH["swl"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["swl"]], NB["swl"], CH["neu"])
            children[CH["swl"]].nb[NB["cl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["swu"])

            children[CH["swl"]].nb[NB["wu"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["seu"])
            children[CH["swl"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["neu"])
            children[CH["swl"]].nb[NB["nu"]] = children[CH["nwu"]]
            children[CH["swl"]].nb[NB["neu"]] = children[CH["neu"]]
            children[CH["swl"]].nb[NB["eu"]] = children[CH["seu"]]
            children[CH["swl"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["neu"])
            children[CH["swl"]].nb[NB["su"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwu"])
            children[CH["swl"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["sw"]], NB["sw"], CH["neu"])
            children[CH["swl"]].nb[NB["cu"]] = children[CH["swu"]]

            # upper-left child, lower plane (= north-west lower)
            children[CH["nwl"]].nb[NB["w"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["nel"])
            children[CH["nwl"]].nb[NB["nw"]] = parent_or_child(cell.nb, check[NB["nw"]], NB["nw"], CH["sel"])
            children[CH["nwl"]].nb[NB["n"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swl"])
            children[CH["nwl"]].nb[NB["ne"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["sel"])
            children[CH["nwl"]].nb[NB["e"]] = children[CH["nel"]]
            children[CH["nwl"]].nb[NB["se"]] = children[CH["sel"]]
            children[CH["nwl"]].nb[NB["s"]] = children[CH["swl"]]
            children[CH["nwl"]].nb[NB["sw"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["sel"])

            children[CH["nwl"]].nb[NB["wl"]] = parent_or_child(cell.nb, check[NB["wl"]], NB["wl"], CH["neu"])
            children[CH["nwl"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["nwl"]], NB["nwl"], CH["seu"])
            children[CH["nwl"]].nb[NB["nl"]] = parent_or_child(cell.nb, check[NB["nl"]], NB["nl"], CH["swu"])
            children[CH["nwl"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["nl"]], NB["nl"], CH["seu"])
            children[CH["nwl"]].nb[NB["el"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["neu"])
            children[CH["nwl"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["seu"])
            children[CH["nwl"]].nb[NB["sl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["swu"])
            children[CH["nwl"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["wl"]], NB["wl"], CH["seu"])
            children[CH["nwl"]].nb[NB["cl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["nwu"])

            children[CH["nwl"]].nb[NB["wu"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["neu"])
            children[CH["nwl"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["nw"]], NB["nw"], CH["seu"])
            children[CH["nwl"]].nb[NB["nu"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swu"])
            children[CH["nwl"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["seu"])
            children[CH["nwl"]].nb[NB["eu"]] = children[CH["neu"]]
            children[CH["nwl"]].nb[NB["seu"]] = children[CH["seu"]]
            children[CH["nwl"]].nb[NB["su"]] = children[CH["swu"]]
            children[CH["nwl"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["seu"])
            children[CH["nwl"]].nb[NB["cu"]] = children[CH["nwu"]]

            # upper-right child, lower plane (= north-east lower)
            children[CH["nel"]].nb[NB["w"]] = children[CH["nwl"]]
            children[CH["nel"]].nb[NB["nw"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swl"])
            children[CH["nel"]].nb[NB["n"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["sel"])
            children[CH["nel"]].nb[NB["ne"]] = parent_or_child(cell.nb, check[NB["ne"]], NB["ne"], CH["swl"])
            children[CH["nel"]].nb[NB["e"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwl"])
            children[CH["nel"]].nb[NB["se"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swl"])
            children[CH["nel"]].nb[NB["s"]] = children[CH["sel"]]
            children[CH["nel"]].nb[NB["sw"]] = children[CH["swl"]]

            children[CH["nel"]].nb[NB["wl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["nwu"])
            children[CH["nel"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["nl"]], NB["nl"], CH["swu"])
            children[CH["nel"]].nb[NB["nl"]] = parent_or_child(cell.nb, check[NB["nl"]], NB["nl"], CH["seu"])
            children[CH["nel"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["nel"]], NB["nel"], CH["swu"])
            children[CH["nel"]].nb[NB["el"]] = parent_or_child(cell.nb, check[NB["el"]], NB["el"], CH["nwu"])
            children[CH["nel"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["el"]], NB["el"], CH["swu"])
            children[CH["nel"]].nb[NB["sl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["seu"])
            children[CH["nel"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["swu"])
            children[CH["nel"]].nb[NB["cl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["neu"])

            children[CH["nel"]].nb[NB["wu"]] = children[CH["nwu"]]
            children[CH["nel"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swu"])
            children[CH["nel"]].nb[NB["nu"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["seu"])
            children[CH["nel"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["ne"]], NB["ne"], CH["swu"])
            children[CH["nel"]].nb[NB["eu"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwu"])
            children[CH["nel"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swu"])
            children[CH["nel"]].nb[NB["su"]] = children[CH["seu"]]
            children[CH["nel"]].nb[NB["swu"]] = children[CH["swu"]]
            children[CH["nel"]].nb[NB["cu"]] = children[CH["neu"]]

            # lower-right child, lower plane (= south-east lower)
            children[CH["sel"]].nb[NB["w"]] = children[CH["swl"]]
            children[CH["sel"]].nb[NB["nw"]] = children[CH["nwl"]]
            children[CH["sel"]].nb[NB["n"]] = children[CH["nel"]]
            children[CH["sel"]].nb[NB["ne"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwl"])
            children[CH["sel"]].nb[NB["e"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swl"])
            children[CH["sel"]].nb[NB["se"]] = parent_or_child(cell.nb, check[NB["se"]], NB["se"], CH["nwl"])
            children[CH["sel"]].nb[NB["s"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nel"])
            children[CH["sel"]].nb[NB["sw"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwl"])

            children[CH["sel"]].nb[NB["wl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["swu"])
            children[CH["sel"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["nwu"])
            children[CH["sel"]].nb[NB["nl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["neu"])
            children[CH["sel"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["el"]], NB["el"], CH["nwu"])
            children[CH["sel"]].nb[NB["el"]] = parent_or_child(cell.nb, check[NB["el"]], NB["el"], CH["swu"])
            children[CH["sel"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["sel"]], NB["sel"], CH["nwu"])
            children[CH["sel"]].nb[NB["sl"]] = parent_or_child(cell.nb, check[NB["sl"]], NB["sl"], CH["neu"])
            children[CH["sel"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["sl"]], NB["sl"], CH["nwu"])
            children[CH["sel"]].nb[NB["cl"]] = parent_or_child(cell.nb, check[NB["cl"]], NB["cl"], CH["seu"])

            children[CH["sel"]].nb[NB["wu"]] = children[CH["swu"]]
            children[CH["sel"]].nb[NB["nwu"]] = children[CH["nwu"]]
            children[CH["sel"]].nb[NB["nu"]] = children[CH["neu"]]
            children[CH["sel"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwu"])
            children[CH["sel"]].nb[NB["eu"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swu"])
            children[CH["sel"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["se"]], NB["se"], CH["nwu"])
            children[CH["sel"]].nb[NB["su"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["neu"])
            children[CH["sel"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwu"])
            children[CH["sel"]].nb[NB["cu"]] = children[CH["seu"]]

            # lower-left child, upper plane (= south-west upper)
            children[CH["swu"]].nb[NB["wl"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["sel"])
            children[CH["swu"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["nel"])
            children[CH["swu"]].nb[NB["nl"]] = children[CH["nwl"]]
            children[CH["swu"]].nb[NB["nel"]] = children[CH["nel"]]
            children[CH["swu"]].nb[NB["el"]] = children[CH["sel"]]
            children[CH["swu"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nel"])
            children[CH["swu"]].nb[NB["sl"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwl"])
            children[CH["swu"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["sw"]], NB["sw"], CH["nel"])
            children[CH["swu"]].nb[NB["cl"]] = children[CH["swl"]]

            children[CH["swu"]].nb[NB["wu"]] = parent_or_child(cell.nb, check[NB["wu"]], NB["wu"], CH["sel"])
            children[CH["swu"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["wu"]], NB["wu"], CH["nel"])
            children[CH["swu"]].nb[NB["nu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nwl"])
            children[CH["swu"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nel"])
            children[CH["swu"]].nb[NB["eu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["sel"])
            children[CH["swu"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["su"]], NB["su"], CH["nel"])
            children[CH["swu"]].nb[NB["su"]] = parent_or_child(cell.nb, check[NB["su"]], NB["su"], CH["nwl"])
            children[CH["swu"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["swu"]], NB["swu"], CH["nel"])
            children[CH["swu"]].nb[NB["cu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["swl"])

            # upper-left child, upper plane (= north-west upper)
            children[CH["nwu"]].nb[NB["wl"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["nel"])
            children[CH["nwu"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["nw"]], NB["nw"], CH["sel"])
            children[CH["nwu"]].nb[NB["nl"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swl"])
            children[CH["nwu"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["sel"])
            children[CH["nwu"]].nb[NB["el"]] = children[CH["nel"]]
            children[CH["nwu"]].nb[NB["sel"]] = children[CH["sel"]]
            children[CH["nwu"]].nb[NB["sl"]] = children[CH["swl"]]
            children[CH["nwu"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["w"]], NB["w"], CH["sel"])
            children[CH["nwu"]].nb[NB["cl"]] = children[CH["nwl"]]

            children[CH["nwu"]].nb[NB["wu"]] = parent_or_child(cell.nb, check[NB["wu"]], NB["wu"], CH["nel"])
            children[CH["nwu"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["nwu"]], NB["nwu"], CH["sel"])
            children[CH["nwu"]].nb[NB["nu"]] = parent_or_child(cell.nb, check[NB["nu"]], NB["nu"], CH["swl"])
            children[CH["nwu"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["nu"]], NB["nu"], CH["sel"])
            children[CH["nwu"]].nb[NB["eu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nel"])
            children[CH["nwu"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["sel"])
            children[CH["nwu"]].nb[NB["su"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["swl"])
            children[CH["nwu"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["wu"]], NB["wu"], CH["sel"])
            children[CH["nwu"]].nb[NB["cu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nwl"])

            # upper right child, upper plane (= north-east upper)
            children[CH["neu"]].nb[NB["wl"]] = children[CH["nwl"]]
            children[CH["neu"]].nb[NB["nwl"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["swl"])
            children[CH["neu"]].nb[NB["nl"]] = parent_or_child(cell.nb, check[NB["n"]], NB["n"], CH["sel"])
            children[CH["neu"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["ne"]], NB["ne"], CH["swl"])
            children[CH["neu"]].nb[NB["el"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwl"])
            children[CH["neu"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swl"])
            children[CH["neu"]].nb[NB["sl"]] = children[CH["sel"]]
            children[CH["neu"]].nb[NB["swl"]] = children[CH["swl"]]
            children[CH["neu"]].nb[NB["cl"]] = children[CH["nel"]]

            children[CH["neu"]].nb[NB["wu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nwl"])
            children[CH["neu"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["nu"]], NB["nu"], CH["swl"])
            children[CH["neu"]].nb[NB["nu"]] = parent_or_child(cell.nb, check[NB["nu"]], NB["nu"], CH["sel"])
            children[CH["neu"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["neu"]], NB["neu"], CH["swl"])
            children[CH["neu"]].nb[NB["eu"]] = parent_or_child(cell.nb, check[NB["eu"]], NB["eu"], CH["nwl"])
            children[CH["neu"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["eu"]], NB["eu"], CH["swl"])
            children[CH["neu"]].nb[NB["su"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["sel"])
            children[CH["neu"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["swl"])
            children[CH["neu"]].nb[NB["cu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nel"])

            # lower right child, upper plane (= south-east upper)
            children[CH["seu"]].nb[NB["wl"]] = children[CH["swl"]]
            children[CH["seu"]].nb[NB["nwl"]] = children[CH["nwl"]]
            children[CH["seu"]].nb[NB["nl"]] = children[CH["nel"]]
            children[CH["seu"]].nb[NB["nel"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["nwl"])
            children[CH["seu"]].nb[NB["el"]] = parent_or_child(cell.nb, check[NB["e"]], NB["e"], CH["swl"])
            children[CH["seu"]].nb[NB["sel"]] = parent_or_child(cell.nb, check[NB["se"]], NB["se"], CH["nwl"])
            children[CH["seu"]].nb[NB["sl"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nel"])
            children[CH["seu"]].nb[NB["swl"]] = parent_or_child(cell.nb, check[NB["s"]], NB["s"], CH["nwl"])
            children[CH["seu"]].nb[NB["cl"]] = children[CH["sel"]]

            children[CH["seu"]].nb[NB["wu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["swl"])
            children[CH["seu"]].nb[NB["nwu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nwl"])
            children[CH["seu"]].nb[NB["nu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["nel"])
            children[CH["seu"]].nb[NB["neu"]] = parent_or_child(cell.nb, check[NB["eu"]], NB["eu"], CH["nwl"])
            children[CH["seu"]].nb[NB["eu"]] = parent_or_child(cell.nb, check[NB["eu"]], NB["eu"], CH["swl"])
            children[CH["seu"]].nb[NB["seu"]] = parent_or_child(cell.nb, check[NB["seu"]], NB["seu"], CH["nwl"])
            children[CH["seu"]].nb[NB["su"]] = parent_or_child(cell.nb, check[NB["su"]], NB["su"], CH["nel"])
            children[CH["seu"]].nb[NB["swu"]] = parent_or_child(cell.nb, check[NB["su"]], NB["su"], CH["nwl"])
            children[CH["seu"]].nb[NB["cu"]] = parent_or_child(cell.nb, check[NB["cu"]], NB["cu"], CH["sel"])

        return children

    def _assign_indices(self, cells):
        """
        Due to round-off errors, accumulation of errors, etc. even with double precision and is_close(), we are
        interpreting the same node shared by adjacent cells as different node, so we need to generate the node indices
        without any coordinate information

        :param cells: Tuple containing the child cells
        :return: None
        """
        for i in range(len(cells)):
            # add the cell center to the set containing all centers -> ensuring that the order of nodes and centers is
            # consistent
            self.all_centers.append(cells[i].center)

            # initialize an empty list
            cells[i].node_idx = [0] * 2 ** self._n_dimensions

            # compute the node locations of the current cell
            nodes = self._compute_cell_centers(_factor=0.5, _keep_parent_center=False, _cell=cells[i])

            # the node of the parent cell, for child cell 0: node 0 == node 0 of parent cell and so on
            cells[i].node_idx[i] = cells[i].parent.node_idx[i]

            # at the moment we treat 2D & 3D separately, if it works then we may make this more efficient by combining
            # in find_nb, we are assigning the parent nb, so we need to go to parent, and then to children
            if self._n_dimensions == 2:
                if i == 0:
                    # left nb
                    if check_nb_node(cells[i], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].nb[NB["w"]].node_idx[CH["neu"]]
                    else:
                        self.all_nodes.append(nodes[1, :])
                        cells[i].node_idx[CH["nwu"]] = len(self.all_nodes) - 1

                    # then the node in the center off all children
                    self.all_nodes.append(nodes[2, :])
                    cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1

                    # lower nb
                    if check_nb_node(cells[i], nb_no=NB["s"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["s"]].node_idx[CH["neu"]]
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1

                elif i == 1:
                    # upper nb
                    if check_nb_node(cells[i], nb_no=NB["n"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].nb[NB["n"]].node_idx[CH["seu"]]
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1
                    cells[i].node_idx[CH["swu"]] = cells[CH["swu"]].node_idx[CH["nwu"]]
                    cells[i].node_idx[CH["seu"]] = cells[CH["swu"]].node_idx[CH["neu"]]

                elif i == 2:
                    # right nb
                    if check_nb_node(cells[i], nb_no=NB["e"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["e"]].node_idx[CH["swu"]]
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
                    if check_nb_node(cells[i], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].nb[NB["w"]].node_idx[CH["neu"]]
                    # check left nb (upper plane) if node 1 is present
                    elif check_nb_node(cells[i], nb_no=NB["wu"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].nb[NB["wu"]].node_idx[CH["nel"]]
                    # check nb above current cell (upper plane) if node 1 is present
                    elif check_nb_node(cells[i], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["nwu"]] = cells[i].nb[NB["cu"]].node_idx[CH["nwl"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[1, :])
                        cells[i].node_idx[CH["nwu"]] = len(self.all_nodes) - 1

                    # check nb above current cell (upper plane) if node 2 is present
                    if check_nb_node(cells[i], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].nb[NB["cu"]].node_idx[CH["nel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 3 is present
                    if check_nb_node(cells[i], nb_no=NB["s"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["s"]].node_idx[CH["neu"]]
                    # check lower nb (upper plane) if node 3 is present
                    elif check_nb_node(cells[i], nb_no=NB["su"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["su"]].node_idx[CH["nel"]]
                    # check nb above current cell (upper plane) if node 2 is present
                    elif check_nb_node(cells[i], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["cu"]].node_idx[CH["sel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1

                    # check left nb (same plane) if node 4 is present
                    if check_nb_node(cells[i], nb_no=NB["w"]):
                        cells[i].node_idx[CH["swl"]] = cells[i].nb[NB["w"]].node_idx[CH["sel"]]
                    # check the lower left corner nb (same plane) if node 4 is present
                    elif check_nb_node(cells[i], nb_no=NB["sw"]):
                        cells[i].node_idx[CH["swl"]] = cells[i].nb[NB["sw"]].node_idx[CH["nel"]]
                    # check lower nb (same plane) if node 4 is present
                    elif check_nb_node(cells[i], nb_no=NB["s"]):
                        cells[i].node_idx[CH["swl"]] = cells[i].nb[NB["s"]].node_idx[CH["nwl"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[4, :])
                        cells[i].node_idx[CH["swl"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 5 is present
                    if check_nb_node(cells[i], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["w"]].node_idx[CH["nel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[CH["nwl"]] = len(self.all_nodes) - 1

                    # node no. 6 is in the center of all child cells, this node can't exist in other nb
                    self.all_nodes.append(nodes[6, :])
                    cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 7 is present
                    if check_nb_node(cells[i], nb_no=NB["s"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["s"]].node_idx[CH["nel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[7, :])
                        cells[i].node_idx[CH["sel"]] = len(self.all_nodes) - 1

                elif i == 1:
                    # child no. 1: new nodes 2, 5, 6 remain to add
                    # check upper nb (same plane) if node 2 is present
                    if check_nb_node(cells[i], nb_no=NB["n"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].nb[NB["n"]].node_idx[CH["seu"]]
                    # check upper nb (upper plane) if node 2 is present
                    elif check_nb_node(cells[i], nb_no=NB["nu"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].nb[NB["nu"]].node_idx[CH["sel"]]
                    # check nb above current cell (upper plane) if node 2 is present
                    elif check_nb_node(cells[i], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["neu"]] = cells[i].nb[NB["cu"]].node_idx[CH["nel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[2, :])
                        cells[i].node_idx[CH["neu"]] = len(self.all_nodes) - 1

                    # check left nb (same plane) if node 5 is present
                    if check_nb_node(cells[i], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["w"]].node_idx[CH["nel"]]
                    # check the upper left corner nb (same plane) if node 5 is present
                    elif check_nb_node(cells[i], nb_no=NB["nw"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["nw"]].node_idx[CH["sel"]]
                    # check upper nb (same plane) if node 5 is present
                    elif check_nb_node(cells[i], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["n"]].node_idx[CH["swl"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[CH["nwl"]] = len(self.all_nodes) - 1

                    # check upper nb (same plane) if node 6 is present
                    if check_nb_node(cells[i], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["n"]].node_idx[CH["sel"]]
                    # otherwise, this node does not exist yet
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
                    if check_nb_node(cells[i], nb_no=NB["e"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["e"]].node_idx[CH["swu"]]
                    # check right nb (upper plane) if node 3 is present
                    elif check_nb_node(cells[i], nb_no=NB["eu"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["eu"]].node_idx[CH["swl"]]
                    # check nb above current cell (upper plane) if node 3 is present
                    elif check_nb_node(cells[i], nb_no=NB["cu"]):
                        cells[i].node_idx[CH["seu"]] = cells[i].nb[NB["cu"]].node_idx[CH["sel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[3, :])
                        cells[i].node_idx[CH["seu"]] = len(self.all_nodes) - 1

                    # check right nb (same plane) if node 6 is present
                    if check_nb_node(cells[i], nb_no=NB["e"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["e"]].node_idx[CH["nwl"]]
                    # check the upper right corner nb (same plane) if node 6 is present
                    elif check_nb_node(cells[i], nb_no=NB["ne"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["ne"]].node_idx[CH["swl"]]
                    # check upper nb (same plane) if node 6 is present
                    elif check_nb_node(cells[i], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["n"]].node_idx[CH["sel"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    # check right nb (same plane) if node 7 is present
                    if check_nb_node(cells[i], nb_no=NB["e"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["e"]].node_idx[CH["swl"]]
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
                    if check_nb_node(cells[i], nb_no=NB["e"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["e"]].node_idx[CH["swl"]]
                    # check the lower right corner nb (same plane) if node 7 is present
                    elif check_nb_node(cells[i], nb_no=NB["se"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["se"]].node_idx[CH["nwl"]]
                    # check lower nb (same plane) if node 7 is present
                    elif check_nb_node(cells[i], nb_no=NB["s"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["s"]].node_idx[CH["nel"]]
                    # otherwise, this node does not exist yet
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
                    if check_nb_node(cells[i], nb_no=NB["w"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["w"]].node_idx[CH["nel"]]
                    # check left nb (lower plane) if node 5 is present
                    elif check_nb_node(cells[i], nb_no=NB["wl"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["wl"]].node_idx[CH["neu"]]
                    # check nb below current cell (lower plane) if node 5 is present
                    elif check_nb_node(cells[i], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["nwl"]] = cells[i].nb[NB["cl"]].node_idx[CH["nwu"]]
                    # otherwise, this node does not exist yet
                    else:
                        self.all_nodes.append(nodes[5, :])
                        cells[i].node_idx[CH["nwl"]] = len(self.all_nodes) - 1

                    # check nb below current cell (lower plane) if node 6 is present
                    if check_nb_node(cells[i], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["cl"]].node_idx[CH["neu"]]
                    else:
                        self.all_nodes.append(nodes[6, :])
                        cells[i].node_idx[CH["nel"]] = len(self.all_nodes) - 1

                    # check lower nb (same plane) if node 7 is present
                    if check_nb_node(cells[i], nb_no=NB["s"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["s"]].node_idx[CH["nel"]]
                    # check lower corner nb (lower plane) if node 7 is present
                    elif check_nb_node(cells[i], nb_no=NB["sl"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["sl"]].node_idx[CH["neu"]]
                    # check nb below current cell (lower plane) if node 7 is present
                    elif check_nb_node(cells[i], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["cl"]].node_idx[CH["seu"]]
                    # otherwise, this node does not exist yet
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
                    if check_nb_node(cells[i], nb_no=NB["n"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["n"]].node_idx[CH["sel"]]
                    # check the upper corner nb (lower plane) if node 6 is present
                    elif check_nb_node(cells[i], nb_no=NB["nl"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["nl"]].node_idx[CH["seu"]]
                    # check nb below current cell (lower plane) if node 6 is present
                    elif check_nb_node(cells[i], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["nel"]] = cells[i].nb[NB["cl"]].node_idx[CH["neu"]]
                    # otherwise, this node does not exist yet
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
                    if check_nb_node(cells[i], nb_no=NB["e"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["e"]].node_idx[CH["swl"]]
                    # check the right corner nb (lower plane) if node 7 is present
                    elif check_nb_node(cells[i], nb_no=NB["el"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["el"]].node_idx[CH["swu"]]
                    # check nb below current cell (lower plane) if node 7 is present
                    elif check_nb_node(cells[i], nb_no=NB["cl"]):
                        cells[i].node_idx[CH["sel"]] = cells[i].nb[NB["cl"]].node_idx[CH["seu"]]
                    # otherwise, this node does not exist yet
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

    def _refine_geometries(self) -> None:
        """
        Check which geometries should be refined and execute the refinement process for each of these geometries

        :return: None
        """
        # check which geometries should be refined
        geometries_to_refine = [idx for idx, g in enumerate(self._geometry) if g.refine]

        # if we have any to refine, then refine them
        if geometries_to_refine:
            self._times["t_end_adaptive"] = self._times["t_start_geometry"] - self._times["t_start_adaptive"]
            self._times["t_start_geometry"] = time()
            self._execute_geometry_refinement(_geometries=geometries_to_refine)

            # save the required time for refining the geometries
            self._times["t_end_geometry"] = time()

    def _create_mesh_info(self, counter: int) -> None:
        """
        create an info dict containing information about the created mesh and required execution times for each part of
        the refinement process

        :param counter: number of iterations required for the adaptive refinement
        :return: None
        """
        self.data_final_mesh["size_initial_cell"] = self._width
        self.data_final_mesh["n_cells_orig"] = self._n_cells_orig
        self.data_final_mesh["n_cells"] = len(self._leaf_cells)
        self.data_final_mesh["iterations"] = counter
        self.data_final_mesh["min_level"] = self._current_min_level
        self.data_final_mesh["max_level"] = self._current_max_level
        self.data_final_mesh["metric_per_iter"] = self._metric
        self.data_final_mesh["cells_per_iter"] = self._n_cells_log
        self.data_final_mesh["t_total"] = self._times["t_end_renumber"] - self._times["t_start_uniform"]
        self.data_final_mesh["t_uniform"] = self._times["t_end_uniform"] - self._times["t_start_uniform"]
        self.data_final_mesh["t_renumbering"] = self._times["t_end_renumber"] - self._times["t_start_renumber"]

        if self._times["t_end_geometry"] > 0:
            self.data_final_mesh["t_geometry"] = self._times["t_end_geometry"] - self._times["t_start_geometry"]
            self.data_final_mesh["t_adaptive"] = self._times["t_start_geometry"] - self._times["t_start_adaptive"]
        else:
            self.data_final_mesh["t_geometry"] = None
            self.data_final_mesh["t_adaptive"] = self._times["t_start_renumber"] - self._times["t_start_adaptive"]

    def __len__(self):
        return self._n_cells

    def __str__(self):
        message = [f"Finished refinement in {self.data_final_mesh['t_total']:2.4f} s ",
                   f"({self.data_final_mesh['iterations']} iterations).",
                   f"Time for uniform refinement: {self.data_final_mesh['t_uniform']:2.4f} s",
                   f"Time for adaptive refinement: {self.data_final_mesh['t_adaptive']:2.4f} s"]

        if self.data_final_mesh['t_geometry'] is not None:
            message += [f"Time for geometry refinement: {self.data_final_mesh['t_geometry']:2.4f} s"]
        message += ["Time for renumbering the final mesh: {:2.4f} s".format(self.data_final_mesh['t_renumbering'])]

        message += ["""
                                    Number of cells: {:d}
                                    Minimum ref. level: {:d}
                                    Maximum ref. level: {:d}
                                    Captured metric of original grid: {:.2f} %
                  """.format(len(self._leaf_cells), self._current_min_level, self._current_max_level,
                             self._metric[-1] * 100)]
        return "\n\t\t\t\t\t\t\t\t".join(message)

    @property
    def n_dimensions(self) -> int:
        return self._n_dimensions

    @property
    def width(self) -> pt.Tensor:
        return self._width

    @property
    def geometry(self) -> list:
        return self._geometry

    def _print_settings(self):
        omit = ["_times", "all_centers", "data_final_mesh", "_width", "_pool", "_knn", "_cells", "_leaf_cells",
                "_n_cells_after_uniform", "_N_cells_per_iter", "all_nodes", "face_ids", "_metric", "_current_min_level",
                "_current_max_level", "_vertices", "_n_cells", "_n_cells_log"]
        max_key_len = max(len(key) for key in self.__dict__ if key not in omit)

        atts = ["\n\tSelected settings:"]
        for key, value in self.__dict__.items():
            if key not in omit:
                if key == "_geometry":
                    val_str = str([v.name for v in value])
                else:
                    val_str = str(value)
                atts.append(f"\t\t{key:<{max_key_len}}:\t{val_str}")
        logger.info("\n".join(atts))


@njit(parallel=True, fastmath=True, nogil=True)
def renumber_node_indices_parallel(all_idx: np.ndarray, all_nodes: np.ndarray,
                                   unused_idx: set, dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    remove all unused nodes and center coordinates from the list, re-number the node coordinates - parallelized version

    :param all_idx: array containing all indices of nodes, which are used within the grid
    :param all_nodes: all nodes which have been created throughout the refinement process
    :param unused_idx: node indices of coordinates, which are not present in the final grid anymore
    :param dims: number of physical dimensions
    :return: array with unique node coordinates used in the final grid and array with re-numbered node indices pointing
             to these node coordinates
    """
    # TODO: documentation
    # build mapping from old index -> new index
    mapping = np.full(all_nodes.shape[0], -1, dtype=np.int64)
    counter = 0
    for i in range(all_nodes.shape[0]):
        if i not in unused_idx:
            mapping[i] = counter
            counter += 1

    # create the new unique node
    _unique_nodes = np.zeros((counter, dims), dtype=all_nodes.dtype)
    for i in prange(all_nodes.shape[0]):
        if mapping[i] != -1:
            _unique_nodes[mapping[i], :] = all_nodes[i, :]

    # remap all_idx using the mapping
    orig_shape = all_idx.shape
    all_idx = all_idx.flatten()
    for j in prange(all_idx.shape[0]):
        all_idx[j] = mapping[all_idx[j]]

    return _unique_nodes, all_idx.reshape(orig_shape)


@njit(fastmath=True, nogil=True)
def renumber_node_indices(all_idx: np.ndarray, all_nodes: np.ndarray, _unused_idx: np.ndarray,
                          dims: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    remove all unused nodes and center coordinates from the list, re-number the node coordinates.

    :param all_idx: array containing all indices of nodes, which are used within the grid
    :param all_nodes: all nodes which have been created throughout the refinement process
    :param _unused_idx: node indices of coordinates, which are not present in the final grid anymore
    :param dims: number of physical dimensions
    :return: array with unique node coordinates used in the final grid and array with re-numbered node indices pointing
             to these node coordinates
    """
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
    # the end we can't run this loop in parallel, because it messes up the idx numbers
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
            # decrement all idx that are > current index by 1, since we are deleting this node. but since we are
            # overwriting the all_idx tensor, we need to account for the entries we already deleted
            _visited += 1
            for j in range(all_idx.shape[0]):
                if all_idx[j] > i - _visited:
                    all_idx[j] -= 1
        else:
            _unique_node_coord[_counter, :] = all_nodes[i, :]
            _counter += 1

    return _unique_node_coord, all_idx.reshape(orig_shape)


def check_nb_node(_cell: Cell, nb_no: int) -> bool:
    # since we already updated the NB prior assigning the indices, we only need to check if the nb cell of the current
    # child cell exists and has the same level
    return _cell.nb[nb_no] is not None and _cell.nb[nb_no].leaf_cell() and _cell.level == _cell.nb[nb_no].level


def parent_or_child(nb: list, check: bool, nb_idx: int, child_idx: int) -> Cell:
    """
    get the neighbor cell of a newly created child cell

    :param nb: list containing the neighbors of the current parent cell
    :param check: flag if the current parent cell has children
    :param nb_idx: index of the neighbor for which we want to check
    :param child_idx: index of the possible child, which would be the nb of the child cell we are currently looking at
    :return: the nb child cell of current child cell if present, else the nb parent cell of current child cell
    """
    return nb[nb_idx].children[child_idx] if check else nb[nb_idx]


def initialize_time_dict() -> dict:
    return {"t_start_uniform": 0.0, "t_end_uniform": 0.0,
            "t_start_adaptive": 0.0,                            # we don't need t_end_adaptive
            "t_start_geometry": 0.0, "t_end_geometry": 0.0,
            "t_start_renumber": 0.0, "t_end_renumber": 0.0}


def _check_cell_validity(args) -> Union[None, int]:
    """
    check if a cell is valid or invalid

    :param args:  cell, node, geometries, refine_geometry
    :return: cell index if cell should be removed or None if cell is valid
    """
    cell, node, geometries, refine_geometry, _pre_select_cells = args
    for g in geometries:
        if _pre_select_cells:
            if (g.type == "STL" or g.type == "coord_2D") and not g.pre_check_cell(node, refine_geometry):
                continue
        elif g.check_cell(node, refine_geometry):
            return cell
    return None


@njit(fastmath=True, nogil=True)
def _update_gain(args) -> float:
    """
    compute gain of a cell

    :param args: _level, _n_children, _n_dims, _width, gain_0, sum_delta_metric
    :return: gain of the cell as float
    """
    _level, _n_dims, _width, gain_0, sum_delta_metric = args

    # scale with the cell size of the children (the cell size is computed based on the cell level and the
    # size of the initial cell), then normalize gain with gain of the initial cell
    return 1 / (2 ** _n_dims) * ((_width / (2 ** _level)) ** _n_dims) * sum_delta_metric / gain_0


if __name__ == "__main__":
    pass
