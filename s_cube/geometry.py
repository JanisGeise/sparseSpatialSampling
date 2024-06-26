"""
    Implements class for storing the geometry and the domain; currently only simple geometries are allowed, such as
        - rectangle (2D), cube (3D)
        - circle (2D), sphere (3D)

    Further, arbitrary geometries, e.g., loaded from STL files, can be used. However, these geometries need to be
    provided in a form such their coordinates represent an enclosed area (2D). For 3D, the STL file can be loaded
    directly using pyVista as

    'pyvista.Polydata("path_to_stl_file")'

    For more information, it is referred to the examples in the 'examples' directory
"""
import logging
import pyvista as pv

from torch import tensor
from shapely import Point, Polygon
from flowtorch.data import mask_box, mask_sphere

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class GeometryObject:
    def __init__(self, lower_bound, upper_bound, obj_type: str, geometry: bool = True, name: str = "cube",
                 _coordinates: any = None, _dimensions: int = None):
        """
        Implements a geometry object acting as geometry inside the numerical domain, or as the numerical domain.

        Note:

            - for 3D-STL files, the STL file needs to be loaded and passed directly via pyVIsta, e.g.

                cube = pv.PolyData(join("..", "tests", "cube.stl"))

            - for 2D STL files, only the coordinates need to be passed; the type is irrelevant, since they will be
              converted to shapely.Polygon() types.However, it is important that these points form an enclosed area.


        Further:
                exactly one geometry needs to be specified as the (main) domain, which is used to compute the
                main dimensions of the domain and to initialize everything. If further subdomains need to be
                specified, e.g., to mask out other areas (e.g., a step at the domain boundary), these geometry
                objects are not allowed to have the 'obj_type' domain. To define that these objects should act
                as a domain, the parameter 'inside' needs to be set to False instead

        :param lower_bound: Lower boundary, sorted as:

                            [x_min, y_min, z_min]               (3D, cube type)
                            [x_center, y_center, z_center]      (3D, sphere type)

                            For 2D, the z-component is not present. If lower_bound is 'None', an STL file needs to be
                            provided as geometry
        :param upper_bound: upper boundary, sorted as:
                            [x_max, y_max, z_max]       (3D, cube type)
                            [radius]                    (3D, sphere type)
                            If 'None', an STL file needs to be provided as geometry
        :param obj_type: Either "sphere", "cube" or "STL". If 'STL', the coordinates of the geometry need to be provided
        :param geometry: if we have a geometry type or a domain type. If 'False', all points outside this geometry will
                         be removed, if 'True' all points inside the geometry will be removed
        :param name: name of the geometry object, can be chosen freely
        :param _coordinates: coordinates of the geometry, required if an STL file is provided as geometry. Note: The
                             coordinates have to form an enclosed ares (2D)
        :param _dimensions: number of physical dimensions, needs to be provided if an STL file for 3D is used
        """
        self.inside = geometry
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._obj_type = obj_type
        self.obj_name = name
        self._dimensions = _dimensions

        if _coordinates is not None:
            assert self._dimensions is not None, ("When providing Coordinates for geometry objects, the number of "
                                                  "physical dimensions have to be provided as well.")
            if _dimensions == 2:
                self._coordinates = Polygon(_coordinates)
            else:
                self._coordinates = _coordinates

        # check if the object type matches the ones currently implemented
        self._check_obj_type()

    def check_geometry(self, cell_nodes, _refine: bool = False) -> bool:
        if self._obj_type == "sphere":
            # for a sphere, the lower bound corresponds to its radius
            mask = mask_sphere(cell_nodes, self._lower_bound, self._upper_bound[0])
        elif self._obj_type == "cube":
            mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)
        else:
            # for each node of the cell, check if it is inside the geometry. We can't compute this at once for all
            # nodes, because within() method only returns a single bool, but we need to have a bool for each node
            if self._dimensions == 2:
                mask = tensor([Point(cell_nodes[i, :]).within(self._coordinates) for i in range(cell_nodes.size(0))])
            else:
                # for 3D geometries represented by STL files, we need to mask using pyVista
                n = pv.PolyData(cell_nodes.tolist())
                mask = tensor(n.select_enclosed_points(self._coordinates, check_surface=False)["SelectedPoints"]).bool()

        # if we are not refining the geometry, then we want to remove the cells which have all nodes located inside a
        # geometry or outside the domain
        if not _refine:
            # any(~geometry), because mask returns False if we are outside, but we want True if we are outside
            if self.inside:
                if any(~mask):
                    invalid = False
                else:
                    invalid = True

            # if we are outside the domain, we want to return False
            else:
                if any(mask):
                    invalid = False
                else:
                    invalid = True

        # otherwise, we want to refine all cells that have at least one node in the geometry / outside the domain
        else:
            if self.inside:
                if all(~mask):
                    invalid = False
                else:
                    invalid = True
            else:
                if all(mask):
                    invalid = False
                else:
                    invalid = True

        return invalid

    def _check_obj_type(self):
        if self._obj_type != "cube" and self._obj_type != "sphere" and self._obj_type.upper() != "STL":
            logger.critical(f"Unknown object type '{self._obj_type}'. Valid object types are 'sphere' or 'cube'")
            exit()
        if self._obj_type.upper() == "STL" and self._coordinates is None:
            # if we have an STL file check if the coordinates are provided, if not then exit
            logger.critical(f"Coordinates of the STL file for geometry '{self.obj_name}' are not provided.")
            exit()


if __name__ == "__main__":
    pass
