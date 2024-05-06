"""
    implements class for storing the geometry and the domain, currently only simple geometries are allowed, such as:
        - rectangle (2D), cube (3D)
        - circle (2D), sphere (3D)
"""
import torch as pt
from flowtorch.data import mask_box, mask_sphere
from shapely import Point


class GeometryObject:
    def __init__(self, lower_bound, upper_bound, obj_type: str, geometry: bool = True, name: str = "cube",
                 _refine: bool = False, _coordinates: any = None):
        self._inside = geometry
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._obj_type = obj_type
        self.obj_name = name
        self._refine = _refine
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
            # for each node of the cell check if it is inside the geometry. We can't compute this at once for all nodes,
            # because within() method only returns a single bool, but we need to have a bool for each node
            mask = pt.tensor([Point(cell_nodes[i, :]).within(self._coordinates) for i in range(cell_nodes.size(0))])

        # if we are not refining the geometry, then we want to remove the cells which have all nodes located inside a
        # geometry or outside the domain
        if not _refine:
            # any(~geometry), because mask returns False if we are outside, but we want True if we are outside
            if self._inside:
                if any(~mask):
                    invalid = False
                else:
                    invalid = True

            # if we are outside the domain we want to return False
            else:
                if any(mask):
                    invalid = False
                else:
                    invalid = True

        # otherwise we want to refine all cells which have at least one node in the geometry / outside the domain
        else:
            if self._inside:
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
            print(f"Unknown object type '{self._obj_type}'. Valid object types are 'sphere' or 'cube'")
            exit()
        if self._obj_type.upper() == "STL" and self._coordinates is None:
            # if we have an STL file check if the coordinates are provided, if not then exit
            print(f"Coordinates of the STL file for geometry '{self.obj_name}' are not provided.")
            exit()


if __name__ == "__main__":
    pass
