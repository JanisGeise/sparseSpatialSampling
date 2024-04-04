"""
    implements class for storing the geometry and the domain, currently only simple geometries are allowed, such as:
        - rectangle (2D), cube (3D)
        - circle (2D), sphere (3D)
"""
from flowtorch.data import mask_box, mask_sphere


class GeometryObject:
    def __init__(self, lower_bound, upper_bound, obj_type: str, geometry: bool = True, name: str = "cube"):
        self._inside = geometry
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._obj_type = obj_type
        self._obj_name = name

        # check if the object type matches the ones currently implemented
        self._check_obj_type()

    def check_geometry(self, cell_nodes) -> bool:
        if self._obj_type == "sphere":
            mask = mask_sphere(cell_nodes, self._lower_bound, self._upper_bound)
        else:
            mask = mask_box(cell_nodes, self._lower_bound, self._upper_bound)

        # any(~geometry), because mask returns False if we are outside, but we want True if we are outside
        if self._inside:
            if any(~mask):
                invalid = False
            else:
                invalid = True

        # if we are outside the domain we want to return False
        elif any(mask):
            invalid = False
        else:
            invalid = True

        return invalid

    def _check_obj_type(self):
        if self._obj_type != "cube" and self._obj_type != "sphere":
            print(f"Unknown object type '{self._obj_type}'. Valid object types are 'sphere' or 'cube'")
            exit()


if __name__ == "__main__":
    pass
