"""
    TODO: implement the tests for masking out geometries and domains
"""
import shapely
import torch as pt
import pyvista as pv

from s_cube.geometry import GeometryObject


def test_domain_2d():
    pass


def test_domain_3d():
    pass


def test_geometry_2d():
    pass


def test_geometry_2d_stl():
    pass


def test_geometry_3d_stl():
    print(f"Starting check for {test_geometry_3d_stl.__name__}.")

    # load the STL file of the cube and create geometry object from it
    cube = GeometryObject(lower_bound=None, upper_bound=None, obj_type="stl", geometry=True, name="cube",
                          _coordinates=pv.PolyData("cube.stl"), _dimensions=3)

    # generate cell completely inside the cube
    cell_inside = pt.tensor([[3.75, 4.25, 0.25], [3.75, 4.75, 0.25], [3.75, 4.25, 0.75], [3.75, 4.75, 0.75],
                             [4.25, 4.25, 0.25], [4.25, 4.75, 0.25], [4.25, 4.25, 0.75], [4.25, 4.75, 0.75]])

    # generate cell completely outside the cube
    cell_outside = pt.tensor([[5.25, 4.25, 0.25], [5.25, 4.75, 0.25], [5.25, 4.25, 0.75], [5.25, 4.75, 0.75],
                              [5.75, 4.25, 0.25], [5.75, 4.75, 0.25], [5.75, 4.25, 0.75], [5.75, 4.75, 0.75]])

    # generate cell partially inside the cube
    cell_part = pt.tensor([[3.25, 4.25, 0.25], [3.25, 4.75, 0.25], [3.25, 4.25, 0.75], [3.25, 4.75, 0.75],
                           [3.75, 4.25, 0.25], [3.75, 4.75, 0.25], [3.75, 4.25, 0.75], [3.75, 4.75, 0.75]])

    # test if points are inside or outside using pyVista
    assert cube.check_geometry(cell_outside) is False, "Cell outside geometry not masked correctly."
    assert cube.check_geometry(cell_inside) is True, "Cell inside geometry not masked correctly."
    assert cube.check_geometry(cell_part) is False, "Cell partially outside geometry not masked correctly."

    print(f"Check {test_geometry_3d_stl.__name__} passed.")


if __name__ == "__main__":
    test_geometry_3d_stl()

