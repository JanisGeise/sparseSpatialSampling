"""
Provides reusable test cells for verifying the ``check_cell()`` method in different geometry objects.
"""
from torch import tensor, float32


class DummyCells:
    """
    Dummy cells for easy 2D and 3D geometry testing.

    Each cell contains **4 nodes (2D)** or **8 nodes (3D)**.
    The following three scenarios are provided for both 2D and 3D:

    - **Inside**: Cell located entirely within the unit square/cube.
    - **Outside**: Cell located far away from the origin.
    - **Partially**: Cell overlapping with the unit square/cube boundary.

    Coordinates of each test cell:

    2D Cells
    --------
        - ``cell_inside_2D``: Inside the unit square
            [[0, 0], [0, 1], [1, 1], [1, 0]]
        - ``cell_outside_2D``: Far away from origin
            [[5, 5], [6, 5], [6, 6], [5, 6]]
        - ``cell_partially_2D``: Overlaps with unit square boundary
            [[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]]

    3D Cells
    --------
        - ``cell_inside_3D``: Inside the unit cube
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
             [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
        - ``cell_outside_3D``: Far away from origin
            [[5, 5, 5], [6, 5, 5], [6, 6, 5], [5, 6, 5],
             [5, 5, 6], [6, 5, 6], [6, 6, 6], [5, 6, 6]]
        - ``cell_partially_3D``: Overlaps with unit cube boundary
            [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [0.5, 1.5, 0.5],
             [0.5, 0.5, 1.5], [1.5, 0.5, 1.5], [1.5, 1.5, 1.5], [0.5, 1.5, 1.5]]
    """
    def __init__(self) -> None:
        # Test cells in 2D, make sure we have floats, otherwise issue in flowtorch
        self.cell_inside_2D = tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float32)
        self.cell_outside_2D = tensor([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float32)
        self.cell_partially_2D = tensor([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]], dtype=float32)

        # Test cells in 3D
        self.cell_inside_3D = tensor([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float32)

        self.cell_outside_3D = tensor([[5, 5, 5], [6, 5, 5], [6, 6, 5], [5, 6, 5],
                                       [5, 5, 6], [6, 5, 6], [6, 6, 6], [5, 6, 6]], dtype=float32)

        self.cell_partially_3D = tensor([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [1.5, 1.5, 0.5], [0.5, 1.5, 0.5],
                                         [0.5, 0.5, 1.5], [1.5, 0.5, 1.5], [1.5, 1.5, 1.5], [0.5, 1.5, 1.5]], dtype=float32)

    @property
    def cells_2D(self) -> dict:
        """
        Return a dictionary of 2D test cells for easier testing.
        """
        return {"inside": self.cell_inside_2D, "outside": self.cell_outside_2D, "partially": self.cell_partially_2D}

    @property
    def cells_3D(self) -> dict:
        """
        Return a dictionary of 3D test cells for easier testing.
        """
        return {"inside": self.cell_inside_3D, "outside": self.cell_outside_3D, "partially": self.cell_partially_3D}

if __name__ == "__main__":
    pass
