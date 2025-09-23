"""
Unit tests for the ``Dataloader`` class using the synthetic ``s_cube_test_dataset.h5`` file.

These tests verify that the dataloader:
- Correctly loads metadata such as write times and field names.
- Provides consistent shapes for vertices, faces, nodes, weights, and refinement levels.
- Correctly loads snapshots for a given field and time.

The test dataset is expected to contain:
- 209 cells,
- 247 nodes,
- A single pressure field ``p`` at time ``t = 0.4``,
- Associated grid, metric, and refinement level information (implicitly checked via weights).
"""
import pytest
from os.path import join

from ..data import Dataloader

FILENAME = r"s_cube_test_dataset.h5"


def test_dataloader():
    """
    Test that the ``Dataloader`` correctly loads the synthetic test dataset.

    This test verifies:
    - The number of write times matches the expected single entry.
    - The field names dictionary matches the expected structure.
    - The shapes of vertices, faces, nodes, weights, and levels are consistent with the dataset.
    - The snapshot loader correctly retrieves the pressure field ``p`` at ``t = 0.4`` with the expected shape.

    Expected dataset properties:
    - Cells: 209
    - Nodes: 247
    - Dimensions: 2
    - Write times: ["0.4"]
    - Fields: {"0.4": ["p"]}
    """
    n_cells = 209
    n_nodes = 247
    n_dimensions = 2
    write_times = ["0.4"]
    field_names = {'0.4': ['p']}

    # instantiate dataloader
    dataloader = Dataloader(join("sparseSpatialSampling", "tests"), FILENAME)

    # check if everything is present and can be loaded correctly
    assert len(dataloader.write_times) == 1
    assert dataloader.write_times == write_times
    assert dataloader.field_names == field_names
    assert dataloader.vertices.shape == (n_cells, n_dimensions)
    assert dataloader.weights.shape == dataloader.levels.shape
    assert dataloader.faces.shape == (n_cells, pow(2, n_dimensions))
    assert dataloader.nodes.shape == (n_nodes, n_dimensions)
    assert dataloader.load_snapshot("p", "0.4").shape == (n_cells, 1)


if __name__ == "__main__":
    pass
