import pytest
import torch
from pymatgen.core import Lattice, Structure
from graph_builder import build_graph, batch_structures_to_graphs
from torch_geometric.data import Data

@pytest.fixture
def simple_structure():
    # A 2-atom simple cubic structure with hydrogen
    lattice = Lattice.cubic(5.0)
    structure = Structure(
        lattice,
        ["H", "H"],
        [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    return structure

def test_build_graph_structure(simple_structure):
    label = 1.0
    data = build_graph(simple_structure, label=label, cutoff=5.0)

    assert isinstance(data, Data)
    assert data.x.shape == (2, 1)  # 2 atoms, 1 feature (atomic number)
    assert data.edge_index.shape[0] == 2  # shape [2, num_edges]
    assert data.edge_attr.shape[0] == data.edge_index.shape[1]
    assert data.y.shape == (1,)
    assert data.y.item() == label

def test_build_graph_no_label(simple_structure):
    data = build_graph(simple_structure, label=None, cutoff=5.0)

    assert isinstance(data, Data)
    assert hasattr(data, 'x')
    assert not hasattr(data, 'y')  # Should not have label

def test_batch_structures_to_graphs(simple_structure):
    test_data_list = [
        {
            "material_id": "mp-test-001",
            "band_gap": 1.23,
            "is_metal": False,
            "structure": simple_structure
        },
        {
            "material_id": "mp-test-002",
            "is_metal": True,
            "structure": simple_structure
        },
        {
            "material_id": "mp-test-003",
            "structure": simple_structure  # No label at all
        }
    ]

    results = batch_structures_to_graphs(test_data_list, cutoff=5.0)

    assert len(results) == 3
    for mid, data in results:
        assert isinstance(mid, str)
        assert isinstance(data, Data)
        assert data.x.shape[0] == 2
        assert data.edge_index.shape[0] == 2

    # Check that first label is 0.0, second is 1.0, third has no label
    assert results[0][1].y.item() == 0.0
    assert results[1][1].y.item() == 1.0
    assert not hasattr(results[2][1], "y")