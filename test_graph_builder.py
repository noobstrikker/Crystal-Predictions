import pytest
import torch
from pymatgen.core import Lattice, Structure
from torch_geometric.data import Data

# Adjust the import to match your actual module/file name:
from graph_builder import build_graph, build_graph_batch

@pytest.fixture
def dummy_structure():
    """
    Creates a trivial 2-atom cubic structure so we can test graph-building.
    We'll just say it's a cubic 5.0 Å box with two hydrogen atoms at 
    fractional coords (0,0,0) and (0.5, 0.5, 0.5).
    """
    lattice = Lattice.cubic(5.0)
    coords = [[0, 0, 0], [0.5, 0.5, 0.5]]
    species = ["H", "H"]
    return Structure(lattice, species, coords)

def test_build_graph_with_label(dummy_structure):
    """
    Test that build_graph returns a PyTorch Geometric Data object 
    with the correct attributes when a label is provided.
    """
    label = 1.0
    data = build_graph(dummy_structure, label=label, cutoff=5.0)

    # Check that we got the correct type
    assert isinstance(data, Data), "Expected a Data object from build_graph"

    # Node features
    assert hasattr(data, 'x'), "Data object should have an x attribute"
    assert data.x.shape == (2, 1), "Expected 2 atoms, each with 1 feature (atomic number)"

    # Edges
    assert hasattr(data, 'edge_index'), "Data object should have an edge_index"
    assert data.edge_index.shape[0] == 2, "edge_index should have shape [2, num_edges]"
    num_edges = data.edge_index.shape[1]
    assert num_edges >= 1, "We should have at least 1 edge if both atoms are within the cutoff"

    # Edge attributes
    assert hasattr(data, 'edge_attr'), "Data object should have edge_attr for distances"
    assert data.edge_attr.shape[0] == num_edges, "edge_attr length should match the number of edges"
    assert data.edge_attr.shape[1] == 1, "Each edge should have a single distance value"

    # Label
    assert hasattr(data, 'y'), "Data object should have a label when provided"
    assert data.y.shape == (1,), "Expected a single label value"
    assert pytest.approx(data.y.item()) == label, "Label mismatch"

def test_build_graph_no_label(dummy_structure):
    """
    If no label is provided, we expect the Data object not to have a y attribute.
    """
    data = build_graph(dummy_structure, label=None, cutoff=5.0)
    assert isinstance(data, Data)
    assert hasattr(data, 'x')
    assert not hasattr(data, 'y'), "Did not expect a label (y) when label=None"

def test_build_graph_batch_dicts(dummy_structure):
    """
    Test build_graph_batch with a list of dictionaries. 
    Some have labels (is_metal) and some do not.
    """
    data_list = [
        {
            "material_id": "mp-test-001",
            "is_metal": False,  # should become label=0.0
            "structure": dummy_structure
        },
        {
            "material_id": "mp-test-002",
            "is_metal": True,   # should become label=1.0
            "structure": dummy_structure
        },
        {
            "material_id": "mp-test-003",
            # No label at all
            "structure": dummy_structure
        },
    ]

    results = build_graph_batch(data_list, cutoff=5.0, label_key="is_metal")
    # results should be a list of Data objects, or possibly something else 
    # depending on your final design; if the function returns just Data objects:
    
    assert len(results) == 3, "We should get 3 items back"
    for i, data in enumerate(results):
        assert isinstance(data, Data), "Each item should be a Data object"
        # Check shapes
        assert data.x.shape == (2, 1), "Expect 2-atom structure"
        num_edges = data.edge_index.shape[1]
        assert data.edge_attr.shape[0] == num_edges
        
        if i == 0:
            # is_metal=False => label=0.0
            assert hasattr(data, 'y')
            assert data.y.item() == 0.0
        elif i == 1:
            # is_metal=True => label=1.0
            assert hasattr(data, 'y')
            assert data.y.item() == 1.0
        else:
            # No label
            assert not hasattr(data, 'y'), "Third item should not have a label"

def test_build_graph_batch_tuples(dummy_structure):
    """
    Test build_graph_batch with a list of tuples in (structure, label) format.
    """
    data_list = [
        (dummy_structure, 0.0),
        (dummy_structure, 1.0),
        (dummy_structure, None)  # no label
    ]

    results = build_graph_batch(data_list, cutoff=5.0)
    assert len(results) == 3, "Expected 3 Data objects"

    # Check each
    data0 = results[0]
    data1 = results[1]
    data2 = results[2]

    # 1) data0 => label=0.0
    assert isinstance(data0, Data)
    assert data0.y.item() == 0.0

    # 2) data1 => label=1.0
    assert isinstance(data1, Data)
    assert data1.y.item() == 1.0

    # 3) data2 => None => no label
    assert isinstance(data2, Data)
    assert not hasattr(data2, 'y'), "Expected no label for the third item"