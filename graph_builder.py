import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure

def build_graph(
    structure: Structure,
    label: float = None,
    cutoff: float = 5.0
) -> Data:
    """
    Convert a pymatgen Structure into a torch_geometric.data.Data object.

    :param structure: A pymatgen.core.structure.Structure object
    :param label: A numeric label for this crystal (e.g., 0.0/1.0 for is_metal)
    :param cutoff: Distance cutoff for neighbor determination (in Å)
    :return: PyTorch Geometric Data object with node features, edges, etc.
    """

    # --------------------
    # 1) Build node features
    # --------------------
    atomic_numbers = []
    for site in structure.sites:
        # If there's only one species per site, this is straightforward:
        element = list(site.species.elements)[0]
        # Or, if partial occupancy, you might combine multiple elements.
        atomic_numbers.append(element.Z)
    
    # Here, node_features=1 => we have just the atomic number for now.
    x = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)

    # --------------------
    # 2) Determine edges (neighbors within cutoff)
    # --------------------
    # pymatgen provides a get_neighbor_list method:
    # i_indices, j_indices, distances = structure.get_neighbor_list(r=cutoff)
    # which returns arrays of neighbor indices (i, j) and their distances.
    i_indices, j_indices, distances = structure.get_neighbor_list(r=cutoff)
    
    # Construct edge_index: a [2, num_edges] tensor listing which nodes are connected
    edge_index = torch.tensor([i_indices, j_indices], dtype=torch.long)

    # We can store distances (and possibly other features) as edge_attr
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)

    # --------------------
    # 3) Create Data object
    # --------------------
    data = Data(
        x=x,                  # Node features
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # If you have a label (e.g., 1 = metal, 0 = not metal), store it
    if label is not None:
        # shape [1] so it's consistent with how PyTorch Geometric expects a label
        data.y = torch.tensor([label], dtype=torch.float)

    return data


def batch_structures_to_graphs(data_list, cutoff=5.0):
    """
    Given a list of dictionaries, each containing a 'structure' (pymatgen Structure)
    and possibly some label info, convert each to a PyG Data object.
    For example, data_list might come from get_materials_with_structure() in data_retrival.
    
    :param data_list: [
        {
          "material_id": "mp-149",
          "band_gap": 1.2,
          "is_metal": True,
          "structure": <pymatgen.core.structure.Structure object>,
          ...
        },
        ...
      ]
    :param cutoff: neighbor cutoff distance
    :return: List of (material_id, PyG Data) pairs or just [Data, Data, ...]
    """

    all_graphs = []
    for entry in data_list:
        structure = entry["structure"]
        
        
        if "is_metal" in entry and entry["is_metal"] is not None:
            label_val = 1.0 if entry["is_metal"] else 0.0
        else:
            label_val = None
        
        graph = structure_to_pygdata(structure, label=label_val, cutoff=cutoff)
        all_graphs.append((entry["material_id"], graph))

    return all_graphs