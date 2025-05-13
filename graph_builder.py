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
    :param cutoff: Distance cutoff for neighbor determination.
    :return: PyTorch Geometric Data object with node features, edges, etc.
    """

    
    # 1) Build node features
    atomic_numbers = []
    for site in structure.sites:
        # If partial occupancy, you might combine site.species.elements 
        # but for a single-element site, this works fine:
        element = list(site.species.elements)[0]
        atomic_numbers.append(element.Z)
    
    x = torch.tensor(atomic_numbers, dtype=torch.float).view(-1, 1)

    
    # 2) Get lattice features (3x3 matrix flattened to 9 values)
    # Reshape to (1, 9) to ensure proper batching
    lattice_matrix = structure.lattice.matrix
    lattice_features = torch.tensor(lattice_matrix.flatten(), dtype=torch.float).view(1, 9)

    
    # 3) Determine edges (neighbors within cutoff)
    #    i_indices and j_indices are the site indices, distances is the site distance
    i_indices, j_indices, images, distances = structure.get_neighbor_list(r=cutoff)
    edge_index = torch.tensor([i_indices, j_indices], dtype=torch.long)
    edge_attr = torch.tensor(distances, dtype=torch.float).view(-1, 1)

   
    # 4) Create the Data object
    data = Data(
        x=x,                    # Node features
        edge_index=edge_index,  # Graph connectivity
        edge_attr=edge_attr,    # Distances
        lattice_features=lattice_features  # Add lattice features with shape (1, 9)
    )

    # Optionally stores label
    if label is not None:
        if isinstance(label,str):
            label = True if label == "True" else False
        data.y = torch.tensor([float(label)], dtype=torch.float)
    
    return data


def build_graph_batch(
    data_list,
    cutoff: float = 5.0,
    label_key: str = "label"
):
    results = []
    for item in data_list:
        # Case 1: dict with structure, label, etc.
        if isinstance(item, tuple) and len(item) == 3:
            structure = item[1]
            # handle label if present
            if label_key in item:
                label = item[2]
            # or item.get("is_metal") if that's how you store it
            else:
                label = item[2]
            data = build_graph(structure, label=label, cutoff=cutoff)

        # Case 2: tuple of (crystalproperites, structure)
        elif isinstance(item, tuple) and len(item) == 2:
            (structure, label) = item
            data = build_graph(item[1], item[0].is_metal, cutoff=cutoff)

        # Case 3: just a Structure
        else:
            structure = item
            data = build_graph(structure, label=None, cutoff=cutoff)

        results.append(data)
    
    
    return results