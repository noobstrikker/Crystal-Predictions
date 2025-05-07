import torch
from torch_geometric.data import Data
from pymatgen.core.structure import Structure
import numpy as np
from crystalclass import crystal

def get_lattice_features(structure):
    """Extract lattice features from a pymatgen Structure object"""
    lattice = structure.lattice
    
    # Get basic lattice parameters
    a, b, c = lattice.abc  # lengths of lattice vectors
    alpha, beta, gamma = lattice.angles  # angles between lattice vectors
    volume = lattice.volume
    density = structure.density
    
    # Combine features
    features = [
        a, b, c,  # lattice parameters
        alpha, beta, gamma,  # angles
        volume,  # cell volume
        density,  # density
        len(structure),  # number of atoms
    ]
    
    # Ensure we have exactly 9 features and reshape to (1, 9)
    return torch.tensor(features[:9], dtype=torch.float).view(1, -1)

def build_graph(
    structure: Structure,
    label: float = None,
    cutoff: float = 5.0
) -> Data:
    """Build a graph from a crystal structure
    
    Args:
        structure: pymatgen Structure object
        label: Optional label for the graph (is_metal)
        cutoff: Cutoff radius for building edges
        
    Returns:
        Data object containing the graph representation
    """
    # 1) Get lattice features
    lattice_features = get_lattice_features(structure)  # Shape: (1, 9)
    
    # 2) Create node features for each atom
    num_atoms = len(structure)
    node_features = []
    
    for i in range(num_atoms):
        # Combine lattice features with atomic features
        atom = structure[i]
        specie = atom.specie
        
        # Get atomic properties with fallback values
        atomic_number = float(specie.Z)
        
        # Handle atomic mass
        try:
            atomic_mass = float(specie.atomic_mass)
        except (AttributeError, TypeError):
            atomic_mass = 0.0
            
        # Handle atomic radius
        try:
            atomic_radius = float(specie.atomic_radius)
        except (AttributeError, TypeError):
            atomic_radius = 0.0
        
        # Get coordinates
        x, y, z = atom.coords
        
        # Combine features
        atom_features = [
            atomic_number,
            atomic_mass,
            atomic_radius,
            float(x), float(y), float(z)
        ]
        node_features.append(atom_features)
    
    x = torch.tensor(node_features, dtype=torch.float)
    
    # 3) Get edges based on periodic neighbors within cutoff
    all_neighbors = structure.get_all_neighbors(cutoff, include_index=True)
    
    i_indices = []
    j_indices = []
    distances = []
    images = []
    
    for i, neighbors in enumerate(all_neighbors):
        for neighbor, dist, j, image in neighbors:
            i_indices.append(i)
            j_indices.append(j)
            distances.append(dist)
            images.append(image)
    
    edge_index = torch.tensor([i_indices, j_indices], dtype=torch.long)
    
    # 4) Get edge features
    edge_features = []
    for idx in range(len(i_indices)):
        i, j = i_indices[idx], j_indices[idx]
        dist = distances[idx]
        img = images[idx]
        
        # Calculate edge features
        # 1. Distance
        # 2. Whether the edge crosses a periodic boundary
        crosses_boundary = any(coord != 0 for coord in img)
        
        edge_feat = [dist, float(crosses_boundary)]
        edge_features.append(edge_feat)
    
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    # 5) Create the Data object
    data = Data(
        x=x,                    # Node features
        edge_index=edge_index,  # Graph connectivity
        edge_attr=edge_attr,    # Edge features
        lattice_features=lattice_features  # Store lattice features separately
    )

    # Optionally stores label
    if label is not None:
        if isinstance(label, str):
            label = True if label == "True" else False
        data.y = torch.tensor([float(label)], dtype=torch.float)
    
    return data

def build_graph_batch(
    data_list,
    cutoff: float = 5.0,
    label_key: str = "label"
):
    """Build a batch of graphs from a list of crystal data
    
    Args:
        data_list: List of crystal objects or (structure, label) tuples
        cutoff: Cutoff radius for building edges
        label_key: Key for label in dictionary data
        
    Returns:
        List of Data objects
    """
    results = []
    for item in data_list:
        # Case 1: crystal object
        if isinstance(item, crystal):
            structure = item.structure
            label = item.is_metal
            
        # Case 2: (structure, label) tuple
        elif isinstance(item, tuple) and len(item) == 2:
            structure, label = item
            
        # Case 3: dictionary
        elif isinstance(item, dict):
            structure = Structure.from_dict(item["structure.json"])
            label = item[label_key]
            
        else:
            raise ValueError(f"Unsupported data type: {type(item)}")
            
        graph = build_graph(structure, label, cutoff)
        results.append(graph)
        
    return results