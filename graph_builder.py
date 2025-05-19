import torch
import numpy as np
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

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
    # Normalize the lattice matrix by dividing by the maximum absolute value
    lattice_matrix = structure.lattice.matrix
    # Normalize by the maximum absolute value to keep values in a reasonable range
    # Add a small epsilon to prevent division by zero if max_abs_val is zero (e.g. vacuum structure with no lattice)
    max_abs_val = np.max(np.abs(lattice_matrix))
    if max_abs_val > 1e-8: # only normalize if max_abs_val is not zero
        lattice_matrix = lattice_matrix / max_abs_val
    else: # if lattice_matrix is all zeros or very small values.
        lattice_matrix = lattice_matrix # keep it as is or handle as an error/special case if needed

    # Reshape to (1, 9) to ensure proper batching
    lattice_features = torch.tensor(lattice_matrix.flatten(), dtype=torch.float).view(1, 9)

    
    # 3) Determine edges (neighbors within cutoff)
    #    i_indices and j_indices are the site indices, distances is the site distance
    i_indices, j_indices, images, distances = structure.get_neighbor_list(r=cutoff)
    
    # Handle case where no edges are found
    if len(i_indices) == 0:
        # Create a self-loop for each node to ensure connectivity
        num_nodes = len(structure)
        if num_nodes == 0: # Handle empty structure
            # Return an empty graph or handle as error
            # For now, creating a graph with no nodes/edges but with lattice features
            return Data(
                x=torch.empty((0,1), dtype=torch.float),
                edge_index=torch.empty((2,0), dtype=torch.long),
                edge_attr=torch.empty((0,4), dtype=torch.float), # 4 for [distance, dx, dy, dz]
                lattice_features=lattice_features,
                y=torch.tensor([float(label if label is not None else np.nan)], dtype=torch.float) if label is not None or label_key == "label" else None # ensure y is present if expected
            )

        i_indices = np.arange(num_nodes)
        j_indices = np.arange(num_nodes)
        distances = np.zeros(num_nodes)
        images = np.zeros((num_nodes, 3), dtype=int) # Ensure images is int type
    
    edge_index = torch.tensor([i_indices, j_indices], dtype=torch.long)
    
    # Calculate direction vectors for each edge
    direction_vectors = []
    for i, j, image in zip(i_indices, j_indices, images):
        # Get positions of the two sites
        pos_i = structure[i].coords
        pos_j = structure[j].coords
        
        # Add periodic image displacement
        displacement = np.dot(image, structure.lattice.matrix)
        pos_j_displaced = pos_j + displacement # use a new variable for clarity
        
        # Calculate direction vector
        direction = pos_j_displaced - pos_i
        # Handle zero-length vectors (self-loops)
        norm = np.linalg.norm(direction)
        if norm > 1e-10:  # Use small threshold to avoid numerical issues
            direction = direction / norm
        else:
            # For true self-loops (i==j and image is zero), direction is ill-defined.
            # Defaulting to [0,0,0] for self-loops if distance is 0,
            # or a fixed vector if we must have a non-zero direction.
            # Given distances[k] for self-loops made above is 0, [0,0,0] is consistent.
            if i == j and np.all(image == 0):
                 direction = np.array([0.0, 0.0, 0.0]) # Zero vector for true self-loops
            else: # For other near-zero norm cases or forced self-loops that might not be at distance 0
                 direction = np.array([1.0, 0.0, 0.0]) # Default direction
        direction_vectors.append(direction)
    
    # Ensure distances is 2D array
    distances = distances.reshape(-1, 1)
    direction_vectors = np.array(direction_vectors, dtype=float) # ensure float type
    
    # Combine distances and direction vectors into edge attributes
    # [distance, dx, dy, dz] for each edge
    edge_attr = np.hstack([distances, direction_vectors])
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

   
    # 4) Create the Data object
    data = Data(
        x=x,                    # Node features
        edge_index=edge_index,  # Graph connectivity
        edge_attr=edge_attr,    # Edge attributes [distance, dx, dy, dz]
        lattice_features=lattice_features  # Add lattice features with shape (1, 9)
    )

    # Optionally stores label
    if label is not None:
        processed_label = label
        if isinstance(label, str):
            if label.lower() == "true":
                processed_label = True
            elif label.lower() == "false":
                processed_label = False
            else:
                try:
                    processed_label = float(label) # Try to convert to float if not "true"/"false"
                except ValueError:
                    print(f"Warning: Could not convert label string '{label}' to float. Setting label to NaN.")
                    processed_label = np.nan # Or handle as an error
        
        # Ensure the label is float before converting to tensor
        try:
            data.y = torch.tensor([float(processed_label)], dtype=torch.float)
        except ValueError: # Catch cases like float(None) if processed_label becomes None
             data.y = torch.tensor([np.nan], dtype=torch.float) # Or handle differently
    
    return data


def build_graph_batch(
    data_list,
    cutoff: float = 5.0,
    label_key: str = "label" 
):
    results = []
    for i, item in enumerate(data_list):
        data_obj = None  # Initialize data_obj for each item

        # Priority 1: Custom 'crystal' object from load_data_local (duck-typing)
        # These objects are expected to have .structure and .is_metal attributes.
        if hasattr(item, 'structure') and hasattr(item, 'is_metal'):
            if isinstance(item.structure, Structure):
                structure_to_process = item.structure
                label_to_process = item.is_metal
                data_obj = build_graph(structure_to_process, label=label_to_process, cutoff=cutoff)
            else:
                print(f"Warning: Item {i} (type: {type(item)}) has .structure but it's not a Pymatgen Structure (type: {type(item.structure)}). Skipping.")
        
        # Priority 2: Tuple of 3 - e.g., (identifier, pymatgen_structure, label_value)
        # Based on original 'Case 1' logic: structure = item[1], label = item[2]
        elif isinstance(item, tuple) and len(item) == 3:
            structure_part = item[1]
            label_part = item[2]
            if isinstance(structure_part, Structure):
                data_obj = build_graph(structure_part, label=label_part, cutoff=cutoff)
            else:
                print(f"Warning: Item {i} is a 3-tuple, but element 1 (expected Pymatgen Structure) is type {type(structure_part)}. Skipping.")
        
        # Priority 3: Tuple of 2 - e.g., (properties_object, pymatgen_structure)
        # Based on original 'Case 2' logic: structure = item[1], label = item[0].is_metal
        elif isinstance(item, tuple) and len(item) == 2:
            props_part = item[0]
            structure_part = item[1]
            if isinstance(structure_part, Structure):
                if hasattr(props_part, 'is_metal'):
                    label_to_process = props_part.is_metal
                    data_obj = build_graph(structure_part, label=label_to_process, cutoff=cutoff)
                else:
                    # If props_part doesn't have 'is_metal', build graph without an explicit label from props_part.
                    # You might decide to extract a label differently or raise an error if essential.
                    print(f"Warning: Item {i} is a 2-tuple with a valid Structure, but element 0 (type: {type(props_part)}) lacks an 'is_metal' attribute. Building graph without this specific label.")
                    data_obj = build_graph(structure_part, label=None, cutoff=cutoff)
            else:
                print(f"Warning: Item {i} is a 2-tuple, but element 1 (expected Pymatgen Structure) is type {type(structure_part)}. Skipping.")

        # Priority 4: Raw Pymatgen Structure object
        elif isinstance(item, Structure):
            data_obj = build_graph(item, label=None, cutoff=cutoff) # No explicit label provided for a raw structure
        
        # If item was not processed by any of the above conditional blocks
        if data_obj is not None:
            results.append(data_obj)
        else:
            # This 'else' corresponds to data_obj remaining None after all checks.
            # This means the 'if/elif' chain was exhausted without a match, or a condition within a branch failed (e.g. structure not Structure instance)
            # The specific print warnings within the branches already cover most failure reasons.
            # This is a fallback for any unhandled item types.
            if not (hasattr(item, 'structure') and hasattr(item, 'is_metal')) and \
               not (isinstance(item, tuple) and (len(item) == 3 or len(item) == 2)) and \
               not isinstance(item, Structure):
                 print(f"Warning: Item {i} (type: {type(item)}) did not match any known data format for graph building. Skipping.")
            
    return results