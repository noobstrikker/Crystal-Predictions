import torch
import numpy as np
from pymatgen.core.structure import Structure
from torch_geometric.data import Data

def build_graph(structure: Structure, crystal_obj=None, label=None, cutoff: float = 5.0):
    """Convert structure to PyG Data object with electronic properties"""
    try:
        # Node features - atomic number only
        x = torch.tensor([
            [list(site.species.elements)[0].Z]  # Atomic number
            for site in structure.sites
        ], dtype=torch.float)

        # Edge connections
        i, j, _, dist = structure.get_neighbor_list(r=cutoff)
        edge_index = torch.tensor(np.vstack([i, j]), dtype=torch.long)
        edge_attr = torch.tensor(dist, dtype=torch.float).view(-1, 1)

        # Global features - always return 2 features (band_gap and cbm)
        band_gap = 0.0  # Default value
        cbm = 0.0       # Default value
        
        if crystal_obj is not None:
            # Handle band_gap
            bg = getattr(crystal_obj, 'band_gap', None)
            if bg is not None and str(bg).lower() != 'null':
                try:
                    band_gap = float(bg)
                except (ValueError, TypeError):
                    pass
            
            # Handle cbm
            cb = getattr(crystal_obj, 'cbm', None)
            if cb is not None and str(cb).lower() != 'null':
                try:
                    cbm = float(cb)
                except (ValueError, TypeError):
                    pass
        
        # Always return both features (default to 0.0 if missing)
        global_features = torch.tensor([[band_gap, cbm]], dtype=torch.float)

        # Target label
        if label is not None:
            if isinstance(label, str):
                y = torch.tensor([float(label.lower() == "true")], dtype=torch.long)
            else:  # Assume boolean/numeric
                y = torch.tensor([float(bool(label))], dtype=torch.long)
        else:
            y = None

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            global_features=global_features
        )
    except Exception as e:
        print(f"Error building graph for {getattr(crystal_obj, 'material_id', 'unknown')}: {str(e)}")
        return None

def build_graph_batch(data_list, cutoff: float = 5.0):
    """Process multiple crystals into graph representations
    
    Args:
        data_list: List of input items (Structure objects or tuples)
        cutoff: Radius for neighbor calculations (Å)
        
    Returns:
        List of valid Data objects (skips invalid entries)
    """
    results = []
    for item in data_list:
        if item is None:
            continue
            
        try:
            # Handle different input formats
            if isinstance(item, tuple):
                if len(item) == 3:
                    crystal_obj, structure, label = item
                elif len(item) == 2:
                    crystal_obj, structure = item
                    label = getattr(crystal_obj, 'is_metal', None)
                else:
                    print(f"Unexpected tuple length {len(item)}")
                    continue
            else:
                structure = item
                crystal_obj = None
                label = None
                
            graph = build_graph(structure, crystal_obj, label, cutoff)
            if graph is not None:
                results.append(graph)
                
        except Exception as e:
            material_id = getattr(item[0], 'material_id', 'unknown') if isinstance(item, tuple) else 'unknown'
            print(f"Skipped {material_id}: {str(e)}")
    
    return results