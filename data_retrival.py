from crystalclass import crystal
from mp_api.client import MPRester
from monty.json import MontyEncoder, MontyDecoder
from itertools import islice
import json
import os

#this is the API key for the Materials Project. You can get your own key by signing up at https://materialsproject.org
API_KEY = 'JlJnqQljbO40Ooy3aGrORBR29rhD6eXO'

#size is the amount of crystals we wanted to importet
def get_materials_data(size):
    with MPRester(API_KEY) as mpr:
        # Get summaries
        docs = mpr.materials.summary.search(
            fields = ["material_id","band_gap", "cbm", "density", "density_atomic", "dos_energy_down", "dos_energy_up", "e_electronic",
                    "e_ij_max", "e_ionic", "e_total", "efermi", "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                    "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor", "surface_anisotropy", "total_magnetization",
                    "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol", "uncorrected_energy_per_atom",
                    "universal_anisotropy", "vbm", "volume", "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                    "weighted_work_function","is_metal"],
            chunk_size=1000,
            num_chunks=None
        )
        summaries = list(islice(docs, size))


        results = []
        for summary in summaries:
            try:
                structure = mpr.get_structure_by_material_id(summary.material_id)
                results.append((summary, structure))
            except Exception as e:
                print(f"Failed to get structure for {summary.material_id}: {e}")
                continue
        return results
    
def get_single_materials(name):
    mpname = "mp-"+str(name)
    with MPRester(API_KEY) as mpr:
        # Get summaries
        docs = mpr.materials.summary.search(
            material_ids=[mpname],
            fields = ["material_id","band_gap", "cbm", "density", "density_atomic", "dos_energy_down", "dos_energy_up", "e_electronic",
                    "e_ij_max", "e_ionic", "e_total", "efermi", "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                    "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor", "surface_anisotropy", "total_magnetization",
                    "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol", "uncorrected_energy_per_atom",
                    "universal_anisotropy", "vbm", "volume", "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                    "weighted_work_function","is_metal"],
            
        )
        summaries = list(islice(docs, 1))


        results = []
        for summary in summaries:
            try:
                structure = mpr.get_structure_by_material_id(summary.material_id)
                results.append((summary, structure))
            except Exception as e:
                print(f"Failed to get structure for {summary.material_id}: {e}")
                continue
        return results

#filename is what it is called if we want differnt file thingymabobs to have fun with, we always get the meterials varible from get_materails_data()   
def save_data_local(filename,data):
    os.makedirs("DownloadedCrystalProperties",exist_ok=True)

    with open(f"DownloadedCrystalProperties/{filename}.txt", "w") as file:
        if not data:
              return
         
        requested_fields = ["material_id","band_gap", "cbm", "density", "density_atomic", "dos_energy_down", "dos_energy_up", "e_electronic",
                    "e_ij_max", "e_ionic", "e_total", "efermi", "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                    "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor", "surface_anisotropy", "total_magnetization",
                    "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol", "uncorrected_energy_per_atom",
                    "universal_anisotropy", "vbm", "volume", "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                    "weighted_work_function","is_metal"]
        
        file.write(", ".join(requested_fields) + ", structure.json\n")

        for summary, structure in data:
            values = []
            for field in requested_fields:
                val = getattr(summary, field)
                values.append(str(val) if val is not None else "None")

            structure_json = json.dumps(structure, cls=MontyEncoder)
            file.write(", ".join(values) + f', {structure_json}\n')

def load_data_local(filename, amount=None):
    filepath = f"DownloadedCrystalProperties/{filename}.txt"
    data = []

    with open(filepath, "r") as file:
        # Skip header
        header = next(file).strip().split(", ")
        
        # Find indices of required fields
        material_id_idx = header.index("material_id")
        density_idx = header.index("density")
        is_metal_idx = header.index("is_metal")

        for i, line in enumerate(file):
            if amount is not None and i >= amount:
                break
            try:
                split_index = line.find(', {')
                if split_index == -1:
                    continue

                crystal_part = line[:split_index]
                json_part = line[split_index + 2:].strip()

                # Parse structure
                try:
                    structure = json.loads(json_part, cls=MontyDecoder)
                except json.JSONDecodeError:
                    try:
                        structure = json.loads(f'"{json_part}"', cls=MontyDecoder)
                    except:
                        json_part = json_part.replace('"', '\\"')
                        structure = json.loads(f'"{json_part}"', cls=MontyDecoder)

                # Parse crystal data
                crystal_data = crystal_part.split(", ")
                
                # Extract only the required fields
                material_id = crystal_data[material_id_idx]
                density = float(crystal_data[density_idx]) if crystal_data[density_idx] != "None" else None
                is_metal = crystal_data[is_metal_idx].lower() == "true" if crystal_data[is_metal_idx] != "None" else None

                # Create crystal object with only required fields
                crystal_obj = crystal(
                    material_id=material_id,
                    structure=structure,
                    density=density,
                    is_metal=is_metal
                )

                data.append(crystal_obj)

            except Exception as e:
                print(f"Error loading line {i+1}: {str(e)}")
                print(f"Problematic line: {line[:200]}...")  # Print first 200 chars of problematic line
                continue
    return data