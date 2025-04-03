from crystalclass import crystal
from mp_api.client import MPRester
from monty.json import MontyEncoder, MontyDecoder
import json
import os

#ik abuse min key :)
API_KEY = 'JlJnqQljbO40Ooy3aGrORBR29rhD6eXO'

#size is the amount of crystals we wanted to importet
def get_materials_data(size):
    with MPRester(API_KEY) as mpr:
        # Get summaries
        summaries = list(mpr.materials.summary.search(
            fields = ["material_id","band_gap", "cbm", "density", "density_atomic", "dos_energy_down", "dos_energy_up", "e_electronic",
                    "e_ij_max", "e_ionic", "e_total", "efermi", "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                    "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor", "surface_anisotropy", "total_magnetization",
                    "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol", "uncorrected_energy_per_atom",
                    "universal_anisotropy", "vbm", "volume", "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                    "weighted_work_function","is_metal"],
            chunk_size=size,
            num_chunks=1
        ))[:size]

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
        #Skips header
        next(file)

        for i, line in enumerate(file):
            if amount is not None and i >= amount:
                break
            try:
                # Find the position where the JSON structure starts
                split_index = line.find(', "{"')
                if split_index == -1:
                    print(f"Warning: No JSON structure found in line {i+1}")
                    continue

                crystal_part = line[:split_index]
                json_part = line[split_index + 2:].strip()
                
                # Remove the surrounding quotes from the JSON string
                if json_part.startswith('"') and json_part.endswith('"'):
                    json_part = json_part[1:-1]
                
                # Unescape any escaped quotes
                json_part = json_part.replace('\\"', '"')

                # Parse crystal properties first
                crystal_data = crystal_part.split(", ")
                converted = []
                for val in crystal_data:
                    if val == "None":
                        converted.append(None)
                    elif val.replace('.','',1).replace('-','',1).isdigit():
                        if 'e' in val.lower():
                            converted.append(float(val))
                        else:
                            if float(val).is_integer():
                                converted.append(int(float(val)))
                            else:
                                converted.append(float(val))
                    else:
                        converted.append(val)

                # Try to parse the structure
                try:
                    structure = json.loads(json_part, cls=MontyDecoder)
                    
                    # Verify we have a proper Structure object
                    if not hasattr(structure, 'sites'):
                        print(f"Warning: Structure at line {i+1} is not a valid Structure object")
                        continue
                        
                except json.JSONDecodeError as e:
                    print(f"JSON decode error at line {i+1}: {e}")
                    print(f"Problematic JSON: {json_part[:100]}...")
                    continue
                except Exception as e:
                    print(f"Error parsing structure at line {i+1}: {str(e)}")
                    continue

                # Create crystal object and add to data
                try:
                    crystal_obj = crystal(*converted)
                    data.append((crystal_obj, structure))
                    print(f"Successfully loaded crystal {i+1}")
                except Exception as e:
                    print(f"Error creating crystal object at line {i+1}: {str(e)}")
                    continue

            except Exception as e:
                print(f"Error processing line {i+1}: {str(e)}")
                print(f"Problematic line: {line[:100]}...")
                continue
    
    print(f"Successfully loaded {len(data)} valid crystal structures")
    if len(data) == 0:
        print("Sample of problematic line:")
        with open(filepath, "r") as file:
            next(file)  # Skip header
            sample_line = next(file)
            print(f"First 500 chars: {sample_line[:500]}")
            print(f"Last 500 chars: {sample_line[-500:]}")
    
    return data