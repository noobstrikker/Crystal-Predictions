from crystalclass import crystal
from mp_api.client import MPRester
from monty.json import MontyEncoder, MontyDecoder
import json
import os
from itertools import islice

#ik abuse min key :)
API_KEY = 'JlJnqQljbO40Ooy3aGrORBR29rhD6eXO'

#size is the amount of crystals we wanted to importet
def get_materials_data(size):
    """
    Return *size* (summary, structure) tuples.

    * Uses mp_api ≥ 0.40 auto‑pagination, so it never asks the server
      for >10 000 docs in one request.
    * Keeps memory low by streaming.
    * No other parts of data_retrival.py have to change.
    """
    requested_fields = [
        "material_id","band_gap","cbm","density","density_atomic",
        "dos_energy_down","dos_energy_up","e_electronic","e_ij_max",
        "e_ionic","e_total","efermi","energy_above_hull","energy_per_atom",
        "equilibrium_reaction_energy_per_atom","formation_energy_per_atom",
        "homogeneous_poisson","n","shape_factor","surface_anisotropy",
        "total_magnetization","total_magnetization_normalized_formula_units",
        "total_magnetization_normalized_vol","uncorrected_energy_per_atom",
        "universal_anisotropy","vbm","volume","weighted_surface_energy",
        "weighted_surface_energy_EV_PER_ANG2","weighted_work_function",
        "is_metal",
    ]

    results = []
    with MPRester(API_KEY) as mpr:
        # iterator that paginates under the hood (10 k/request max)
        docs = mpr.materials.summary.search(fields=requested_fields)

        for summary in islice(docs, size):          # pull exactly `size`
            try:
                structure = mpr.get_structure_by_material_id(summary.material_id)
                results.append((summary, structure))
            except Exception as err:
                print(f"[WARN] skipping {summary.material_id}: {err}")

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
                split_index = line.find(', {')
                if split_index == -1:
                    continue

                crystal_part = line[:split_index]
                json_part = line[split_index +2:].strip()

                json_part = json_part.rstrip('"').strip()

                try:
                    structure = json.loads(json_part, cls=MontyDecoder)
                except json.JSONDecodeError:
                    try:
                        structure = json.loads(f'"{json_part}"', cls=MontyDecoder)
                    except:
                        json_part = json_part.replace('"', '\\"')
                        structure = json.loads(f'"{json_part}"', cls=MontyDecoder)
                crystal_data = crystal_part.split(", ")
                converted = []
                for val in crystal_data:
                    if val == "None":
                        converted.append("Null")
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

                crystal_obj = crystal(*converted)

                data.append((crystal_obj, structure))

            except Exception as e:
                print(f"Error loading line {i+1}: {str(e)}")
                print(f"Problematic line: {line[:200]}...")  # Print first 200 chars of problematic line
                continue
    return data