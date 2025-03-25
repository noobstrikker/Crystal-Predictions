from mp_api.client import MPRester
from pymatgen.core import Structure
from crystalclass import crystal

API_KEY = 'YOUR_API_KEY'


def get_materails_data(size):
    """
    Fetch a batch of materials from the Materials Project summary API
    and return them as 'summary doc' objects. This includes numeric properties
    but NOT the actual crystal structure.
    """
    with MPRester(API_KEY) as mpr:
        # Fetch
        materials = mpr.materials.summary.search(
            fields=[
                "material_id", "band_gap", "cbm", "density", "density_atomic", "dos_energy_down",
                "dos_energy_up", "e_electronic", "e_ij_max", "e_ionic", "e_total", "efermi",
                "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor",
                "surface_anisotropy", "total_magnetization",
                "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol",
                "uncorrected_energy_per_atom", "universal_anisotropy", "vbm", "volume",
                "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                "weighted_work_function", "is_metal"
            ],
            num_chunks=1,
            chunk_size=size
        )
        return materials


def get_materials_with_structure(size=100):
    """
    Fetch a batch of materials from the Materials Project, including both
    numeric properties (band_gap, density, etc.) AND the pymatgen Structure.
    
    Returns:
        List[dict] -- each dict contains:
            {
              "material_id": str,
              "band_gap": float or None,
              "density": float or None,
              "is_metal": bool or None,
              ... (any other summary fields we want) ...
              "structure": pymatgen.core.structure.Structure
            }
    """
    data_with_struct = []

    with MPRester(API_KEY) as mpr:
        # 1) Get summary documents for the first 'size' materials
        summary_docs = mpr.materials.summary.search(
            fields=["material_id", "band_gap", "density", "is_metal"],
            num_chunks=1,
            chunk_size=size
        )

        # 2) For each doc, fetch structure doc by material_id
        for doc in summary_docs:
            struc_doc_list = mpr.structures.search(
                material_ids=[doc.material_id],
                fields=["material_id", "structure"]
            )
            if not struc_doc_list:
                continue  # No structure found

            # For simplicity, grab the first structure doc
            structure = struc_doc_list[0].structure  # This is a pymatgen Structure

            # Combine numeric props + structure into a single dictionary
            data_with_struct.append({
                "material_id": doc.material_id,
                "band_gap": doc.band_gap,
                "density": doc.density,
                "is_metal": doc.is_metal,
                # add other fields from doc if needed
                "structure": structure
            })

    return data_with_struct


def save_data_local(filename, materials):
    """
    Save the numeric properties from a list of summary doc objects
    to a plain text file, comma-separated.

    NOTE: This does NOT save 'Structure' objects. If we want to save
    structures, we need a more robust format (e.g. JSON, CIF, etc.).
    """
    file = open("DownloadedCrystalProperties/" + filename + ".txt", "w")  # overwrites anything

    for material in materials:
        file.write(material.material_id + ", ")
        file.write(str(material.band_gap) + ", ")
        file.write(str(material.cbm) + ", ")
        file.write(str(material.density) + ", ")
        file.write(str(material.density_atomic) + ", ")
        file.write(str(material.dos_energy_down) + ", ")
        file.write(str(material.dos_energy_up) + ", ")
        file.write(str(material.e_electronic) + ", ")
        file.write(str(material.e_ij_max) + ", ")
        file.write(str(material.e_ionic) + ", ")
        file.write(str(material.e_total) + ", ")
        file.write(str(material.efermi) + ", ")
        file.write(str(material.energy_above_hull) + ", ")
        file.write(str(material.energy_per_atom) + ", ")
        file.write(str(material.equilibrium_reaction_energy_per_atom) + ", ")
        file.write(str(material.formation_energy_per_atom) + ", ")
        file.write(str(material.homogeneous_poisson) + ", ")
        file.write(str(material.n) + ", ")
        file.write(str(material.shape_factor) + ", ")
        file.write(str(material.surface_anisotropy) + ", ")
        file.write(str(material.total_magnetization) + ", ")
        file.write(str(material.total_magnetization_normalized_formula_units) + ", ")
        file.write(str(material.total_magnetization_normalized_vol) + ", ")
        file.write(str(material.uncorrected_energy_per_atom) + ", ")
        file.write(str(material.universal_anisotropy) + ", ")
        file.write(str(material.vbm) + ", ")
        file.write(str(material.volume) + ", ")
        file.write(str(material.weighted_surface_energy) + ", ")
        file.write(str(material.weighted_surface_energy_EV_PER_ANG2) + ", ")
        file.write(str(material.weighted_work_function) + ", ")
        file.write(str(material.is_metal) + ", ")
        file.write("\n")

    file.close()  # Make sure to call close() with parentheses


def ReadFileToComma(file):
    """
    Read characters until a comma is found.
    Return the accumulated text as either a float (if numeric) or string.
    Replaces 'None' with 'Null'.
    """
    currentword = ""
    while True:
        cl = file.read(1)
        if cl == ",":
            file.read(1)  # skip the space after comma
            if currentword == "None":
                currentword = "Null"
            break
        if cl == "\n":
            # might just ignore newlines
            continue
        if not cl:  # end-of-file
            break
        currentword += cl

    # Try to convert to float if possible
    if (currentword.replace('.', '', 1).replace('-', '', 1).isdigit()):
        currentword = float(currentword)
    return currentword


def load_data_local(filename, amount):
    """
    Load 'amount' lines from the text file and build a list of crystal objects
    with numeric fields. This is the old approach for reading from .txt.
    """
    file = open("DownloadedCrystalProperties/" + filename + ".txt", "r")
    materials = []

    for _ in range(amount):
        material_id = ReadFileToComma(file)
        band_gap = ReadFileToComma(file)
        cbm = ReadFileToComma(file)
        density = ReadFileToComma(file)
        density_atomic = ReadFileToComma(file)
        dos_energy_down = ReadFileToComma(file)
        dos_energy_up = ReadFileToComma(file)
        e_electronic = ReadFileToComma(file)
        e_ij_max = ReadFileToComma(file)
        e_ionic = ReadFileToComma(file)
        e_total = ReadFileToComma(file)
        efermi = ReadFileToComma(file)
        energy_above_hull = ReadFileToComma(file)
        energy_per_atom = ReadFileToComma(file)
        equilibrium_reaction_energy_per_atom = ReadFileToComma(file)
        formation_energy_per_atom = ReadFileToComma(file)
        homogeneous_poisson = ReadFileToComma(file)
        n = ReadFileToComma(file)
        shape_factor = ReadFileToComma(file)
        surface_anisotropy = ReadFileToComma(file)
        total_magnetization = ReadFileToComma(file)
        total_magnetization_normalized_formula_units = ReadFileToComma(file)
        total_magnetization_normalized_vol = ReadFileToComma(file)
        uncorrected_energy_per_atom = ReadFileToComma(file)
        universal_anisotropy = ReadFileToComma(file)
        vbm = ReadFileToComma(file)
        volume = ReadFileToComma(file)
        weighted_surface_energy = ReadFileToComma(file)
        weighted_surface_energy_EV_PER_ANG2 = ReadFileToComma(file)
        weighted_work_function = ReadFileToComma(file)
        is_metal = ReadFileToComma(file)

        materials.append(crystal(
            material_id, band_gap, cbm, density, density_atomic,
            dos_energy_down, dos_energy_up, e_electronic, e_ij_max, e_ionic,
            e_total, efermi, energy_above_hull, energy_per_atom,
            equilibrium_reaction_energy_per_atom, formation_energy_per_atom,
            homogeneous_poisson, n, shape_factor, surface_anisotropy,
            total_magnetization, total_magnetization_normalized_formula_units,
            total_magnetization_normalized_vol, uncorrected_energy_per_atom,
            universal_anisotropy, vbm, volume, weighted_surface_energy,
            weighted_surface_energy_EV_PER_ANG2, weighted_work_function,
            is_metal
        ))

    file.close()
    return materials