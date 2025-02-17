from mp_api.client import MPRester

#ik abuse min key :)
API_KEY = '3pY5xARryp2zkZkILLrWtz8zUqiSlnZZ'

with MPRester(API_KEY) as mpr:
    #Fetch
    materials = mpr.materials.summary.search(
        fields=["material_id", "formula_pretty", "band_gap"],
        num_chunks=1,
        #Antal crystals
        chunk_size=10
    )

    #Print
    for material in materials:
        print(f"ID: {material.material_id}, Formula: {material.formula_pretty}, Band Gap: {material.band_gap} eV")
