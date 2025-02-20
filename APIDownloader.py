from mp_api.client import MPRester

#ik abuse min key :)
API_KEY = '3pY5xARryp2zkZkILLrWtz8zUqiSlnZZ'


#size is the amount of crystals we wanted to importet, filename is what it is called if we want differnt file thingymabobs to have fun with
def ApiDownloader(size,filename):
    with MPRester(API_KEY) as mpr:
        #Fetch
        materials = mpr.materials.summary.search(
            fields=["material_id","band_gap", "cbm", "density", "density_atomic", "dos_energy_down", "dos_energy_up", "e_electronic",
                    "e_ij_max", "e_ionic", "e_total", "efermi", "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                    "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor", "surface_anisotropy", "total_magnetization",
                    "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol", "uncorrected_energy_per_atom",
                    "universal_anisotropy", "vbm", "volume", "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                    "weighted_work_function"],
            num_chunks=1,
            #Antal crystals
            chunk_size=size
        )
        file = open("DownloadedCrystalProperties/"+filename+".txt","w") #overwrites anything
        
        #pural is list(alex is 100% gonna say um actually its an vector/array/or what ever it is), singluar is the one we are looking for
        for material in materials:
            file.write(material.material_id)
            file.write(material.band_gap)
            file.write(material.cbm)
            file.write(material.density)
            file.write(material.density_atomic)   
            file.write("\n")
            



        file.close
