import crystalclass
def ReadFileToComma(*file):
    currectword = ""
    while(True):
            cl =file.read(1)
            if cl == ",":
                file.read(1)#skips space after ,
                break
            else:
                currectword += cl
    



def LoadFromDisc(filename, amount):
    file = open("DownloadedCrystalProperties/"+filename+".txt","r")
    materials = []
    for x in range(amount): # its from to 0 to "> amount"
        
        material_id = ReadFileToComma(*file) #only string rest is floats, do some covertation
        band_gap = float(ReadFileToComma(*file))
        cbm = float(ReadFileToComma(*file))
        density = float(ReadFileToComma(*file))
        density_atomic = float(ReadFileToComma(*file))
        dos_energy_down = float(ReadFileToComma(*file))
        dos_energy_up = float(ReadFileToComma(*file))
        e_electronic = float(ReadFileToComma(*file))
        e_ij_max = float(ReadFileToComma(*file))
        e_ionic = float(ReadFileToComma(*file))
        e_total = float(ReadFileToComma(*file))
        efermi = float(ReadFileToComma(*file))
        energy_above_hull = float(ReadFileToComma(*file))
        energy_per_atom = float(ReadFileToComma(*file))
        equilibrium_reaction_energy_per_atom = float(ReadFileToComma(*file))
        formation_energy_per_atom = float(ReadFileToComma(*file))
        homogeneous_poisson = float(ReadFileToComma(*file))
        n = float(ReadFileToComma(*file))
        shape_factor = float(ReadFileToComma(*file))
        surface_anisotropy = float(ReadFileToComma(*file))
        total_magnetization = float(ReadFileToComma(*file))
        total_magnetization_normalized_formula_units = float(ReadFileToComma(*file))
        total_magnetization_normalized_vol = float(ReadFileToComma(*file))
        uncorrected_energy_per_atom = float(ReadFileToComma(*file))
        universal_anisotropy = float(ReadFileToComma(*file))
        vbm = float(ReadFileToComma(*file))
        volume = float(ReadFileToComma(*file))
        weighted_surface_energy = float(ReadFileToComma(*file))
        weighted_surface_energy_EV_PER_ANG2 = float(ReadFileToComma(*file))
        weighted_work_function = float(ReadFileToComma(*file))

    
    materials.append(crystalclass(material_id,band_gap, cbm, density, density_atomic, dos_energy_down, dos_energy_up, e_electronic,
                    e_ij_max, e_ionic, e_total, efermi, energy_above_hull, energy_per_atom, equilibrium_reaction_energy_per_atom,
                    formation_energy_per_atom, homogeneous_poisson, n, shape_factor, surface_anisotropy, total_magnetization,
                    total_magnetization_normalized_formula_units, total_magnetization_normalized_vol, uncorrected_energy_per_atom,
                    universal_anisotropy, vbm, volume, weighted_surface_energy, weighted_surface_energy_EV_PER_ANG2,weighted_work_function))
    file.close
    return materials