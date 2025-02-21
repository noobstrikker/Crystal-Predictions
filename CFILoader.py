#get class in this line
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
        currectword = ""
        for y in range(30):
            currectword = ""
        while(True):
            cl =file.read(1)
            if cl == ",":
                file.read(1)#skips space after ,
                break
            else:
                currectword += cl
        
    
    materials.append(crystalclass(material_id,band_gap, cbm, density, density_atomic, dos_energy_down, dos_energy_up, e_electronic,
                    e_ij_max, e_ionic, e_total, efermi, energy_above_hull, energy_per_atom, equilibrium_reaction_energy_per_atom,
                    formation_energy_per_atom, homogeneous_poisson, n, shape_factor, surface_anisotropy, total_magnetization,
                    total_magnetization_normalized_formula_units, total_magnetization_normalized_vol, uncorrected_energy_per_atom,
                    universal_anisotropy, vbm, volume, weighted_surface_energy, weighted_surface_energy_EV_PER_ANG2,weighted_work_function))
    file.close
    return materials