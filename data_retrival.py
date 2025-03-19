from crystalclass import crystal
from mp_api.client import MPRester

#ik abuse min key :)
API_KEY = 'JlJnqQljbO40Ooy3aGrORBR29rhD6eXO'


#size is the amount of crystals we wanted to importet, filename is what it is called if we want differnt file thingymabobs to have fun with
def get_materails_data(size):
    with MPRester(API_KEY) as mpr:
        #Fetch
        materials = mpr.materials.summary.search(
            fields=["material_id","band_gap", "cbm", "density", "density_atomic", "dos_energy_down", "dos_energy_up", "e_electronic",
                    "e_ij_max", "e_ionic", "e_total", "efermi", "energy_above_hull", "energy_per_atom", "equilibrium_reaction_energy_per_atom",
                    "formation_energy_per_atom", "homogeneous_poisson", "n", "shape_factor", "surface_anisotropy", "total_magnetization",
                    "total_magnetization_normalized_formula_units", "total_magnetization_normalized_vol", "uncorrected_energy_per_atom",
                    "universal_anisotropy", "vbm", "volume", "weighted_surface_energy", "weighted_surface_energy_EV_PER_ANG2",
                    "weighted_work_function","is_metal"],
            num_chunks=1,
            #Antal crystals
            chunk_size=size
        )
        return materials
def save_data_local(filename,materials):
    file = open("DownloadedCrystalProperties/"+filename+".txt","w") #overwrites anything
    
    #pural is list(alex is 100% gonna say um actually its an vector/array/or what ever it is), singluar is the one we are looking for
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
        file.write(str(material.weighted_work_function)+ ", ") 
        file.write(str(material.is_metal)+ ", ") 
        #hate me for "", " on the last one but it makes it eaiser to unravel of the otherside insted of doing mental gymnastics 

        file.write("\n")
            



        file.close


def ReadFileToComma(file):
    currentword = ""
    while(True):
            cl = file.read(1)
            if cl == ",":
                file.read(1)#skips space after ,
                if currentword == "None":
                     currentword = "Null" # or 0 but there is a diffrence is some cases of the logic
                break
            if cl == "\n":
                 1+1
            else:
                currentword += cl 
    if (currentword.replace('.','',1).replace('-','',1).isdigit()):
            currentword = float(currentword)
    return currentword
    



def load_data_local(filename, amount):
    file = open("DownloadedCrystalProperties/"+filename+".txt","r")
    materials = []
    for x in range(amount): # its from to 0 to "> amount"
        
        material_id = ReadFileToComma(file) #only string rest is floats, do some covertation
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
        
        materials.append(crystal(material_id,band_gap, cbm, density, density_atomic, dos_energy_down, dos_energy_up, e_electronic,
                    e_ij_max, e_ionic, e_total, efermi, energy_above_hull, energy_per_atom, equilibrium_reaction_energy_per_atom,
                    formation_energy_per_atom, homogeneous_poisson, n, shape_factor, surface_anisotropy, total_magnetization,
                    total_magnetization_normalized_formula_units, total_magnetization_normalized_vol, uncorrected_energy_per_atom,
                    universal_anisotropy, vbm, volume, weighted_surface_energy, weighted_surface_energy_EV_PER_ANG2,weighted_work_function,
                    is_metal))
    file.close
    return materials