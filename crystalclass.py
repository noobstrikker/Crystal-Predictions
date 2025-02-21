class crystal:
    def __init__(self,material_id,band_gap, cbm, density, density_atomic, dos_energy_down, dos_energy_up, e_electronic,
                    e_ij_max, e_ionic, e_total, efermi, energy_above_hull, energy_per_atom, equilibrium_reaction_energy_per_atom,
                    formation_energy_per_atom, homogeneous_poisson, n, shape_factor, surface_anisotropy, total_magnetization,
                    total_magnetization_normalized_formula_units, total_magnetization_normalized_vol, uncorrected_energy_per_atom,
                    universal_anisotropy, vbm, volume, weighted_surface_energy, weighted_surface_energy_EV_PER_ANG2,weighted_work_function):
        self.material_id = material_id
        self.band_gap = band_gap
        self.cbm = cbm
        self.density = density
        self.density_atomic = density_atomic
        self.dos_energy_down = dos_energy_down
        self.dos_energy_up = dos_energy_up
        self.e_electronic = e_electronic
        self.e_ij_max = e_ij_max
        self.e_ionic = e_ionic
        self.e_total = e_total
        self.efermi = efermi
        self.energy_above_hull = energy_above_hull
        self.energy_per_aton = energy_per_atom
        self.equilibrium_reaction_energy_per_atom = equilibrium_reaction_energy_per_atom
        self.formation_energy_per_atom = formation_energy_per_atom
        self.homogeneous_poisson = homogeneous_poisson
        self.n = n
        self.shape_factor = shape_factor
        self.surface_anisotropy = surface_anisotropy
        self.total_magnetization = total_magnetization
        self.total_magnetization_normalized_formula_units = total_magnetization_normalized_formula_units
        self.total_magnetization_normalized_vol = total_magnetization_normalized_vol
        self.uncorrected_energy_per_atom = uncorrected_energy_per_atom
        self.universal_anisotropy = universal_anisotropy
        self.vbm = vbm
        self.volume = volume
        self.weighted_surface_energy = weighted_surface_energy
        self.weighted_surface_energy_EV_PER_ANG2 = weighted_surface_energy_EV_PER_ANG2
        self.weighted_work_function = weighted_work_function

