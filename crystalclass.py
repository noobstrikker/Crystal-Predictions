class crystal:
    def __init__(self, material_id, structure, density, is_metal):
        self.material_id = material_id
        self.structure = structure  # pymatgen Structure
        self.density = density
        self.is_metal = is_metal

    @classmethod
    def from_dict(cls, data):
        """Create a crystal object from a dictionary containing Materials Project data"""
        from pymatgen.core.structure import Structure
        
        structure = Structure.from_dict(data["structure.json"])
        
        return cls(
            material_id=data["material_id"],
            structure=structure,
            density=data["density"],
            is_metal=data["is_metal"]
        )