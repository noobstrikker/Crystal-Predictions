import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#For some reason I need to add parent directory to path

from crystalclass import crystal
from data_preprocessing import (
    extract_label, split_data, filter_unique_by_id,
    remove_all_null, check_against_example, filter_data
)

def create_test_crystal(**kwargs):
    """Helper to create test crystal objects with default null values"""
    defaults = {
        'material_id': "test_id",
        'band_gap': 0.0,
        'cbm': 0.0,
        'density': 0.0,
        'density_atomic': 0.0,
        'dos_energy_down': 0.0,
        'dos_energy_up': 0.0,
        'e_electronic': 0.0,
        'e_ij_max': 0.0,
        'e_ionic': 0.0,
        'e_total': 0.0,
        'efermi': 0.0,
        'energy_above_hull': 0.0,
        'energy_per_atom': 0.0,
        'equilibrium_reaction_energy_per_atom': 0.0,
        'formation_energy_per_atom': 0.0,
        'homogeneous_poisson': 0.0,
        'n': 0.0,
        'shape_factor': 0.0,
        'surface_anisotropy': 0.0,
        'total_magnetization': 0.0,
        'total_magnetization_normalized_formula_units': 0.0,
        'total_magnetization_normalized_vol': 0.0,
        'uncorrected_energy_per_atom': 0.0,
        'universal_anisotropy': 0.0,
        'vbm': 0.0,
        'volume': 0.0,
        'weighted_surface_energy': 0.0,
        'weighted_surface_energy_EV_PER_ANG2': 0.0,
        'weighted_work_function': 0.0,
        'is_metal': False
    }
    defaults.update(kwargs)
    return crystal(**defaults)

class TestDataPreprocessing(unittest.TestCase):
    def setUp(self):
        # Create test data using helper
        self.crystal1 = create_test_crystal(
            material_id="id1",
            band_gap=1.0,
            cbm=2.0,
            density="Null",
            is_metal=True,
            weighted_surface_energy=0.5
        )
        self.crystal2 = create_test_crystal(
            material_id="id2",
            band_gap="Null",
            cbm=1.5,
            density=3.0,
            is_metal=False,
            weighted_surface_energy="Null"
        )
        self.crystal3 = create_test_crystal(
            material_id="id1",  # Duplicate ID
            band_gap=2.0,
            cbm=1.5,
            density="Null",
            is_metal="Null"
        )
        self.example_crystal = create_test_crystal(
            material_id="example",
            band_gap=None,
            cbm=1.0,
            density="Null",
            is_metal=True
        )
        self.example_crystal2 = create_test_crystal(
            material_id="example2",
            band_gap=1.0,
            cbm=1.0,
            density="Null",
            is_metal=True
        )

        self.mock_structure = "mock_structure"

    def test_extract_label(self):
        data = [(self.crystal1, self.mock_structure), 
                    (self.crystal2, self.mock_structure), 
                    (self.crystal3, self.mock_structure)]
        labeled_data = extract_label(data)
        
        # Should have 2 items (crystal3 has is_metal="Null")
        self.assertEqual(len(labeled_data), 2)
        
        # Check is_metal was removed and labels are correct
        labels = [label for _, _ , label in labeled_data]
        self.assertIn(True, labels)
        self.assertIn(False, labels)
        
        for crystal_obj, _ , _ in labeled_data:
            self.assertFalse(hasattr(crystal_obj, 'is_metal'))

    def test_split_data(self):
        data = [(self.crystal1, self.mock_structure, True), 
               (self.crystal2, self.mock_structure, False), 
               (self.crystal3, self.mock_structure, True)]
        
        # Test default ratios (80/10/10)
        train, val, test = split_data(data.copy())
        self.assertEqual(len(train), 2)  # 80% of 3 = 2.4 → int(2.4) = 2
        self.assertEqual(len(val), 0)    # 10% of 3 = 0.3 → int(0.3) = 0
        self.assertEqual(len(test), 1)   # Remainder (3 - 2 - 0 = 1)
        
        # Test custom ratios (60/20/20)
        train, val, test = split_data(data.copy(), train_ratio=0.6, val_ratio=0.2)
        self.assertEqual(len(train), 1)  # 60% of 3 = 1.8 → int(1.8) = 1
        self.assertEqual(len(val), 0)    # 20% of 3 = 0.6 → int(0.6) = 0
        self.assertEqual(len(test), 2)   # Remainder (3 - 1 - 0 = 2)

    def test_filter_unique_by_id(self):
        data = [(self.crystal1, self.mock_structure, True), 
               (self.crystal2, self.mock_structure, False), 
               (self.crystal3, self.mock_structure, True)]
        filtered = filter_unique_by_id(data)
        
        # Should have 2 items (crystal1 and crystal3 have same ID)
        self.assertEqual(len(filtered), 2)
        ids = [crystal_obj.material_id for crystal_obj, _, _ in filtered]
        self.assertIn("id1", ids)
        self.assertIn("id2", ids)

    def test_remove_all_null(self):
        data = [(self.crystal1, self.mock_structure, True), 
               (self.crystal2, self.mock_structure, False)]
        filtered = remove_all_null(data)
        
        # Verify we still have 2 items
        self.assertEqual(len(filtered), 2)
        
        # Check crystal1 (had density="Null")
        crystal1 = filtered[0][0]
        self.assertFalse(hasattr(crystal1, 'density'), "density should be removed from crystal1")
        self.assertTrue(hasattr(crystal1, 'band_gap'), "band_gap should remain in crystal1")
        self.assertEqual(crystal1.band_gap, 1.0)
        
        # Check crystal2 (had band_gap="Null" and weighted_surface_energy="Null")
        crystal2 = filtered[1][0]
        self.assertFalse(hasattr(crystal2, 'band_gap'), "band_gap should be removed from crystal2")
        self.assertFalse(hasattr(crystal2, 'weighted_surface_energy'), "weighted_surface_energy should be removed from crystal2")
        self.assertTrue(hasattr(crystal2, 'density'), "density should remain in crystal2")
        self.assertEqual(crystal2.density, 3.0)

    def test_check_against_example(self):
        data = [(self.crystal1, self.mock_structure, True), 
               (self.crystal2, self.mock_structure, False)]
        
        # Test with example that has band_gap=None and density="Null"
        filtered = check_against_example(data, self.example_crystal)
        
        # Only crystal1 should pass (crystal2 has band_gap="Null" which is invalid when example expects band_gap)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0].material_id, "id1")
        
        # Check that density was removed (since example has density="Null")
        self.assertFalse(hasattr(filtered[0][0], 'density'))
        
        # Check with example that expects band_gap
        filtered2 = check_against_example(data, self.example_crystal2)
        # Only crystal1 should pass (has band_gap=1.0)
        self.assertEqual(len(filtered2), 1)
        self.assertEqual(filtered2[0][0].material_id, "id1")

    def test_filter_data_integration(self):
        data = [
            (self.crystal1, self.mock_structure, True),
            (self.crystal2, self.mock_structure, False),
            (self.crystal3, self.mock_structure, True),  # Duplicate ID
            (create_test_crystal(material_id="id3", band_gap="Null", cbm=1.0, is_metal=True), self.mock_structure, True)
        ]
        
        # Filter data using example_crystal as template
        filtered = filter_data(data, self.example_crystal)
        
        # Should have:
        # - Duplicates removed (crystal3 removed)
        # - Nulls removed
        # - Only crystals matching example structure (crystal1 and crystal3 would pass, but crystal3 is duplicate)
        # - crystal4 fails because it has band_gap="Null" when example expects band_gap
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0][0].material_id, "id1")
        self.assertTrue(filtered[0][1])  # Label should be True
        
        # Check that the filtering was applied correctly
        self.assertFalse(hasattr(filtered[0][0], 'density'))
        self.assertTrue(hasattr(filtered[0][0], 'band_gap'))
        self.assertTrue(hasattr(filtered[0][0], 'cbm'))
        
if __name__ == '__main__':
    unittest.main()