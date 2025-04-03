import random
from crystalclass import crystal

def extract_label(data):
    labeled_data = []
    for crystal_obj, structure in data:
        if hasattr(crystal_obj, 'is_metal') and crystal_obj.is_metal != "Null":
            label = crystal_obj.is_metal
            del crystal_obj.is_metal
            labeled_data.append((crystal_obj, structure, label))
    return labeled_data

def split_data(data, train_ratio=0.8, val_ratio=0.1):
    random.shuffle(data)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def filter_unique_by_id(data):
    filtered_data = []
    seen_ids = set()

    for crystal_obj, structure, label in data:
        if hasattr(crystal_obj, 'material_id') and crystal_obj.material_id not in seen_ids:
            seen_ids.add(crystal_obj.material_id)
            filtered_data.append((crystal_obj, structure, label))
    return filtered_data

def remove_all_null(data):
    no_null_data = []
    for crystal_obj, structure, label in data:
        # Get list of attributes to delete first
        attrs_to_delete = [attr for attr, value in vars(crystal_obj).items() 
                          if value == "Null"]
        # Then delete them
        for attr in attrs_to_delete:
            delattr(crystal_obj, attr)
        no_null_data.append((crystal_obj, structure, label))
    return no_null_data

def check_against_example(data,example):

    filtered_data = []
    for crystal_obj, structure, label in data:
        valid = True
        for attr in vars(example):
            example_value = getattr(example, attr)
            # Case 1: If example has "Null", remove from material
            if example_value == "Null":
                if hasattr(crystal_obj, attr):
                    delattr(crystal_obj, attr)
            # Case 2: If example has non-"Null" value
            else:
                if not hasattr(crystal_obj, attr) or getattr(crystal_obj, attr) == "Null":
                    valid = False
                    break
        if valid:
            filtered_data.append((crystal_obj, structure, label))
    return filtered_data

def filter_data(data,example_crystal):
    filtered_data = filter_unique_by_id(data)
    
    filtered_data = remove_all_null(filtered_data)

    filtered_data = check_against_example(filtered_data,example_crystal)

    return filtered_data