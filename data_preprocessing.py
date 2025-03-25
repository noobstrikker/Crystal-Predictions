import random
from crystalclass import crystal

def extract_label(materials):
    labeled_data = []
    for material in materials:
        if material.is_metal != "Null":
            label = material.is_metal
            del material.is_metal
            labeled_data.append((material, label))
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

    for material, label in data:
        if material.material_id and material.material_id not in seen_ids:
            seen_ids.add(material.material_id)
            filtered_data.append((material, label))
    return filtered_data

def remove_all_null(data):
    no_null_data = []
    for material, label in data:
        attributes = list(material.__dict__.keys())

        # Delete attributes with value "Null"
        for attr in attributes:
            if getattr(material, attr) == "Null":
                delattr(material, attr)

        no_null_data.append((material, label))
    return no_null_data

def check_against_example(data, example):
    filtered_data = []
    for material, label in data:
        valid = True
        for attr in example.__dict__:
            example_value = getattr(example, attr)
            # Case 1: If example has "Null", remove from material
            if example_value == "Null":
                if hasattr(material, attr):
                    delattr(material, attr)
            # Case 2: If example has non-"Null" value
            else:
                if not hasattr(material, attr) or getattr(material, attr) == "Null":
                    valid = False
                    break
        if valid:
            filtered_data.append((material, label))
    return filtered_data

def filter_data(data,crystal_example):
    filtered_data = filter_unique_by_id(data)
    
    filtered_data = remove_all_null(filtered_data)

    filtered_data = check_against_example(filtered_data,crystal_example)

    return filtered_data
