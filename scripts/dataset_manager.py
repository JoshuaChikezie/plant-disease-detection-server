#!/usr/bin/env python3
import os
import json
from pathlib import Path

def count_images(crop, disease):
    disease_path = Path(f"data/datasets/{crop}/{disease}")
    if not disease_path.exists():
        return 0
    
    count = 0
    for file_path in disease_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            count += 1
    return count

def update_counts():
    print(" Updating image counts...")
    
    config_path = "data/datasets/dataset_config.json"
    if not os.path.exists(config_path):
        print(" Config file not found!")
        return
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    for crop_name, crop_data in config["datasets"].items():
        crop_path = Path(f"data/datasets/{crop_name}")
        if not crop_path.exists():
            print(f"  {crop_name}: Directory not found")
            continue
        
        total = 0
        for disease_name in crop_data["diseases"]:
            count = count_images(crop_name, disease_name)
            config["datasets"][crop_name]["diseases"][disease_name]["image_count"] = count
            total += count
            print(f"  {disease_name}: {count} images")
        
        config["datasets"][crop_name]["total_images"] = total
        print(f" {crop_name} total: {total} images")
    
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(" Image counts updated successfully!")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_manager.py [count|validate|summary]")
        sys.exit(1)
    
    action = sys.argv[1]
    if action == "count":
        update_counts()
    else:
        print(f"Action {action} not implemented yet")
        print(f"Action {action} not implemented yet")
