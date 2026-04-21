# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
from locate3d_data.locate3d_dataset import Locate3DDataset
from models.encoder_3djepa import Encoder3DJEPA
from models.locate_3d import downsample

# Set paths to data directories (update these paths as needed)
SCANNET_DATA_DIR = "[scannet_data_dir]"  # Replace with actual path
SCANNETPP_DATA_DIR = "[scannetpp_data_dir]"  # Replace with actual path
ARKITSCENES_DATA_DIR = "[arkitscenes_data_dir]"  # Replace with actual path

# Load ScanNet dataset
dataset = Locate3DDataset(
    annotations_fpath="locate3d_data/dataset/train_scannet.json",  # or val_scannet.json
    return_featurized_pointcloud=True,
    scannet_data_dir=SCANNET_DATA_DIR,
    scannetpp_data_dir=SCANNETPP_DATA_DIR,
    arkitscenes_data_dir=ARKITSCENES_DATA_DIR,
)

# Load 3D JEPA model
model_3djepa = Encoder3DJEPA.from_pretrained("facebook/3d-jepa")

# Create output directory
output_dir = "outputs/3d_jepa_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Track processed scenes to avoid duplicates
processed_scenes = set()

# Process each scene (skip duplicates)
for idx in range(len(dataset)):
    try:
        data = dataset[idx]
        scene_name = data["scene_name"]
        
        # Skip if already processed
        if scene_name in processed_scenes:
            continue
        processed_scenes.add(scene_name)
        
        # Downsample pointcloud (optional, adjust as needed)
        featurized_pc = downsample(data["featurized_sensor_pointcloud"], 30000)
        
        try:
            # Run encoder
            output = model_3djepa(featurized_pc)
            
            # Save output
            output_path = os.path.join(output_dir, f"{scene_name}.pt")
            torch.save(output, output_path)
            print(f"Saved embedding for scene {scene_name} to {output_path}")
            
        except Exception as model_e:
            print(f"Model error for scene {scene_name}: {model_e}")
            continue
        
    except Exception as data_e:
        # This is likely due to missing cache (not downloaded/preprocessed)
        print(f"Skipped scene at index {idx} (likely missing cache): {data_e}")
        continue

print("Processing complete.")
