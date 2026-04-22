# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import torch
import traceback
from locate3d_data.locate3d_dataset import Locate3DDataset
from models.encoder_3djepa import Encoder3DJEPA
from models.locate_3d import downsample

# Set paths to data directories (update these paths as needed)
SCANNET_DATA_DIR = "[scannet_data_dir]"  # Replace with actual path
SCANNETPP_DATA_DIR = "[scannetpp_data_dir]"  # Replace with actual path
ARKITSCENES_DATA_DIR = "[arkitscenes_data_dir]"  # Replace with actual path

# Downsampling parameters
MAX_POINTS = 20000  # Reduced from 30000 to avoid OOM

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
model_3djepa.eval()  # Set to evaluation mode
model_3djepa = model_3djepa.cuda()  # Move to GPU

# Create output directory
output_dir = "outputs/3d_jepa_embeddings"
os.makedirs(output_dir, exist_ok=True)

# Track processed scenes to avoid duplicates
processed_scenes = set()
failed_scenes = []

# Process each scene (skip duplicates)
for idx in range(len(dataset)):
    try:
        data = dataset[idx]
        scene_name = data["scene_name"]
        
        # Skip if already processed
        if scene_name in processed_scenes:
            continue
        processed_scenes.add(scene_name)
        
        # Downsample pointcloud
        featurized_pc = downsample(data["featurized_sensor_pointcloud"], MAX_POINTS)
        
        try:
            # Check input data
            if isinstance(featurized_pc, dict):
                # If it's a dict, extract the point cloud tensor
                if "xyz" in featurized_pc:
                    pc_input = featurized_pc["xyz"]
                elif "points" in featurized_pc:
                    pc_input = featurized_pc["points"]
                else:
                    pc_input = list(featurized_pc.values())[0]
            else:
                pc_input = featurized_pc
            
            # Ensure input is on GPU
            if not pc_input.is_cuda:
                pc_input = pc_input.cuda()
            
            # Log input info for debugging
            print(f"\nProcessing scene {scene_name}:")
            print(f"  Input shape: {pc_input.shape}")
            print(f"  Input dtype: {pc_input.dtype}")
            print(f"  Input device: {pc_input.device}")
            
            # Run encoder with no_grad
            with torch.no_grad():
                output = model_3djepa(pc_input)
            
            # Save output
            output_path = os.path.join(output_dir, f"{scene_name}.pt")
            torch.save(output.cpu() if hasattr(output, 'cpu') else output, output_path)
            print(f"  ✓ Saved embedding to {output_path}")
            
        except RuntimeError as model_e:
            print(f"  ✗ CUDA/Runtime error for scene {scene_name}:")
            print(f"    {str(model_e)}")
            failed_scenes.append((scene_name, "CUDA/Runtime", str(model_e)))
            continue
        except Exception as model_e:
            print(f"  ✗ Model error for scene {scene_name}:")
            print(f"    {type(model_e).__name__}: {str(model_e)}")
            failed_scenes.append((scene_name, type(model_e).__name__, str(model_e)))
            traceback.print_exc()
            continue
        
    except FileNotFoundError as data_e:
        print(f"Skipped scene at index {idx} (missing cache file): {data_e}")
        continue
    except Exception as data_e:
        print(f"Skipped scene at index {idx}: {type(data_e).__name__}: {data_e}")
        continue

print("\n" + "="*60)
print("Processing complete.")
print(f"Total scenes processed: {len(processed_scenes)}")
if failed_scenes:
    print(f"Failed scenes: {len(failed_scenes)}")
    for scene, error_type, error_msg in failed_scenes:
        print(f"  - {scene}: {error_type}")
print("="*60)
