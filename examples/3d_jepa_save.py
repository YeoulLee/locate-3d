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
if torch.cuda.is_available():
    device = torch.device("cuda")
    model_3djepa = model_3djepa.to(device)  # Move to GPU
else:
    device = torch.device("cpu")
    print("CUDA not available, using CPU")

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
        
        # Ensure data is on the correct device
        if isinstance(data["featurized_sensor_pointcloud"], dict):
            data["featurized_sensor_pointcloud"] = {k: v.to(device) if hasattr(v, 'to') else v for k, v in data["featurized_sensor_pointcloud"].items()}
        elif hasattr(data["featurized_sensor_pointcloud"], 'to'):
            data["featurized_sensor_pointcloud"] = data["featurized_sensor_pointcloud"].to(device)

        # Skip if already processed
        if scene_name in processed_scenes:
            continue
        processed_scenes.add(scene_name)

        # Align tensor lengths: CLIP/DINO featurization in preprocessing can
        # produce point counts that differ by a few, which breaks torch.cat
        # inside the encoder. Truncate every tensor to the shared minimum.
        pc_dict = data["featurized_sensor_pointcloud"]
        if isinstance(pc_dict, dict):
            lengths = [v.shape[0] for v in pc_dict.values() if hasattr(v, "shape") and v.ndim > 0]
            if lengths and min(lengths) != max(lengths):
                min_len = min(lengths)
                print(f"  ! Aligning mismatched tensor lengths for {scene_name}: {lengths} -> {min_len}")
                pc_dict = {
                    k: (v[:min_len] if hasattr(v, "shape") and v.ndim > 0 else v)
                    for k, v in pc_dict.items()
                }
                data["featurized_sensor_pointcloud"] = pc_dict

        # Downsample pointcloud
        featurized_pc = downsample(data["featurized_sensor_pointcloud"], MAX_POINTS)
        
        try:
            # Check input data
            if isinstance(featurized_pc, dict):
                # Log input info for debugging
                print(f"\nProcessing scene {scene_name}:")
                print(f"  Dict keys: {list(featurized_pc.keys())}")
                if "points" in featurized_pc:
                    print(f"  Points shape: {featurized_pc['points'].shape}")
                    print(f"  Points device: {featurized_pc['points'].device}")
                if "features_clip" in featurized_pc:
                    print(f"  Features_clip shape: {featurized_pc['features_clip'].shape}")
                    print(f"  Features_clip device: {featurized_pc['features_clip'].device}")
            else:
                print(f"  Input type: {type(featurized_pc)}")
            
            # Run encoder with no_grad
            with torch.no_grad():
                model_output = model_3djepa(featurized_pc)
            
            output = {
                "features": model_output.cpu() if hasattr(model_output, 'cpu') else model_output,
                "points": featurized_pc["points"].cpu() if hasattr(featurized_pc["points"], 'cpu') else featurized_pc["points"]
            }
            
            # Save output
            output_path = os.path.join(output_dir, f"{scene_name}.pt")
            torch.save(output, output_path)
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
