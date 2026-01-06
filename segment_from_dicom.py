"""
Segmentation from DICOM Folders Script

This script:
1. Converts DICOM folders to NIfTI files (normalized to 1mm³ voxel spacing)
2. Runs TotalSegmentator on NIfTI files (keeps only left/right kidney outputs)
3. Runs UNET segmentation on NIfTI files
4. Compares UNET vs TotalSegmentator using 3D DICE scores (combined kidneys)

Author: Generated for kidney radiology segmentation comparison
Date: December 2025
"""

import numpy as np
import torch
import torch.nn.functional as F
from unet import UNet
from pathlib import Path
from tqdm import tqdm
import nibabel as nib
import subprocess
import argparse
import pandas as pd
import SimpleITK as sitk
import os
import shutil
from scipy.ndimage import zoom
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Suppress warnings to keep progress bars clean
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
# Suppress SimpleITK warnings
sitk.ProcessObject.SetGlobalWarningDisplay(False)
# Suppress NumPy warnings
np.seterr(all='ignore')
# Configure tqdm to write to stderr (doesn't interfere with progress bars)
import sys
tqdm.write = lambda s, file=sys.stderr, end='\n': file.write(s + end)


# ============================================================================
# Configuration
# ============================================================================

# File paths
METADATA_XLSX = '/mnt/vstor/Data7/bxf169/KidneyRadiology/kidney_cancer3_metadata.xlsx'
UNET_MODEL_PATH = 'ktseg_kits2023_best_model.pth'
OUTPUT_CSV = '/mnt/vstor/Data7/bxf169/KidneyRadiology/dice_scores_comparison.csv'

# Output directories (will be created under --output_dir)
# CONVERTED_NII_DIR and TS_OUTPUTS_DIR are now created as subdirectories of --output_dir

# TotalSegmentator paths
TS_ENV_PYTHON = '/mnt/vstor/CSE_BME_CCIPD/home/bxf169/total_segmentator_3_10_8/bin/python'
TS_EXECUTABLE = '/mnt/vstor/CSE_BME_CCIPD/home/bxf169/total_segmentator_3_10_8/bin/TotalSegmentator'

# Device configuration
DEVICE_ID = 0

# TotalSegmentator configuration
TS_FAST = True  # Use fast mode
TS_ORGANS = ['kidney_left', 'kidney_right']  # Only segment kidneys

# Voxel spacing (uniform 1mm isotropic)
VOXEL_SPACING = (1.0, 1.0, 1.0)

# Segmentation thresholds
UNET_KIDNEY_THRESHOLD = 0.34  # From reference code


# ============================================================================
# Model Loading
# ============================================================================

def load_unet_model(model_path, device):
    """
    Load pre-trained UNET model for kidney/tumor segmentation.
    
    Args:
        model_path: Path to model checkpoint file
        device: PyTorch device (cuda or cpu)
        
    Returns:
        Loaded UNET model in eval mode
    """
    print(f"Loading UNET model from: {model_path}")
    
    # Initialize model architecture
    model = UNet(
        n_classes=3,
        in_channels=1,
        padding=True,
        depth=6,
        wf=4,
        up_mode='upconv',
        batch_norm=True
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["model_dict"])
    
    # Set to evaluation mode
    model.eval()
    
    print("UNET model loaded successfully")
    return model


# ============================================================================
# DICOM to NIfTI Conversion
# ============================================================================

def resample_image_to_spacing(image, new_spacing, is_label=False):
    """
    Resample image to new voxel spacing.
    
    Args:
        image: SimpleITK Image
        new_spacing: Tuple of (x, y, z) spacing in mm
        is_label: If True, use nearest neighbor interpolation (default: False)
        
    Returns:
        Resampled SimpleITK Image
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # Calculate new size
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]
    
    # Create resample filter
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(0)
    
    # Set interpolator
    if is_label:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    
    return resample.Execute(image)


def convert_dicom_folder_to_nii(dicom_folder, output_nii_path, target_spacing=None, resample=True):
    """
    Convert DICOM folder to NIfTI file.
    
    Args:
        dicom_folder: Path to DICOM folder
        output_nii_path: Path to save output NIfTI file
        target_spacing: Target voxel spacing in mm (None = keep original, default: None)
        resample: Whether to resample to target_spacing (default: True)
                  If False, preserves original resolution
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read DICOM series
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(str(dicom_folder))
        
        if not series_ids:
            tqdm.write(f"    ERROR: No DICOM series found in folder: {dicom_folder}")
            return False
        
        # Use the first series ID
        series_files = reader.GetGDCMSeriesFileNames(str(dicom_folder), series_ids[0])
        reader.SetFileNames(series_files)
        
        # Execute reading
        image = reader.Execute()
        
        # Resample to target spacing if requested
        if resample and target_spacing is not None:
            resampled_image = resample_image_to_spacing(image, target_spacing, is_label=False)
            # Save as NIfTI
            sitk.WriteImage(resampled_image, str(output_nii_path))
        else:
            # Save original resolution (for TotalSegmentator - it will resample internally)
            sitk.WriteImage(image, str(output_nii_path))
        
        return True
        
    except Exception as e:
        tqdm.write(f"    ERROR: Failed to convert DICOM folder {dicom_folder}: {e}")
        return False


def _process_single_dicom_folder(args):
    """
    Worker function to process a single DICOM folder conversion.
    
    Args:
        args: Tuple of (row, base_folder, output_dir, overwrite, existing_mapping, start_id_lock, start_id_counter)
        
    Returns:
        Dictionary with conversion result
    """
    (idx, row), base_folder, output_dir, overwrite, existing_mapping, start_id_lock, start_id_counter = args
    
    # Get folder path from metadata (original path)
    original_folder_path = str(row['Folder'])
    
    # Replace base path
    folder_path_normalized = original_folder_path.replace('\\', '/')
    
    # Try to find the pattern to replace
    if 'TCGA-KIRC' in folder_path_normalized:
        parts = folder_path_normalized.split('TCGA-KIRC', 1)
        if len(parts) == 2:
            server_path = os.path.join(base_folder, parts[1].lstrip('/'))
        else:
            server_path = original_folder_path
    else:
        base_patterns = [
            'E:/Brennan/TCGA_KIRC/CT/manifest-1671033260647/TCGA-KIRC',
            'E:\\Brennan\\TCGA_KIRC\\CT\\manifest-1671033260647\\TCGA-KIRC',
        ]
        server_path = original_folder_path
        for pattern in base_patterns:
            if pattern in original_folder_path:
                server_path = original_folder_path.replace(pattern, base_folder)
                break
    
    # Normalize path
    server_path = os.path.normpath(server_path)
    dicom_folder = Path(server_path)
    
    # Check if DICOM folder exists
    if not dicom_folder.exists():
        # Generate numerical ID
        with start_id_lock:
            numerical_id = f"{start_id_counter[0]:04d}"
            start_id_counter[0] += 1
        output_nii_path = output_dir / f"{numerical_id}.nii.gz"
        return {
            'folder_path': str(dicom_folder),
            'original_folder_path': original_folder_path,
            'nii_path': str(output_nii_path),
            'status': 'failed',
            'numerical_id': numerical_id,
            'error': 'Folder not found'
        }
    
    # Check if this folder is already in mapping
    numerical_id = None
    if existing_mapping is not None and not overwrite:
        matching_row = existing_mapping[existing_mapping['original_folder_path'] == original_folder_path]
        if len(matching_row) > 0:
            numerical_id_raw = matching_row.iloc[0]['numerical_id']
            # Ensure numerical_id has leading zeros format
            try:
                numerical_id = f"{int(numerical_id_raw):04d}"
            except (ValueError, TypeError):
                numerical_id = str(numerical_id_raw)
            output_nii_path = output_dir / f"{numerical_id}.nii.gz"
            
            if output_nii_path.exists():
                return {
                    'folder_path': str(dicom_folder),
                    'original_folder_path': original_folder_path,
                    'nii_path': str(output_nii_path),
                    'status': 'skipped',
                    'numerical_id': numerical_id
                }
    
    # Generate new numerical ID if not found
    if numerical_id is None:
        with start_id_lock:
            numerical_id = f"{start_id_counter[0]:04d}"
            start_id_counter[0] += 1
    
    output_nii_path = output_dir / f"{numerical_id}.nii.gz"
    
    # Check if already exists
    if output_nii_path.exists() and not overwrite:
        return {
            'folder_path': str(dicom_folder),
            'original_folder_path': original_folder_path,
            'nii_path': str(output_nii_path),
            'status': 'skipped',
            'numerical_id': numerical_id
        }
    
    # Convert DICOM to NIfTI at original resolution
    # TotalSegmentator will resample internally (1.5mm standard or 3mm fast)
    # UNET will need 1mm³, so we'll resample for UNET separately
    success = convert_dicom_folder_to_nii(dicom_folder, output_nii_path, target_spacing=None, resample=False)
    
    if success:
        return {
            'folder_path': str(dicom_folder),
            'original_folder_path': original_folder_path,
            'nii_path': str(output_nii_path),
            'status': 'converted',
            'numerical_id': numerical_id
        }
    else:
        return {
            'folder_path': str(dicom_folder),
            'original_folder_path': original_folder_path,
            'nii_path': str(output_nii_path),
            'status': 'failed',
            'numerical_id': numerical_id,
            'error': 'Conversion failed'
        }


def convert_dicom_folders(metadata_df, base_folder, output_dir, overwrite=False, num_workers=1):
    """
    Convert DICOM folders listed in metadata to NIfTI files with numerical IDs.
    
    Args:
        metadata_df: DataFrame with 'Folder' column containing paths
        base_folder: Base folder path to replace in metadata paths
        output_dir: Directory to save converted NIfTI files
        overwrite: If True, overwrite existing files (default: False)
        num_workers: Number of worker threads for parallel processing (default: 1)
        
    Returns:
        Tuple of (conversion_results, mapping_df) where:
        - conversion_results: List of dictionaries with conversion results
        - mapping_df: DataFrame mapping numerical_id to original_folder_path
    """
    print("\n" + "="*70)
    print("STEP 1: Converting DICOM folders to NIfTI files")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for existing mapping file
    mapping_file = output_dir / 'folder_mapping.csv'
    existing_mapping = None
    if mapping_file.exists() and not overwrite:
        try:
            existing_mapping = pd.read_csv(mapping_file)
            print(f"  Found existing mapping file with {len(existing_mapping)} entries")
        except Exception as e:
            print(f"  WARNING: Could not read existing mapping file: {e}")
    
    conversion_results = []
    mapping_data = []
    skipped = 0
    converted = 0
    failed = 0
    
    print(f"\nProcessing {len(metadata_df)} DICOM folders...")
    print(f"Base folder replacement: {base_folder}")
    print(f"Output directory: {output_dir}")
    print(f"Number of workers: {num_workers}")
    
    # Determine starting ID number
    if existing_mapping is not None and not overwrite:
        # Find the highest existing ID
        existing_ids = existing_mapping['numerical_id'].astype(str).str.extract(r'(\d+)')[0].astype(int)
        start_id = existing_ids.max() + 1 if len(existing_ids) > 0 else 1
    else:
        start_id = 1
    
    # Thread-safe counter and lock
    start_id_lock = Lock()
    start_id_counter = [start_id]  # Use list to allow modification in nested function
    
    # Prepare arguments for worker function
    worker_args = [
        ((idx, row), base_folder, output_dir, overwrite, existing_mapping, start_id_lock, start_id_counter)
        for idx, row in metadata_df.iterrows()
    ]
    
    # Process with ThreadPoolExecutor
    conversion_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_args = {executor.submit(_process_single_dicom_folder, args): args for args in worker_args}
        
        # Create progress bar
        pbar = tqdm(total=len(metadata_df), desc="Converting DICOM to NIfTI")
        
        # Track counts (thread-safe updates)
        converted = 0
        skipped = 0
        failed = 0
        
        # Process completed tasks
        for future in as_completed(future_to_args):
            try:
                result = future.result()
                conversion_results.append(result)
                
                # Update counts
                if result['status'] == 'converted':
                    converted += 1
                elif result['status'] == 'skipped':
                    skipped += 1
                elif result['status'] == 'failed':
                    failed += 1
                    # Log warnings for failed cases
                    if result.get('error') == 'Folder not found':
                        tqdm.write(f"    WARNING: DICOM folder does not exist: {result['folder_path']}")
                        tqdm.write(f"      Original path in metadata: {result['original_folder_path']}")
                
                # Update progress bar
                pbar.set_postfix({'Converted': converted, 'Skipped': skipped, 'Failed': failed})
                pbar.update(1)
                
            except Exception as e:
                failed += 1
                tqdm.write(f"    ERROR: Exception in worker thread: {e}")
                pbar.set_postfix({'Converted': converted, 'Skipped': skipped, 'Failed': failed})
                pbar.update(1)
        
        pbar.close()
    
    # Build mapping data from conversion results
    mapping_data = []
    for result in conversion_results:
        if result['status'] in ['converted', 'skipped']:
            mapping_data.append({
                'numerical_id': result['numerical_id'],
                'original_folder_path': result['original_folder_path'],
                'server_folder_path': result['folder_path']
            })
    
    # Create or update mapping DataFrame
    if mapping_data:
        new_mapping_df = pd.DataFrame(mapping_data)
        if existing_mapping is not None and not overwrite:
            # Combine with existing mapping
            mapping_df = pd.concat([existing_mapping, new_mapping_df], ignore_index=True)
        else:
            mapping_df = new_mapping_df
        
        # Save mapping file
        mapping_df.to_csv(mapping_file, index=False)
        print(f"\n  Saved folder mapping to: {mapping_file}")
    
    print(f"\nConversion Summary:")
    print(f"  Converted: {converted}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(metadata_df)}")
    
    # Create mapping DataFrame for return (include all entries)
    all_mapping_data = []
    for result in conversion_results:
        if 'original_folder_path' in result:
            all_mapping_data.append({
                'numerical_id': result['numerical_id'],
                'original_folder_path': result['original_folder_path'],
                'server_folder_path': result['folder_path']
            })
    
    if existing_mapping is not None and not overwrite:
        # Include existing entries not in current batch
        for _, row in existing_mapping.iterrows():
            if row['original_folder_path'] not in [r.get('original_folder_path', '') for r in all_mapping_data]:
                all_mapping_data.append({
                    'numerical_id': row['numerical_id'],
                    'original_folder_path': row['original_folder_path'],
                    'server_folder_path': row.get('server_folder_path', '')
                })
    
    mapping_df = pd.DataFrame(all_mapping_data) if all_mapping_data else pd.DataFrame(columns=['numerical_id', 'original_folder_path', 'server_folder_path'])
    
    return conversion_results, mapping_df


# ============================================================================
# TotalSegmentator Functions
# ============================================================================

def run_totalsegmentator(input_path, output_dir, fast=True, device='gpu', gpu_id=None):
    """
    Run TotalSegmentator on a NIfTI file via subprocess using CLI.
    
    Args:
        input_path: Path to input NIfTI file
        output_dir: Directory to save segmentation outputs
        fast: Use fast mode (default: True)
        device: Device to use ('gpu' or 'cpu', default: 'gpu') - Note: TotalSegmentator auto-detects GPU
        gpu_id: Specific GPU ID to use (None = use default, default: None)
        
    Returns:
        True if successful, False otherwise
    """
    # Construct command using CLI executable (matches original working script)
    cmd = [
        TS_EXECUTABLE,
        '-i', str(input_path),
        '-o', str(output_dir)
    ]
    
    if fast:
        cmd.append('--fast')
    
    # Prepare environment (TotalSegmentator auto-detects GPU, but we can set CUDA_VISIBLE_DEVICES)
    env = os.environ.copy()
    if gpu_id is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    elif device == 'cpu':
        # Force CPU by hiding all GPUs
        env['CUDA_VISIBLE_DEVICES'] = ''
    
    try:
        # Run TotalSegmentator CLI (matches original working script exactly)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
            env=env
        )
        
        if result.returncode != 0:
            tqdm.write(f"    ERROR: TotalSegmentator failed with return code {result.returncode}")
            if result.stderr:
                tqdm.write(f"    STDERR: {result.stderr[:500]}")
            if result.stdout:
                tqdm.write(f"    STDOUT: {result.stdout[:500]}")
            return False
        
        return True
        
    except subprocess.TimeoutExpired:
        tqdm.write(f"    ERROR: TotalSegmentator timed out after 300 seconds")
        return False
    except Exception as e:
        tqdm.write(f"    ERROR: Failed to run TotalSegmentator: {e}")
        return False


def clean_totalsegmentator_outputs(output_dir, keep_organs=['kidney_left', 'kidney_right']):
    """
    Remove all TotalSegmentator outputs except specified organs.
    
    Args:
        output_dir: Directory containing TotalSegmentator outputs
        keep_organs: List of organ names to keep (default: ['kidney_left', 'kidney_right'])
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return
    
    # Get all files in output directory
    for file_path in output_dir.iterdir():
        if file_path.is_file():
            # Check if it's a NIfTI file
            if file_path.suffix in ['.nii', '.gz']:
                # Get base name without extension
                base_name = file_path.stem
                if file_path.suffix == '.gz':
                    base_name = Path(base_name).stem
                
                # Delete if not in keep list
                if base_name not in keep_organs:
                    try:
                        file_path.unlink()
                    except Exception as e:
                        tqdm.write(f"    WARNING: Failed to delete {file_path}: {e}")


def get_available_gpus():
    """
    Get list of available GPU IDs.
    
    Returns:
        List of GPU IDs (integers) or empty list if no GPUs available
    """
    try:
        import torch
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except:
        pass
    return []


def _process_single_ts(args):
    """
    Worker function for multi-GPU TotalSegmentator processing.
    
    Args:
        args: Tuple of (result, output_base_dir, overwrite, gpu_id, device)
        
    Returns:
        Tuple of (numerical_id, result_dict)
    """
    result, output_base_dir, overwrite, gpu_id, device = args
    
    numerical_id = result['numerical_id']
    # Ensure numerical_id is a string with leading zeros (CSV might load it as int)
    # Convert to int first to handle both string and int, then format with leading zeros
    try:
        numerical_id = f"{int(numerical_id):04d}"
    except (ValueError, TypeError):
        numerical_id = str(numerical_id)
    nii_path = Path(result['nii_path'])
    
    if not nii_path.exists():
        return (numerical_id, {'status': 'failed', 'error': 'NIfTI file not found'})
    
    output_dir = output_base_dir / numerical_id
    kidney_left_path = output_dir / 'kidney_left.nii.gz'
    kidney_right_path = output_dir / 'kidney_right.nii.gz'
    
    # Check for existing outputs (before running TS)
    if kidney_left_path.exists() and kidney_right_path.exists():
        if not overwrite:
            return (numerical_id, {
                'kidney_left': str(kidney_left_path),
                'kidney_right': str(kidney_right_path),
                'status': 'skipped'
            })
        else:
            # Overwrite mode: remove existing files
            try:
                if kidney_left_path.exists():
                    kidney_left_path.unlink()
                if kidney_right_path.exists():
                    kidney_right_path.unlink()
            except Exception as e:
                pass  # Continue even if deletion fails
    
    output_dir.mkdir(parents=True, exist_ok=True)
    success = run_totalsegmentator(nii_path, output_dir, fast=TS_FAST, device=device, gpu_id=gpu_id)
    
    if success:
        clean_totalsegmentator_outputs(output_dir, keep_organs=TS_ORGANS)
        if kidney_left_path.exists() and kidney_right_path.exists():
            return (numerical_id, {
                'kidney_left': str(kidney_left_path),
                'kidney_right': str(kidney_right_path),
                'status': 'processed'
            })
        else:
            return (numerical_id, {'status': 'failed', 'error': 'Kidney outputs not found'})
    else:
        return (numerical_id, {'status': 'failed', 'error': 'TotalSegmentator execution failed'})


def _run_totalsegmentator_multigpu(valid_results, output_base_dir, overwrite, num_gpus, available_gpus):
    """
    Run TotalSegmentator using multiple GPUs in parallel.
    
    Args:
        valid_results: List of valid conversion results
        output_base_dir: Base directory for outputs
        overwrite: Whether to overwrite existing outputs
        num_gpus: Number of GPUs to use
        available_gpus: List of available GPU IDs
        
    Returns:
        Dictionary mapping numerical_id to TS output paths
    """
    from multiprocessing import Pool
    
    # Limit to available GPUs
    num_gpus = min(num_gpus, len(available_gpus))
    gpus_to_use = available_gpus[:num_gpus]
    
    ts_results = {}
    skipped = 0
    processed = 0
    failed = 0
    
    # Prepare arguments for workers
    worker_args = []
    for i, result in enumerate(valid_results):
        gpu_id = gpus_to_use[i % num_gpus]  # Round-robin GPU assignment
        worker_args.append((result, output_base_dir, overwrite, gpu_id, 'gpu'))
    
    # Process with multiprocessing
    pbar = tqdm(total=len(valid_results), desc="Running TotalSegmentator (Multi-GPU)")
    
    with Pool(processes=num_gpus) as pool:
        for numerical_id, result_dict in pool.imap(_process_single_ts, worker_args):
            ts_results[numerical_id] = result_dict
            
            if result_dict['status'] == 'processed':
                processed += 1
            elif result_dict['status'] == 'skipped':
                skipped += 1
            else:
                failed += 1
            
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
            pbar.update(1)
    
    pbar.close()
    
    print(f"\nTotalSegmentator Summary (Multi-GPU):")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(valid_results)}")
    
    return ts_results


def run_totalsegmentator_on_nii_files(conversion_results, output_base_dir, overwrite=False, 
                                      num_gpus=None, use_gpu=True):
    """
    Run TotalSegmentator on all converted NIfTI files.
    
    Args:
        conversion_results: List of dictionaries from convert_dicom_folders
        output_base_dir: Base directory for TotalSegmentator outputs
        overwrite: If True, overwrite existing outputs (default: False)
        num_gpus: Number of GPUs to use for parallel processing (None = sequential, default: None)
        use_gpu: Whether to use GPU (default: True)
        
    Returns:
        Dictionary mapping numerical_id to TS output paths
    """
    print("\n" + "="*70)
    print("STEP 2: Running TotalSegmentator on NIfTI files")
    print("="*70)
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    ts_results = {}
    skipped = 0
    processed = 0
    failed = 0
    
    # Filter to only valid results for progress bar
    valid_results = [r for r in conversion_results if r['status'] in ['converted', 'skipped']]
    print(f"\nProcessing {len(valid_results)} NIfTI files with TotalSegmentator...")
    
    # Determine GPU usage
    available_gpus = get_available_gpus() if use_gpu else []
    device = 'gpu' if use_gpu and available_gpus else 'cpu'
    
    # Default behavior: sequential processing with 1 GPU (if available)
    # num_gpus=None or num_gpus=1 means sequential
    # num_gpus>1 means multi-GPU parallel processing
    if num_gpus is not None and num_gpus > 1 and available_gpus:
        # Multi-GPU processing (only if explicitly requested and multiple GPUs available)
        if num_gpus > len(available_gpus):
            tqdm.write(f"    WARNING: Requested {num_gpus} GPUs but only {len(available_gpus)} available. Using {len(available_gpus)} GPUs.")
            num_gpus = len(available_gpus)
        print(f"  Using {num_gpus} GPUs for parallel processing")
        print(f"  Available GPUs: {available_gpus}")
        return _run_totalsegmentator_multigpu(valid_results, output_base_dir, overwrite, num_gpus, available_gpus)
    else:
        # Sequential processing (default - optimal for single GPU)
        if device == 'gpu' and available_gpus:
            print(f"  Using GPU: {available_gpus[0]} (sequential processing - default)")
        else:
            if use_gpu and not available_gpus:
                print(f"  WARNING: GPU requested but not available. Using CPU (sequential processing)")
            else:
                print(f"  Using CPU (sequential processing)")
    
    pbar = tqdm(valid_results, desc="Running TotalSegmentator")
    for result in pbar:
        if result['status'] != 'converted' and result['status'] != 'skipped':
            continue  # Skip failed conversions
        
        numerical_id = result['numerical_id']
        # Ensure numerical_id is a string with leading zeros (CSV might load it as int)
        # Convert to int first to handle both string and int, then format with leading zeros
        try:
            numerical_id = f"{int(numerical_id):04d}"
        except (ValueError, TypeError):
            numerical_id = str(numerical_id)
        nii_path = Path(result['nii_path'])
        
        if not nii_path.exists():
            tqdm.write(f"    WARNING: NIfTI file does not exist: {nii_path}")
            failed += 1
            ts_results[numerical_id] = {
                'status': 'failed',
                'error': 'NIfTI file not found'
            }
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
            continue
        
        # Create output directory for this case
        output_dir = output_base_dir / numerical_id
        
        # Check if outputs already exist
        kidney_left_path = output_dir / 'kidney_left.nii.gz'
        kidney_right_path = output_dir / 'kidney_right.nii.gz'
        
        # Check for existing outputs (before running TS)
        if kidney_left_path.exists() and kidney_right_path.exists():
            if not overwrite:
                skipped += 1
                ts_results[numerical_id] = {
                    'kidney_left': str(kidney_left_path),
                    'kidney_right': str(kidney_right_path),
                    'status': 'skipped'
                }
                pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
                continue
            else:
                # Overwrite mode: remove existing files
                try:
                    if kidney_left_path.exists():
                        kidney_left_path.unlink()
                    if kidney_right_path.exists():
                        kidney_right_path.unlink()
                except Exception as e:
                    tqdm.write(f"    WARNING: Failed to remove existing outputs for {numerical_id}: {e}")
        
        # Run TotalSegmentator
        output_dir.mkdir(parents=True, exist_ok=True)
        gpu_id = available_gpus[0] if device == 'gpu' and available_gpus else None
        success = run_totalsegmentator(nii_path, output_dir, fast=TS_FAST, device=device, gpu_id=gpu_id)
        
        if success:
            # Clean outputs (keep only kidneys)
            clean_totalsegmentator_outputs(output_dir, keep_organs=TS_ORGANS)
            
            # Verify kidney outputs exist
            if kidney_left_path.exists() and kidney_right_path.exists():
                processed += 1
                ts_results[numerical_id] = {
                    'kidney_left': str(kidney_left_path),
                    'kidney_right': str(kidney_right_path),
                    'status': 'processed'
                }
                pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
            else:
                tqdm.write(f"    WARNING: Kidney outputs not found for {numerical_id}")
                failed += 1
                ts_results[numerical_id] = {
                    'status': 'failed',
                    'error': 'Kidney outputs not found'
                }
                pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
        else:
            failed += 1
            ts_results[numerical_id] = {
                'status': 'failed',
                'error': 'TotalSegmentator execution failed'
            }
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
    
    print(f"\nTotalSegmentator Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len([r for r in conversion_results if r['status'] in ['converted', 'skipped']])}")
    
    return ts_results


# ============================================================================
# UNET Segmentation Functions
# ============================================================================

def save_volume_as_nifti(volume, output_path, spacing=VOXEL_SPACING):
    """
    Save a numpy volume as NIfTI file.
    
    Args:
        volume: 3D numpy array (D, H, W) from segment_volume_unet
        output_path: Path to save NIfTI file
        spacing: Voxel spacing in mm (default: 1mm isotropic)
    """
    # segment_volume_unet returns volumes in (D, H, W) format
    # NIfTI expects (H, W, D) format, so transpose
    if len(volume.shape) == 3:
        volume = np.transpose(volume, (1, 2, 0))
    
    # Create affine matrix with proper spacing
    affine = np.eye(4)
    affine[0, 0] = spacing[0]
    affine[1, 1] = spacing[1]
    affine[2, 2] = spacing[2]
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume.astype(np.float32), affine)
    
    # Save to file
    nib.save(nifti_img, output_path)


def load_nifti_as_volume(nifti_path, target_spacing=None):
    """
    Load a NIfTI file as numpy volume, optionally resampling to target spacing.
    
    Args:
        nifti_path: Path to NIfTI file
        target_spacing: Target voxel spacing in mm (None = keep original, default: None)
        
    Returns:
        3D numpy array (D, H, W) - depth, height, width
    """
    nifti_img = nib.load(nifti_path)
    
    # Get original spacing from affine matrix
    affine = nifti_img.affine
    original_spacing = np.abs([affine[0, 0], affine[1, 1], affine[2, 2]])
    
    volume = nifti_img.get_fdata()
    
    # Resample if target spacing is specified and different from original
    if target_spacing is not None:
        # Check if resampling is needed
        spacing_diff = np.abs(np.array(target_spacing) - original_spacing)
        if np.any(spacing_diff > 0.01):  # More than 0.01mm difference
            # Load as SimpleITK for resampling
            sitk_image = sitk.ReadImage(str(nifti_path))
            resampled_image = resample_image_to_spacing(sitk_image, target_spacing, is_label=False)
            volume = sitk.GetArrayFromImage(resampled_image)
            # SimpleITK returns (D, H, W) already, so no transpose needed
            return volume
    
    # Ensure volume is in correct orientation (D, H, W) for UNET
    # NIfTI files loaded with nibabel are typically (H, W, D), so we transpose to (D, H, W)
    # This matches the format expected by UNET (slice-by-slice processing along depth axis)
    if len(volume.shape) == 3:
        volume = np.transpose(volume, (2, 0, 1))
    
    return volume


def segment_volume_unet(img_stack, model, device):
    """
    Segment a 3D volume using UNET model (slice-by-slice).
    
    Args:
        img_stack: 3D numpy array (D, H, W) containing CT volume
        model: Pre-trained UNET model
        device: PyTorch device
        
    Returns:
        kidney_stack: 3D numpy array with kidney segmentation probabilities
        tumor_stack: 3D numpy array with tumor segmentation probabilities
    """
    kidney_stack = np.zeros(np.shape(img_stack))
    tumor_stack = np.zeros(np.shape(img_stack))
    
    # Get expected output size from input
    expected_h, expected_w = img_stack.shape[1], img_stack.shape[2]
    
    # Process each slice
    for i in range(np.shape(img_stack)[0]):
        # Prepare input tensor: add batch and channel dimensions
        input_tensor = torch.Tensor(img_stack[i, ::].copy())[None, None, ::].to(device)
        
        # Run through model with softmax
        with torch.no_grad():
            output = torch.softmax(model(input_tensor), dim=1)[0, ::].permute(1, 2, 0).detach().cpu().numpy()
        
        # Get actual output size
        output_h, output_w = output.shape[0], output.shape[1]
        
        # Handle size mismatch: resize output to match input if needed
        if output_h != expected_h or output_w != expected_w:
            # Use bilinear interpolation to resize output to match input dimensions
            # Convert to torch tensor for interpolation
            output_tensor = torch.from_numpy(output).permute(2, 0, 1).unsqueeze(0).float()
            # Resize using bilinear interpolation
            output_resized = F.interpolate(
                output_tensor, 
                size=(expected_h, expected_w), 
                mode='bilinear', 
                align_corners=False
            )
            # Convert back to numpy
            output = output_resized.squeeze(0).permute(1, 2, 0).numpy()
        
        # Extract kidney and tumor probability maps
        kidney_stack[i, ::] = output[:, :, 0]
        tumor_stack[i, ::] = output[:, :, 1]
    
    return kidney_stack, tumor_stack


def run_unet_on_nii_files(conversion_results, unet_model, device, output_base_dir, overwrite=False):
    """
    Run UNET segmentation on all converted NIfTI files and save outputs to disk.
    
    Args:
        conversion_results: List of dictionaries from convert_dicom_folders
        unet_model: Loaded UNET model
        device: PyTorch device
        output_base_dir: Base directory for UNET outputs
        overwrite: If True, overwrite existing outputs (default: False)
        
    Returns:
        Dictionary mapping numerical_id to UNET output file paths
    """
    print("\n" + "="*70)
    print("STEP 3: Running UNET segmentation on NIfTI files")
    print("="*70)
    
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    unet_results = {}
    
    # Filter to only valid results for progress bar
    valid_results = [r for r in conversion_results if r['status'] in ['converted', 'skipped']]
    print(f"\nProcessing {len(valid_results)} NIfTI files with UNET...")
    print(f"UNET outputs directory: {output_base_dir}")
    
    processed = 0
    skipped = 0
    failed = 0
    
    pbar = tqdm(valid_results, desc="Running UNET")
    for result in pbar:
        
        numerical_id = result['numerical_id']
        # Ensure numerical_id has leading zeros format (CSV might load it as int)
        try:
            numerical_id = f"{int(numerical_id):04d}"
        except (ValueError, TypeError):
            numerical_id = str(numerical_id)
        nii_path = Path(result['nii_path'])
        
        if not nii_path.exists():
            failed += 1
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
            continue
        
        # Check if UNET outputs already exist
        kidney_prob_path = output_base_dir / f"{numerical_id}_kidney_prob.nii.gz"
        tumor_prob_path = output_base_dir / f"{numerical_id}_tumor_prob.nii.gz"
        
        if kidney_prob_path.exists() and tumor_prob_path.exists() and not overwrite:
            skipped += 1
            unet_results[numerical_id] = {
                'kidney_prob': str(kidney_prob_path),
                'tumor_prob': str(tumor_prob_path),
                'status': 'skipped'
            }
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
            continue
        
        try:
            # Load NIfTI volume and resample to 1mm³ for UNET
            # Note: NIfTI is at original resolution (for TS), but UNET needs 1mm³
            volume = load_nifti_as_volume(nii_path, target_spacing=VOXEL_SPACING)
            
            # Ensure volume is float32 (same as original h5 data format)
            if volume.dtype != np.float32:
                volume = volume.astype(np.float32)
            
            # Run UNET segmentation (slice-by-slice, identical to original code)
            kidney_prob, tumor_prob = segment_volume_unet(volume, unet_model, device)
            
            # Save UNET outputs to disk (free memory immediately)
            save_volume_as_nifti(kidney_prob, kidney_prob_path, spacing=VOXEL_SPACING)
            save_volume_as_nifti(tumor_prob, tumor_prob_path, spacing=VOXEL_SPACING)
            
            processed += 1
            unet_results[numerical_id] = {
                'kidney_prob': str(kidney_prob_path),
                'tumor_prob': str(tumor_prob_path),
                'status': 'processed'
            }
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
            
            # Explicitly delete from memory to free up space
            del kidney_prob, tumor_prob, volume
            
        except Exception as e:
            tqdm.write(f"    ERROR: Failed to process {numerical_id} with UNET: {e}")
            failed += 1
            unet_results[numerical_id] = {
                'status': 'failed',
                'error': str(e)
            }
            pbar.set_postfix({'Processed': processed, 'Skipped': skipped, 'Failed': failed})
    
    print(f"\nUNET Summary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    
    return unet_results


# ============================================================================
# DICE Score Calculation
# ============================================================================

def calculate_dice_score_3d(pred, target, smooth=1e-7):
    """
    Calculate 3D DICE coefficient between prediction and target.
    
    Args:
        pred: Binary prediction mask (3D numpy array)
        target: Binary target mask (3D numpy array)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        DICE score (float between 0 and 1)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = np.sum(pred_flat * target_flat)
    union = np.sum(pred_flat) + np.sum(target_flat)
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    
    return dice


def calculate_bounding_box(binary_mask):
    """
    Calculate bounding box from binary mask.
    
    Args:
        binary_mask: 3D binary numpy array (D, H, W)
        
    Returns:
        Tuple of (min_d, max_d, min_h, max_h, min_w, max_w) or None if mask is empty
    """
    if np.sum(binary_mask) == 0:
        return None
    
    # Find all non-zero indices
    coords = np.where(binary_mask > 0)
    
    if len(coords[0]) == 0:
        return None
    
    min_d, max_d = np.min(coords[0]), np.max(coords[0])
    min_h, max_h = np.min(coords[1]), np.max(coords[1])
    min_w, max_w = np.min(coords[2]), np.max(coords[2])
    
    return (min_d, max_d, min_h, max_h, min_w, max_w)


def calculate_ts_coverage_in_bbox(unet_binary, ts_binary):
    """
    Calculate what percentage of TS output is contained within UNET bounding box.
    
    Args:
        unet_binary: UNET binary segmentation (3D array)
        ts_binary: TotalSegmentator binary segmentation (3D array)
        
    Returns:
        Percentage (0-100) of TS volume within UNET bounding box, or None if UNET is empty
    """
    # Calculate bounding box from UNET
    bbox = calculate_bounding_box(unet_binary)
    
    if bbox is None:
        return None
    
    min_d, max_d, min_h, max_h, min_w, max_w = bbox
    
    # Extract bounding box region from TS
    ts_in_bbox = ts_binary[min_d:max_d+1, min_h:max_h+1, min_w:max_w+1]
    
    # Calculate total TS volume
    total_ts_volume = np.sum(ts_binary)
    
    if total_ts_volume == 0:
        return None
    
    # Calculate TS volume within bounding box
    ts_volume_in_bbox = np.sum(ts_in_bbox)
    
    # Calculate percentage
    coverage_percentage = (ts_volume_in_bbox / total_ts_volume) * 100.0
    
    return coverage_percentage


def compare_segmentations(unet_kidney_prob, ts_kidney_left, ts_kidney_right, 
                         numerical_id, threshold=UNET_KIDNEY_THRESHOLD):
    """
    Compare UNET and TotalSegmentator kidney segmentations using 3D DICE score.
    
    Args:
        unet_kidney_prob: UNET kidney probability map (3D array)
        ts_kidney_left: TotalSegmentator left kidney segmentation (3D array)
        ts_kidney_right: TotalSegmentator right kidney segmentation (3D array)
        numerical_id: Numerical identifier for this case
        threshold: Threshold for binarizing UNET kidney probabilities
        
    Returns:
        Dictionary with comparison results
    """
    # Binarize UNET kidney segmentation
    unet_kidney_binary = (unet_kidney_prob > threshold).astype(np.uint8)
    
    # Combine TotalSegmentator left and right kidneys
    ts_kidney_combined = ((ts_kidney_left > 0) | (ts_kidney_right > 0)).astype(np.uint8)
    
    # Ensure same shape
    if unet_kidney_binary.shape != ts_kidney_combined.shape:
        # Resize ts_kidney_combined to match unet_kidney_binary
        # This should be rare if both are from same source, but handle it
        zoom_factors = [s1/s2 for s1, s2 in zip(unet_kidney_binary.shape, ts_kidney_combined.shape)]
        ts_kidney_combined = zoom(ts_kidney_combined, zoom_factors, order=0).astype(np.uint8)
    
    # Calculate 3D DICE score
    dice_score = calculate_dice_score_3d(unet_kidney_binary, ts_kidney_combined)
    
    # Calculate volumes (in mm³ with 1mm isotropic spacing)
    unet_volume = np.sum(unet_kidney_binary)
    ts_volume = np.sum(ts_kidney_combined)
    
    # Calculate TS coverage within UNET bounding box
    ts_coverage_in_bbox = calculate_ts_coverage_in_bbox(unet_kidney_binary, ts_kidney_combined)
    
    return {
        'numerical_id': numerical_id,
        'dice_score': dice_score,
        'unet_volume_mm3': unet_volume,
        'ts_volume_mm3': ts_volume,
        'volume_difference_mm3': abs(unet_volume - ts_volume),
        'volume_ratio': unet_volume / (ts_volume + 1e-7),
        'ts_coverage_in_bbox_pct': ts_coverage_in_bbox if ts_coverage_in_bbox is not None else 0.0
    }


def calculate_dice_scores(conversion_results, ts_results, unet_results, output_csv_path=None, overwrite=False):
    """
    Calculate DICE scores for all processed cases.
    
    Args:
        conversion_results: List of dictionaries from convert_dicom_folders
        ts_results: Dictionary mapping numerical_id to TS output paths
        unet_results: Dictionary mapping numerical_id to UNET segmentation results
        output_csv_path: Path to existing DICE CSV file (to load existing results)
        overwrite: If True, recalculate all DICE scores even if they exist
        
    Returns:
        List of dictionaries with DICE results
    """
    print("\n" + "="*70)
    print("STEP 4: Calculating DICE scores")
    print("="*70)
    
    dice_results = []
    existing_dice = {}
    
    # Load existing DICE results if CSV exists and not overwriting
    if output_csv_path and Path(output_csv_path).exists() and not overwrite:
        print(f"\nLoading existing DICE results from: {output_csv_path}")
        try:
            existing_df = pd.read_csv(output_csv_path, dtype={'numerical_id': str})  # Keep as string to preserve leading zeros
            for _, row in existing_df.iterrows():
                numerical_id = str(row['numerical_id']).strip()  # Remove any whitespace
                # Ensure numerical_id has leading zeros format
                # If it's already in "0001" format, keep it; otherwise convert
                try:
                    # Try to convert to int and back to ensure consistent format
                    numerical_id = f"{int(numerical_id):04d}"
                except (ValueError, TypeError):
                    # If conversion fails, use as-is (shouldn't happen, but handle it)
                    numerical_id = str(numerical_id)
                existing_dice[numerical_id] = {
                    'numerical_id': numerical_id,
                    'dice_score': row['dice_score'],
                    'unet_volume_mm3': row.get('unet_volume_mm3', 0),
                    'ts_volume_mm3': row.get('ts_volume_mm3', 0),
                    'volume_difference_mm3': row.get('volume_difference_mm3', 0),
                    'volume_ratio': row.get('volume_ratio', 0),
                    'ts_coverage_in_bbox_pct': row.get('ts_coverage_in_bbox_pct', None)
                }
            print(f"  Loaded {len(existing_dice)} existing DICE results")
            if len(existing_dice) > 0:
                # Show a sample of loaded IDs for debugging
                sample_ids = list(existing_dice.keys())[:5]
                print(f"  Sample IDs loaded: {sample_ids}")
        except Exception as e:
            tqdm.write(f"    WARNING: Failed to load existing DICE results: {e}")
            import traceback
            tqdm.write(f"    Traceback: {traceback.format_exc()}")
            existing_dice = {}
    
    # Filter to only cases with both UNET and TS results
    valid_cases = []
    for result in conversion_results:
        numerical_id = result['numerical_id']
        # Ensure numerical_id has leading zeros format for dictionary lookups
        try:
            numerical_id = f"{int(numerical_id):04d}"
        except (ValueError, TypeError):
            numerical_id = str(numerical_id)
        
        if (numerical_id in unet_results and 
            numerical_id in ts_results and
            unet_results[numerical_id]['status'] in ['processed', 'skipped'] and
            ts_results[numerical_id]['status'] in ['processed', 'skipped']):
            valid_cases.append(result)
    
    print(f"\nProcessing {len(valid_cases)} cases...")
    
    calculated = 0
    skipped = 0
    failed = 0
    
    pbar = tqdm(valid_cases, desc="Calculating DICE scores")
    for result in pbar:
        numerical_id = result['numerical_id']
        # Ensure numerical_id has leading zeros format (CSV might load it as int)
        try:
            numerical_id = f"{int(numerical_id):04d}"
        except (ValueError, TypeError):
            numerical_id = str(numerical_id)
        
        # Check if DICE score and bounding box metric already exist
        if not overwrite and numerical_id in existing_dice:
            existing_result = existing_dice[numerical_id]
            # Check if all metrics exist (both DICE and bounding box coverage)
            has_dice = existing_result.get('dice_score') is not None and not np.isnan(existing_result.get('dice_score', np.nan))
            has_bbox = existing_result.get('ts_coverage_in_bbox_pct') is not None and not np.isnan(existing_result.get('ts_coverage_in_bbox_pct', np.nan))
            
            if has_dice and has_bbox:
                skipped += 1
                dice_results.append(existing_result)
                pbar.set_postfix({'Calculated': calculated, 'Skipped': skipped, 'Failed': failed})
                continue
            else:
                # Some metrics missing, need to recalculate
                missing = []
                if not has_dice:
                    missing.append('dice_score')
                if not has_bbox:
                    missing.append('ts_coverage_in_bbox_pct')
                tqdm.write(f"    INFO: Recalculating {numerical_id} (missing metrics: {', '.join(missing)})")
        elif overwrite and numerical_id in existing_dice:
            # Overwrite mode: recalculate even if exists
            tqdm.write(f"    Recalculating DICE for {numerical_id} (overwrite mode)")
        
        try:
            # Load TS kidney segmentations
            ts_kidney_left_path = Path(ts_results[numerical_id]['kidney_left'])
            ts_kidney_right_path = Path(ts_results[numerical_id]['kidney_right'])
            
            # Load TS outputs and resample to 1mm³ for comparison with UNET
            # TS outputs are at TS's internal resolution (1.5mm standard or 3mm fast)
            # We resample to 1mm³ to match UNET's resolution for fair comparison
            ts_kidney_left = load_nifti_as_volume(ts_kidney_left_path, target_spacing=VOXEL_SPACING)
            ts_kidney_right = load_nifti_as_volume(ts_kidney_right_path, target_spacing=VOXEL_SPACING)
            
            # Load UNET kidney probabilities from disk
            unet_kidney_prob_path = Path(unet_results[numerical_id]['kidney_prob'])
            unet_kidney_prob = load_nifti_as_volume(unet_kidney_prob_path, target_spacing=None)  # Already at 1mm³
            
            # Compare segmentations
            dice_result = compare_segmentations(
                unet_kidney_prob,
                ts_kidney_left,
                ts_kidney_right,
                numerical_id
            )
            
            calculated += 1
            dice_results.append(dice_result)
            pbar.set_postfix({'Calculated': calculated, 'Failed': failed})
            
        except Exception as e:
            tqdm.write(f"    ERROR: Failed to calculate DICE for {numerical_id}: {e}")
            failed += 1
            pbar.set_postfix({'Calculated': calculated, 'Failed': failed})
            continue
    
    print(f"\nDICE Calculation Summary:")
    print(f"  Successfully calculated: {calculated}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(valid_cases)}")
    
    return dice_results


# ============================================================================
# Results Saving and Statistics
# ============================================================================

def save_dice_results(all_dice_results, output_path):
    """
    Save DICE score results to CSV file.
    
    Args:
        all_dice_results: List of dictionaries with DICE results
        output_path: Path to output CSV file
    """
    print(f"\nSaving DICE results to: {output_path}")
    
    # Convert to DataFrame
    df = pd.DataFrame(all_dice_results)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print("Results saved successfully")


def analyze_dice_results(csv_path, dice_threshold=0.01):
    """
    Load DICE results from CSV, filter outliers, and calculate statistics.
    
    Args:
        csv_path: Path to DICE results CSV file
        dice_threshold: Minimum DICE score to include (default: 0.01 = 1%)
        
    Returns:
        Dictionary with filtered statistics
    """
    if not Path(csv_path).exists():
        print(f"\nDICE CSV not found: {csv_path}")
        return None
    
    print(f"\nLoading DICE results from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    print(f"  Total samples in CSV: {len(df)}")
    
    # Filter out low DICE scores
    df_filtered = df[df['dice_score'] >= dice_threshold].copy()
    removed_count = len(df) - len(df_filtered)
    
    print(f"  Removed {removed_count} samples with DICE < {dice_threshold} ({dice_threshold*100}%)")
    print(f"  Remaining samples: {len(df_filtered)}")
    
    if len(df_filtered) == 0:
        print("  WARNING: No samples remaining after filtering!")
        return None
    
    # Calculate DICE statistics
    mean_dice = df_filtered['dice_score'].mean()
    median_dice = df_filtered['dice_score'].median()
    std_dice = df_filtered['dice_score'].std()
    min_dice = df_filtered['dice_score'].min()
    max_dice = df_filtered['dice_score'].max()
    
    print("\n" + "="*70)
    print("FILTERED DICE SCORE STATISTICS (DICE >= 1%)")
    print("="*70)
    print(f"  Mean DICE:   {mean_dice:.4f}")
    print(f"  Median DICE: {median_dice:.4f}")
    print(f"  Std DICE:    {std_dice:.4f}")
    print(f"  Min DICE:    {min_dice:.4f}")
    print(f"  Max DICE:    {max_dice:.4f}")
    print("="*70)
    
    # Calculate TS coverage in bounding box statistics (if available)
    if 'ts_coverage_in_bbox_pct' in df_filtered.columns:
        # Filter out NaN values for this metric
        bbox_df = df_filtered[df_filtered['ts_coverage_in_bbox_pct'].notna()].copy()
        
        if len(bbox_df) > 0:
            mean_bbox = bbox_df['ts_coverage_in_bbox_pct'].mean()
            median_bbox = bbox_df['ts_coverage_in_bbox_pct'].median()
            std_bbox = bbox_df['ts_coverage_in_bbox_pct'].std()
            min_bbox = bbox_df['ts_coverage_in_bbox_pct'].min()
            max_bbox = bbox_df['ts_coverage_in_bbox_pct'].max()
            
            print("\n" + "="*70)
            print("TS COVERAGE IN UNET BOUNDING BOX STATISTICS")
            print("="*70)
            print(f"  Samples with metric: {len(bbox_df)}")
            print(f"  Mean coverage:   {mean_bbox:.2f}%")
            print(f"  Median coverage: {median_bbox:.2f}%")
            print(f"  Std coverage:    {std_bbox:.2f}%")
            print(f"  Min coverage:    {min_bbox:.2f}%")
            print(f"  Max coverage:    {max_bbox:.2f}%")
            print("="*70)
            
            return {
                'total_samples': len(df),
                'filtered_samples': len(df_filtered),
                'removed_samples': removed_count,
                'mean_dice': mean_dice,
                'median_dice': median_dice,
                'std_dice': std_dice,
                'min_dice': min_dice,
                'max_dice': max_dice,
                'mean_bbox_coverage': mean_bbox,
                'median_bbox_coverage': median_bbox,
                'std_bbox_coverage': std_bbox,
                'min_bbox_coverage': min_bbox,
                'max_bbox_coverage': max_bbox,
                'bbox_samples': len(bbox_df),
                'df_filtered': df_filtered
            }
        else:
            print("\n  WARNING: No valid TS coverage in bounding box metrics found")
    
    return {
        'total_samples': len(df),
        'filtered_samples': len(df_filtered),
        'removed_samples': removed_count,
        'mean_dice': mean_dice,
        'median_dice': median_dice,
        'std_dice': std_dice,
        'min_dice': min_dice,
        'max_dice': max_dice,
        'df_filtered': df_filtered
    }


def find_max_unet_slice(kidney_prob_volume):
    """
    Find the slice index with the maximum UNET kidney probability.
    
    Args:
        kidney_prob_volume: 3D numpy array (D, H, W) with kidney probabilities
        
    Returns:
        Slice index (int) with maximum sum of probabilities
    """
    # Sum probabilities across each slice
    slice_sums = np.sum(kidney_prob_volume, axis=(1, 2))
    max_slice_idx = np.argmax(slice_sums)
    return max_slice_idx


def create_visualization(numerical_id, original_nii_path, unet_kidney_prob_path, 
                         ts_kidney_left_path, ts_kidney_right_path, output_path):
    """
    Create a visualization showing UNET and TS segmentations on a CT slice.
    
    Steps:
    1. Find the slice with largest UNET output
    2. Get TS output from the same slice
    3. Get original CT slice from the same slice
    4. Verify sizes match, interpolate segmentation slices to match original if needed
    5. Plot overlays: UNET=green, TS=red, overlap=purple
    6. Save as PNG
    
    Args:
        numerical_id: Numerical ID for this case
        original_nii_path: Path to original NIfTI file (for CT image)
        unet_kidney_prob_path: Path to UNET kidney probability map
        ts_kidney_left_path: Path to TS left kidney segmentation
        ts_kidney_right_path: Path to TS right kidney segmentation
        output_path: Path to save PNG file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Step 1: Load UNET kidney probabilities and find slice with largest output
        unet_kidney_prob = load_nifti_as_volume(unet_kidney_prob_path, target_spacing=None)
        slice_idx = find_max_unet_slice(unet_kidney_prob)
        
        # Step 2: Get TS output from the same slice
        ts_kidney_left = load_nifti_as_volume(ts_kidney_left_path, target_spacing=VOXEL_SPACING)
        ts_kidney_right = load_nifti_as_volume(ts_kidney_right_path, target_spacing=VOXEL_SPACING)
        
        # Ensure slice_idx is within bounds for all volumes
        min_depth = min(
            unet_kidney_prob.shape[0],
            ts_kidney_left.shape[0],
            ts_kidney_right.shape[0]
        )
        if slice_idx >= min_depth:
            slice_idx = min_depth - 1
        if slice_idx < 0:
            slice_idx = 0
        
        # Get TS slices from the same slice index
        ts_left_slice = ts_kidney_left[slice_idx, :, :]
        ts_right_slice = ts_kidney_right[slice_idx, :, :]
        ts_combined_slice = ((ts_left_slice > 0) | (ts_right_slice > 0)).astype(float)
        
        # Step 3: Get original CT slice from the same slice
        original_volume = load_nifti_as_volume(original_nii_path, target_spacing=VOXEL_SPACING)
        
        # Ensure slice_idx is within bounds for original volume
        if slice_idx >= original_volume.shape[0]:
            slice_idx = original_volume.shape[0] - 1
        if slice_idx < 0:
            slice_idx = 0
        
        ct_slice = original_volume[slice_idx, :, :]
        target_shape = ct_slice.shape  # This is our reference size
        
        # Step 4: Get UNET slice and verify sizes match
        unet_slice = unet_kidney_prob[slice_idx, :, :]
        
        # Step 5: Verify sizes match, interpolate segmentation slices to match original if needed
        from scipy.ndimage import zoom
        
        # Resize UNET slice to match original CT slice size if needed
        if unet_slice.shape != target_shape:
            zoom_factors = [t/s for t, s in zip(target_shape, unet_slice.shape)]
            unet_slice = zoom(unet_slice, zoom_factors, order=1)  # Linear interpolation for probabilities
        
        # Resize TS combined slice to match original CT slice size if needed
        if ts_combined_slice.shape != target_shape:
            zoom_factors = [t/s for t, s in zip(target_shape, ts_combined_slice.shape)]
            ts_combined_slice = zoom(ts_combined_slice, zoom_factors, order=0)  # Nearest neighbor for binary
        
        # Binarize UNET output (using threshold)
        unet_binary_slice = (unet_slice > UNET_KIDNEY_THRESHOLD).astype(float)
        
        # Step 6: Create overlays - UNET=green, TS=red, overlap=purple
        # Create RGB overlay image
        overlay = np.zeros((*target_shape, 3), dtype=np.float32)
        
        # UNET in green (channel 1)
        unet_mask = unet_binary_slice > 0
        overlay[unet_mask, 1] = 1.0  # Green channel
        
        # TS in red (channel 0)
        ts_mask = ts_combined_slice > 0
        overlay[ts_mask, 0] = 1.0  # Red channel
        
        # Overlap in purple (both red and blue channels)
        overlap_mask = (unet_mask & ts_mask)
        overlay[overlap_mask, 0] = 1.0  # Red channel
        overlay[overlap_mask, 2] = 1.0  # Blue channel (red + blue = purple)
        overlay[overlap_mask, 1] = 0.0  # Remove green from overlap
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Display CT image in grayscale
        ax.imshow(ct_slice, cmap='gray', vmin=-150, vmax=250)
        
        # Overlay segmentations
        ax.imshow(overlay, alpha=0.6)
        
        # Add legend
        green_patch = mpatches.Patch(color='green', alpha=0.6, label='UNET')
        red_patch = mpatches.Patch(color='red', alpha=0.6, label='TotalSegmentator')
        purple_patch = mpatches.Patch(color='purple', alpha=0.6, label='Overlap')
        ax.legend(handles=[green_patch, red_patch, purple_patch], loc='upper right')
        
        # Add title
        ax.set_title(f'Case {numerical_id} - Slice {slice_idx}\nUNET (green), TS (red), Overlap (purple)', 
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        # Step 7: Save as PNG
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return True
        
    except Exception as e:
        import traceback
        tqdm.write(f"    ERROR: Failed to create visualization for {numerical_id}: {e}")
        tqdm.write(f"    Traceback: {traceback.format_exc()}")
        return False


def create_visualizations(dice_csv_path, conversion_results, ts_results, unet_results,
                         output_dir, num_examples=10, overwrite=False):
    """
    Create visualization images for randomly selected examples.
    
    Steps:
    1. Randomly select num_examples images from valid cases
    2. For each image, follow the visualization workflow
    
    Args:
        dice_csv_path: Path to DICE results CSV
        conversion_results: List of conversion results
        ts_results: Dictionary of TS results
        unet_results: Dictionary of UNET results
        output_dir: Output directory for visualizations
        num_examples: Number of examples to visualize (default: 10)
        overwrite: If True, overwrite existing visualizations
        
    Returns:
        Number of visualizations created
    """
    print("\n" + "="*70)
    print("STEP 5: Creating Visualization Images")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load DICE results
    if not Path(dice_csv_path).exists():
        print(f"  ERROR: DICE CSV not found: {dice_csv_path}")
        return 0
    
    # Read CSV and ensure numerical_id is treated as string to preserve format
    df = pd.read_csv(dice_csv_path, dtype={'numerical_id': str})
    print(f"  Loaded {len(df)} DICE results")
    
    # Filter to valid cases (DICE >= 1%)
    df_filtered = df[df['dice_score'] >= 0.01].copy()
    
    if len(df_filtered) == 0:
        print("  WARNING: No valid cases after filtering (DICE >= 1%)")
        return 0
    
    # Step 1: Randomly select num_examples images from valid cases
    num_examples = min(num_examples, len(df_filtered))
    selected_examples = df_filtered.sample(n=num_examples, random_state=42).copy()
    
    print(f"  Randomly selected {num_examples} examples from {len(df_filtered)} valid cases")
    
    # Create mapping from numerical_id to conversion results
    conversion_dict = {}
    for result in conversion_results:
        numerical_id = result['numerical_id']
        try:
            numerical_id = f"{int(numerical_id):04d}"
        except (ValueError, TypeError):
            numerical_id = str(numerical_id)
        conversion_dict[numerical_id] = result
    
    # Debug: Show what IDs are in conversion_dict vs selected examples
    selected_ids = []
    for _, row in selected_examples.iterrows():
        numerical_id_raw = str(row['numerical_id']).strip()
        try:
            numerical_id = f"{int(float(numerical_id_raw)):04d}"  # Handle both "617" and "617.0"
        except (ValueError, TypeError):
            numerical_id = numerical_id_raw.zfill(4)
        selected_ids.append(numerical_id)
    
    missing_in_conversion = [id for id in selected_ids if id not in conversion_dict]
    if missing_in_conversion:
        tqdm.write(f"    DEBUG: {len(missing_in_conversion)} selected IDs not in conversion_dict: {missing_in_conversion[:5]}")
        tqdm.write(f"    DEBUG: Sample conversion_dict keys: {list(conversion_dict.keys())[:5]}")
    
    created = 0
    skipped = 0
    failed = 0
    
    pbar = tqdm(selected_examples.iterrows(), total=len(selected_examples), desc="Creating visualizations")
    for idx, row in pbar:
        numerical_id_raw = str(row['numerical_id']).strip()
        # Ensure numerical_id has leading zeros - handle both "617" and "617.0" formats
        try:
            # Convert to float first to handle "617.0", then to int, then format
            numerical_id = f"{int(float(numerical_id_raw)):04d}"
        except (ValueError, TypeError):
            # If it's already a string with leading zeros, try to preserve it
            try:
                numerical_id = f"{int(numerical_id_raw):04d}"
            except:
                numerical_id = numerical_id_raw.zfill(4)
        
        output_path = output_dir / f"{numerical_id}_visualization.png"
        
        # Check if already exists
        if output_path.exists() and not overwrite:
            skipped += 1
            pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
            continue
        
        # Get paths
        if numerical_id not in conversion_dict:
            tqdm.write(f"    WARNING: No conversion result found for {numerical_id}")
            failed += 1
            pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
            continue
        
        if numerical_id not in ts_results or numerical_id not in unet_results:
            tqdm.write(f"    WARNING: Missing TS or UNET results for {numerical_id}")
            failed += 1
            pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
            continue
        
        original_nii_path = Path(conversion_dict[numerical_id]['nii_path'])
        unet_kidney_prob_path = Path(unet_results[numerical_id]['kidney_prob'])
        ts_kidney_left_path = Path(ts_results[numerical_id]['kidney_left'])
        ts_kidney_right_path = Path(ts_results[numerical_id]['kidney_right'])
        
        # Verify all files exist
        if not original_nii_path.exists():
            tqdm.write(f"    WARNING: Original NIfTI not found: {original_nii_path}")
            failed += 1
            pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
            continue
        
        if not unet_kidney_prob_path.exists():
            tqdm.write(f"    WARNING: UNET output not found: {unet_kidney_prob_path}")
            failed += 1
            pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
            continue
        
        if not ts_kidney_left_path.exists() or not ts_kidney_right_path.exists():
            tqdm.write(f"    WARNING: TS outputs not found for {numerical_id}")
            failed += 1
            pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
            continue
        
        # Create visualization (follows steps 2-7 internally)
        success = create_visualization(
            numerical_id,
            original_nii_path,
            unet_kidney_prob_path,
            ts_kidney_left_path,
            ts_kidney_right_path,
            output_path
        )
        
        if success:
            created += 1
        else:
            failed += 1
        pbar.set_postfix({'Created': created, 'Skipped': skipped, 'Failed': failed})
    
    print(f"\nVisualization Summary:")
    print(f"  Created: {created}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {failed}")
    
    return created


def print_dice_statistics(all_dice_results):
    """
    Print summary statistics of DICE scores.
    
    Args:
        all_dice_results: List of dictionaries with DICE results
    """
    if not all_dice_results:
        print("\nNo DICE results to display")
        return
    
    df = pd.DataFrame(all_dice_results)
    
    print("\n" + "="*70)
    print("DICE SCORE STATISTICS")
    print("="*70)
    
    # Overall statistics
    print("\nOverall Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Mean DICE: {df['dice_score'].mean():.4f}")
    print(f"  Std DICE:  {df['dice_score'].std():.4f}")
    print(f"  Min DICE:  {df['dice_score'].min():.4f}")
    print(f"  Max DICE:  {df['dice_score'].max():.4f}")
    print(f"  Median DICE: {df['dice_score'].median():.4f}")
    
    # Volume statistics
    print("\n\nVolume Statistics:")
    print(f"  Mean UNET volume: {df['unet_volume_mm3'].mean():.1f} mm³")
    print(f"  Mean TS volume:   {df['ts_volume_mm3'].mean():.1f} mm³")
    print(f"  Mean volume ratio (UNET/TS): {df['volume_ratio'].mean():.3f}")
    
    # Worst cases
    print("\n\nWorst 5 DICE Scores:")
    worst_5 = df.nsmallest(5, 'dice_score')[['numerical_id', 'dice_score', 'volume_ratio']]
    print(worst_5.to_string(index=False))
    
    # Best cases
    print("\n\nBest 5 DICE Scores:")
    best_5 = df.nlargest(5, 'dice_score')[['numerical_id', 'dice_score', 'volume_ratio']]
    print(best_5.to_string(index=False))


# ============================================================================
# Main Function
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare UNET and TotalSegmentator kidney segmentations from DICOM folders'
    )
    parser.add_argument(
        '--base_folder',
        type=str,
        required=True,
        help='Base folder path to replace in metadata spreadsheet paths'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=METADATA_XLSX,
        help=f'Path to metadata Excel file (default: {METADATA_XLSX})'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory (will create subdirectories for converted NIfTI and TS outputs)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=UNET_KIDNEY_THRESHOLD,
        help=f'Threshold for binarizing UNET kidney probabilities (default: {UNET_KIDNEY_THRESHOLD})'
    )
    parser.add_argument(
        '--overwrite_conv',
        action='store_true',
        help='Overwrite existing converted NIfTI files'
    )
    parser.add_argument(
        '--overwrite_ts',
        action='store_true',
        help='Overwrite existing TotalSegmentator outputs'
    )
    parser.add_argument(
        '--overwrite_unet',
        action='store_true',
        help='Overwrite existing UNET outputs'
    )
    parser.add_argument(
        '--skip_conv',
        action='store_true',
        help='Skip DICOM conversion step (use existing NIfTI files)'
    )
    parser.add_argument(
        '--skip_ts',
        action='store_true',
        help='Skip TotalSegmentator step (use existing TS outputs)'
    )
    parser.add_argument(
        '--skip_unet',
        action='store_true',
        help='Skip UNET step (use existing UNET results)'
    )
    parser.add_argument(
        '--skip_dice',
        action='store_true',
        help='Skip DICE calculation step (use existing DICE results from CSV)'
    )
    parser.add_argument(
        '--overwrite_dice',
        action='store_true',
        help='Overwrite existing DICE scores (recalculate all)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=1,
        help='Number of worker threads for DICOM to NIfTI conversion (default: 1)'
    )
    parser.add_argument(
        '--ts_num_gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for TotalSegmentator (None or 1 = sequential with 1 GPU, 2+ = parallel with multiple GPUs, default: None). Default uses 1 GPU sequentially (optimal).'
    )
    parser.add_argument(
        '--ts_use_cpu',
        action='store_true',
        help='Force TotalSegmentator to use CPU instead of GPU'
    )
    parser.add_argument(
        '--num_viz_examples',
        type=int,
        default=10,
        help='Number of visualization examples to create (default: 10)'
    )
    parser.add_argument(
        '--overwrite_viz',
        action='store_true',
        help='Overwrite existing visualization images'
    )
    parser.add_argument(
        '--skip_viz',
        action='store_true',
        help='Skip visualization creation step'
    )
    
    return parser.parse_args()


def main():
    """Main function to process DICOM folders and compare segmentations."""
    
    # Parse arguments
    args = parse_args()
    
    # Update global threshold if specified
    global UNET_KIDNEY_THRESHOLD
    UNET_KIDNEY_THRESHOLD = args.threshold
    
    print("="*70)
    print("UNET vs TotalSegmentator DICE Score Comparison (DICOM Input)")
    print("="*70)
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    converted_nii_dir = output_dir / 'converted_nii'
    ts_output_dir = output_dir / 'ts_outputs'
    unet_output_dir = output_dir / 'unet_outputs'
    
    print(f"\nConfiguration:")
    print(f"  Base folder: {args.base_folder}")
    print(f"  Metadata file: {args.metadata}")
    print(f"  Output directory: {output_dir}")
    print(f"  Converted NIfTI directory: {converted_nii_dir}")
    print(f"  TotalSegmentator output directory: {ts_output_dir}")
    print(f"  UNET output directory: {unet_output_dir}")
    print(f"  UNET kidney threshold: {UNET_KIDNEY_THRESHOLD}")
    print(f"  Overwrite converted: {args.overwrite_conv}")
    print(f"  Overwrite TS outputs: {args.overwrite_ts}")
    
    # Check if files exist
    if not Path(args.metadata).exists():
        print(f"ERROR: Metadata file not found: {args.metadata}")
        return
    
    if not Path(UNET_MODEL_PATH).exists():
        print(f"ERROR: UNET model not found: {UNET_MODEL_PATH}")
        return
    
    if not Path(TS_EXECUTABLE).exists():
        print(f"ERROR: TotalSegmentator executable not found: {TS_EXECUTABLE}")
        return
    
    # Load metadata
    print(f"\nLoading metadata from: {args.metadata}")
    try:
        metadata_df = pd.read_excel(args.metadata)
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return
    
    if 'Folder' not in metadata_df.columns:
        print("ERROR: Metadata file must contain 'Folder' column")
        return
    
    print(f"  Loaded {len(metadata_df)} rows from metadata")
    
    # Setup device
    torch.cuda.set_device(DEVICE_ID)
    device = torch.device(f'cuda:{DEVICE_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"TotalSegmentator mode: {'FAST' if TS_FAST else 'STANDARD'}")
    
    # Step 1: Convert DICOM to NIfTI
    if not args.skip_conv:
        conversion_results, mapping_df = convert_dicom_folders(
            metadata_df,
            args.base_folder,
            converted_nii_dir,
            overwrite=args.overwrite_conv,
            num_workers=args.num_workers
        )
    else:
        print("\nSkipping DICOM conversion step...")
        # Load existing conversion results from mapping file
        mapping_file = converted_nii_dir / 'folder_mapping.csv'
        if mapping_file.exists():
            mapping_df = pd.read_csv(mapping_file)
            conversion_results = []
            for _, row in mapping_df.iterrows():
                # Ensure numerical_id has leading zeros (CSV might load it as int)
                try:
                    numerical_id = f"{int(row['numerical_id']):04d}"
                except (ValueError, TypeError):
                    numerical_id = str(row['numerical_id'])
                nii_path = converted_nii_dir / f"{numerical_id}.nii.gz"
                if nii_path.exists():
                    conversion_results.append({
                        'folder_path': row.get('server_folder_path', 'unknown'),
                        'original_folder_path': row['original_folder_path'],
                        'nii_path': str(nii_path),
                        'status': 'skipped',
                        'numerical_id': numerical_id
                    })
            print(f"  Found {len(conversion_results)} existing NIfTI files from mapping")
        else:
            print(f"  ERROR: Mapping file not found: {mapping_file}")
            print(f"  Cannot skip conversion without mapping file")
            return
    
    # Step 2: Run TotalSegmentator
    if not args.skip_ts:
        ts_results = run_totalsegmentator_on_nii_files(
            conversion_results,
            ts_output_dir,
            overwrite=args.overwrite_ts,
            num_gpus=args.ts_num_gpus,
            use_gpu=not args.ts_use_cpu
        )
    else:
        print("\nSkipping TotalSegmentator step...")
        # Load existing TS results
        ts_results = {}
        for case_dir in ts_output_dir.iterdir():
            if case_dir.is_dir():
                numerical_id = case_dir.name
                # Ensure numerical_id has leading zeros format for consistency
                try:
                    numerical_id = f"{int(numerical_id):04d}"
                except (ValueError, TypeError):
                    numerical_id = str(numerical_id)
                kidney_left_path = case_dir / 'kidney_left.nii.gz'
                kidney_right_path = case_dir / 'kidney_right.nii.gz'
                if kidney_left_path.exists() and kidney_right_path.exists():
                    ts_results[numerical_id] = {
                        'kidney_left': str(kidney_left_path),
                        'kidney_right': str(kidney_right_path),
                        'status': 'skipped'
                    }
        print(f"  Found {len(ts_results)} existing TS outputs")
    
    # Step 3: Run UNET
    if not args.skip_unet:
        unet_model = load_unet_model(UNET_MODEL_PATH, device)
        unet_results = run_unet_on_nii_files(
            conversion_results, 
            unet_model, 
            device, 
            unet_output_dir,
            overwrite=args.overwrite_unet if hasattr(args, 'overwrite_unet') else False
        )
    else:
        print("\nSkipping UNET step...")
        # Load existing UNET results from disk
        unet_results = {}
        unet_output_dir.mkdir(parents=True, exist_ok=True)
        for prob_file in unet_output_dir.glob('*_kidney_prob.nii.gz'):
            numerical_id = prob_file.stem.replace('_kidney_prob', '').replace('.nii', '')
            # Ensure numerical_id has leading zeros format
            try:
                numerical_id = f"{int(numerical_id):04d}"
            except (ValueError, TypeError):
                numerical_id = str(numerical_id)
            
            tumor_prob_path = unet_output_dir / f"{numerical_id}_tumor_prob.nii.gz"
            if tumor_prob_path.exists():
                unet_results[numerical_id] = {
                    'kidney_prob': str(prob_file),
                    'tumor_prob': str(tumor_prob_path),
                    'status': 'skipped'
                }
        print(f"  Found {len(unet_results)} existing UNET outputs")
    
    # Step 4: Calculate DICE scores
    if not args.skip_dice:
        dice_results = calculate_dice_scores(
            conversion_results, 
            ts_results, 
            unet_results,
            output_csv_path=OUTPUT_CSV,
            overwrite=args.overwrite_dice if hasattr(args, 'overwrite_dice') else False
        )
        
        # Print statistics
        print_dice_statistics(dice_results)
        
        # Save results to CSV
        if dice_results:
            save_dice_results(dice_results, OUTPUT_CSV)
    else:
        print("\nSkipping DICE calculation step...")
        # Load existing DICE results for analysis
        if Path(OUTPUT_CSV).exists():
            existing_df = pd.read_csv(OUTPUT_CSV)
            dice_results = existing_df.to_dict('records')
            print(f"  Loaded {len(dice_results)} existing DICE results from CSV")
        else:
            print(f"  ERROR: DICE CSV not found: {OUTPUT_CSV}")
            dice_results = []
    
    # Step 5: Analyze DICE results (load from CSV, filter outliers, calculate stats)
    analyze_dice_results(OUTPUT_CSV, dice_threshold=0.01)
    
    # Step 6: Create visualization images
    if not args.skip_viz:
        viz_output_dir = output_dir / 'visualizations'
        create_visualizations(
            OUTPUT_CSV,
            conversion_results,
            ts_results,
            unet_results,
            viz_output_dir,
            num_examples=args.num_viz_examples,
            overwrite=args.overwrite_viz
        )
    else:
        print("\nSkipping visualization creation step...")
    
    print("\n" + "="*70)
    print("Processing complete!")
    print("="*70)
    print(f"\nDICE results saved to: {OUTPUT_CSV}")
    print(f"Output directory: {output_dir}")
    print(f"Converted NIfTI files: {converted_nii_dir}")
    print(f"TotalSegmentator outputs: {ts_output_dir}")
    print(f"UNET outputs: {unet_output_dir}")
    if not args.skip_viz:
        print(f"Visualizations: {output_dir / 'visualizations'}")
    print(f"Folder mapping file: {converted_nii_dir / 'folder_mapping.csv'}")


if __name__ == '__main__':
    main()

