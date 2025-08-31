#!/usr/bin/env python3
"""
Organized Visualization Generator for LeFusion Methods
Creates structured folders and comprehensive visualizations
"""

import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

class OrganizedVisualizationGenerator:
    def __init__(self, base_dir="/Users/skb/Documents/LeFusion"):
        self.base_dir = Path(base_dir)
        self.synthetic_data_dir = self.base_dir / "evaluation_training" / "synthetic_data" / "hf_synthetic_data" / "lidc"
        self.real_data_dir = self.base_dir / "data" / "LIDC"
        
        # Create main visualization directory with model_type subfolders
        self.vis_dir = self.base_dir / "evaluation_training" / "visualizations"
        
        # Create directories
        self.vis_dir.mkdir(exist_ok=True)
        
        # Define techniques and conditions (based on actual data structure)
        self.techniques = {
            'lefusion': {
                'name': 'LeFusion',
                'conditions': ['P_P_prime'],
                'description': 'Basic LeFusion without histogram control'
            },
            'lefusion_h': {
                'name': 'LeFusion-H',  
                'conditions': ['P_N_prime', 'P_P_prime'],
                'description': 'LeFusion with Histogram-based texture control'
            },
            'lefusion_h_diffmask': {
                'name': 'LeFusion-H-DiffMask',
                'conditions': ['P_N_prime', 'P_N_double_prime', 'P_P_prime_N_double_prime'],
                'description': 'LeFusion-H enhanced with DiffMask for lesion mask generation'
            }
        }
        
        # Get available cases from synthetic data
        self.available_cases = self.get_available_cases()
        
        print(f"📁 Visualization directory: {self.vis_dir}")
        print(f"🔍 Found {len(self.available_cases)} available cases")
        
    def get_available_cases(self):
        """Get all available patient cases from synthetic data"""
        cases = set()
        
        # Check both pretrained and from_scratch
        for model_type in ['pretrained', 'from_scratch']:
            for technique in self.techniques.keys():
                for condition in self.techniques[technique]['conditions']:
                    images_dir = self.synthetic_data_dir / model_type / technique / condition / "imagesTr"
                    if images_dir.exists():
                        # Check subdirectories first
                        for img_subdir in ["Image_1", "Image_2", "Image_3"]:
                            sub_dir = images_dir / img_subdir
                            if sub_dir.exists():
                                for img_file in sub_dir.glob("*.nii.gz"):
                                    # Extract patient_id and vol_id from filename
                                    name = img_file.stem.replace('.nii', '')
                                    if '_CVol_' in name:
                                        parts = name.split('_CVol_')
                                        patient_id = parts[0]
                                        vol_id = parts[1]
                                        cases.add((patient_id, vol_id))
                                    elif '_Vol_' in name:
                                        parts = name.split('_Vol_')
                                        patient_id = parts[0]
                                        vol_id = parts[1]
                                        cases.add((patient_id, vol_id))
                        
                        # Also check direct imagesTr folder
                        for img_file in images_dir.glob("*.nii.gz"):
                            # Extract patient_id and vol_id from filename
                            name = img_file.stem.replace('.nii', '')
                            if '_CVol_' in name:
                                parts = name.split('_CVol_')
                                patient_id = parts[0]
                                vol_id = parts[1]
                                cases.add((patient_id, vol_id))
                            elif '_Vol_' in name:
                                parts = name.split('_Vol_')
                                patient_id = parts[0]
                                vol_id = parts[1]
                                cases.add((patient_id, vol_id))
        
        return sorted(list(cases))  # Return all available cases
    
    def load_nifti_slice(self, file_path, slice_idx=None):
        """Load a specific slice from NiFTI file"""
        try:
            if not os.path.exists(file_path):
                return None, None
                
            img = nib.load(file_path)
            data = img.get_fdata()
            
            if len(data.shape) != 3:
                return None, None
            
            # Find slice with lesion if it's a mask, otherwise use middle slice
            if slice_idx is None:
                if 'mask' in file_path.lower() or 'label' in file_path.lower():
                    # For masks, find slice with maximum content
                    slice_sums = [np.sum(data[:, :, i] > 0) for i in range(data.shape[2])]
                    if max(slice_sums) > 0:
                        slice_idx = np.argmax(slice_sums)
                    else:
                        slice_idx = data.shape[2] // 2
                else:
                    slice_idx = data.shape[2] // 2
            
            slice_data = data[:, :, slice_idx]
            return slice_data, slice_idx
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def find_original_image(self, patient_id, vol_id):
        """Find original LIDC image"""
        # Check Normal directory first
        normal_image_path = self.real_data_dir / "Normal" / "Image" / f"{patient_id}_CVol_{vol_id}.nii.gz"
        if normal_image_path.exists():
            return str(normal_image_path)
        
        # Check Pathological directory
        pathological_dir = self.real_data_dir / "Pathological" / "Image" / patient_id
        if pathological_dir.exists():
            for img_file in pathological_dir.glob("*.nii.gz"):
                if vol_id in img_file.stem:
                    return str(img_file)
        
        return None
    
    def find_original_mask(self, patient_id, vol_id):
        """Find original LIDC mask"""
        # Check Normal directory first
        normal_mask_path = self.real_data_dir / "Normal" / "Mask" / f"{patient_id}_CMask_{vol_id}.nii.gz"
        if normal_mask_path.exists():
            return str(normal_mask_path)
        
        # Check Pathological directory
        pathological_mask_dir = self.real_data_dir / "Pathological" / "Mask" / patient_id
        if pathological_mask_dir.exists():
            for mask_file in pathological_mask_dir.glob("*.nii.gz"):
                if vol_id in mask_file.stem:
                    return str(mask_file)
        
        return None
    
    def find_synthetic_data_all_variants(self, technique, condition, model_type, patient_id, vol_id):
        """Find all 3 synthetic image variants and their corresponding masks for given parameters"""
        technique_path = self.synthetic_data_dir / model_type / technique / condition
        
        all_files = {}
        
        # Look for image files in imagesTr subdirectories (Image_1, Image_2, Image_3)
        images_dir = technique_path / "imagesTr"
        if images_dir.exists():
            has_subdirs = False
            for idx, img_subdir in enumerate(["Image_1", "Image_2", "Image_3"], 1):
                img_dir = images_dir / img_subdir
                if img_dir.exists():
                    has_subdirs = True
                    # Try different file patterns
                    patterns = [f"{patient_id}_CVol_{vol_id}.nii.gz", f"{patient_id}_Vol_{vol_id}.nii.gz"]
                    for pattern in patterns:
                        img_path = img_dir / pattern
                        if img_path.exists():
                            all_files[f'image_{idx}'] = str(img_path)
                            break
            
            # If no subdirs exist, use same file for all 3 variants (e.g., for lefusion_h_diffmask)
            if not has_subdirs:
                patterns = [f"{patient_id}_CVol_{vol_id}.nii.gz", f"{patient_id}_Vol_{vol_id}.nii.gz"]
                for pattern in patterns:
                    img_path = images_dir / pattern
                    if img_path.exists():
                        # Use the same file for all 3 variants
                        for idx in range(1, 4):
                            all_files[f'image_{idx}'] = str(img_path)
                        break
        
        # Look for mask files in labelsTr subdirectories (Mask_1, Mask_2, Mask_3)
        labels_dir = technique_path / "labelsTr"
        if labels_dir.exists():
            has_subdirs = False
            for idx, mask_subdir in enumerate(["Mask_1", "Mask_2", "Mask_3"], 1):
                mask_dir = labels_dir / mask_subdir
                if mask_dir.exists():
                    has_subdirs = True
                    # Try different mask patterns
                    patterns = [f"{patient_id}_GenMask_{vol_id}.nii.gz", f"{patient_id}_Mask_{vol_id}.nii.gz", f"{patient_id}_CMask_{vol_id}.nii.gz"]
                    for pattern in patterns:
                        mask_path = mask_dir / pattern
                        if mask_path.exists():
                            all_files[f'mask_{idx}'] = str(mask_path)
                            break
            
            # If no subdirs exist, use same file for all 3 variants
            if not has_subdirs:
                patterns = [f"{patient_id}_GenMask_{vol_id}.nii.gz", f"{patient_id}_Mask_{vol_id}.nii.gz", f"{patient_id}_CMask_{vol_id}.nii.gz"]
                for pattern in patterns:
                    mask_path = labels_dir / pattern
                    if mask_path.exists():
                        # Use the same file for all 3 variants
                        for idx in range(1, 4):
                            all_files[f'mask_{idx}'] = str(mask_path)
                        break
        
        return all_files
    
    def find_synthetic_data(self, technique, condition, model_type, patient_id, vol_id):
        """Find synthetic image and mask for given parameters (for overview - just first variant)"""
        technique_path = self.synthetic_data_dir / model_type / technique / condition
        
        files = {}
        
        # Look for image files in imagesTr subdirectories
        images_dir = technique_path / "imagesTr"
        if images_dir.exists():
            # Check subdirectories Image_1, Image_2, Image_3
            for img_subdir in ["Image_1", "Image_2", "Image_3"]:
                img_dir = images_dir / img_subdir
                if img_dir.exists():
                    # Try different file patterns
                    patterns = [f"{patient_id}_CVol_{vol_id}.nii.gz", f"{patient_id}_Vol_{vol_id}.nii.gz"]
                    for pattern in patterns:
                        img_path = img_dir / pattern
                        if img_path.exists():
                            files['image'] = str(img_path)
                            break
                    if 'image' in files:
                        break
            
            # If not found in subdirectories, try direct imagesTr folder
            if 'image' not in files:
                patterns = [f"{patient_id}_CVol_{vol_id}.nii.gz", f"{patient_id}_Vol_{vol_id}.nii.gz"]
                for pattern in patterns:
                    img_path = images_dir / pattern
                    if img_path.exists():
                        files['image'] = str(img_path)
                        break
        
        # Look for mask files in labelsTr subdirectories
        labels_dir = technique_path / "labelsTr"
        if labels_dir.exists():
            # Check subdirectories Mask_1, Mask_2, Mask_3
            for mask_subdir in ["Mask_1", "Mask_2", "Mask_3"]:
                mask_dir = labels_dir / mask_subdir
                if mask_dir.exists():
                    # Try different mask patterns
                    patterns = [f"{patient_id}_GenMask_{vol_id}.nii.gz", f"{patient_id}_Mask_{vol_id}.nii.gz", f"{patient_id}_CMask_{vol_id}.nii.gz"]
                    for pattern in patterns:
                        mask_path = mask_dir / pattern
                        if mask_path.exists():
                            files['mask'] = str(mask_path)
                            break
                    if 'mask' in files:
                        break
            
            # If not found in subdirectories, try direct labelsTr folder
            if 'mask' not in files:
                patterns = [f"{patient_id}_GenMask_{vol_id}.nii.gz", f"{patient_id}_Mask_{vol_id}.nii.gz", f"{patient_id}_CMask_{vol_id}.nii.gz"]
                for pattern in patterns:
                    mask_path = labels_dir / pattern
                    if mask_path.exists():
                        files['mask'] = str(mask_path)
                        break
        
        return files
    
    def create_method_visualization(self, technique_key, patient_id, vol_id, model_type="pretrained"):
        """Create visualization for a single method showing Normal, Mask, and all 3 image variants per condition"""
        technique_info = self.techniques[technique_key]
        
        # Create model_type-specific directory structure
        model_vis_dir = self.vis_dir / model_type
        method_vis_dir = model_vis_dir / "individual_methods"
        method_dir = method_vis_dir / technique_key
        
        model_vis_dir.mkdir(exist_ok=True)
        method_vis_dir.mkdir(exist_ok=True)
        method_dir.mkdir(exist_ok=True)
        
        # Calculate figure size: Normal + Mask + 3 images only
        n_cols = 2 + 3  # Normal + Mask + Image_1 + Image_2 + Image_3
        
        fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        
        # Load original data and find reference slice
        original_img_path = self.find_original_image(patient_id, vol_id)
        original_mask_path = self.find_original_mask(patient_id, vol_id)
        reference_slice_idx = None
        
        # Column 1: Normal/Original
        if original_img_path:
            img_data, slice_idx = self.load_nifti_slice(original_img_path)
            reference_slice_idx = slice_idx
            
            if img_data is not None:
                img_data = np.rot90(img_data)
                axes[0].imshow(img_data, cmap='gray', aspect='equal')
                axes[0].set_title('Normal', fontsize=12, weight='bold')
                axes[0].axis('off')
            else:
                axes[0].text(0.5, 0.5, 'Normal\nLoad Failed', ha='center', va='center',
                           transform=axes[0].transAxes, fontsize=10)
                axes[0].axis('off')
        else:
            axes[0].text(0.5, 0.5, 'Normal\nNot Found', ha='center', va='center',
                       transform=axes[0].transAxes, fontsize=10)
            axes[0].axis('off')
        
        # Column 2: Mask overlay
        if original_img_path and original_mask_path:
            img_data, _ = self.load_nifti_slice(original_img_path, reference_slice_idx)
            mask_data, _ = self.load_nifti_slice(original_mask_path, reference_slice_idx)
            
            if img_data is not None and mask_data is not None:
                img_data = np.rot90(img_data)
                mask_data = np.rot90(mask_data)
                
                # Display image
                axes[1].imshow(img_data, cmap='gray', aspect='equal')
                
                # Overlay mask in orange color (like reference image)
                mask_overlay = np.ma.masked_where(mask_data == 0, mask_data)
                axes[1].imshow(mask_overlay, cmap='Oranges', alpha=0.7, aspect='equal')
                
                axes[1].set_title('Mask', fontsize=12, weight='bold')
                axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, 'Mask\nLoad Failed', ha='center', va='center',
                           transform=axes[1].transAxes, fontsize=10)
                axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'Mask\nNot Found', ha='center', va='center',
                       transform=axes[1].transAxes, fontsize=10)
            axes[1].axis('off')
        
        # Remaining columns: Show only Image_1, Image_2, Image_3 from first condition
        first_condition = technique_info['conditions'][0]
        synthetic_files = self.find_synthetic_data_all_variants(technique_key, first_condition, model_type, patient_id, vol_id)
        
        # Create 3 columns for Image_1, Image_2, Image_3
        for variant_idx in range(1, 4):
            col_idx = 1 + variant_idx  # positions 2, 3, 4
            image_key = f'image_{variant_idx}'
            mask_key = f'mask_{variant_idx}'
            
            title = f"Image_{variant_idx}"
            
            if image_key in synthetic_files:
                img_data, _ = self.load_nifti_slice(synthetic_files[image_key], reference_slice_idx)
                
                if img_data is not None:
                    img_data = np.rot90(img_data)
                    axes[col_idx].imshow(img_data, cmap='gray', aspect='equal')
                    
                    # Set consistent axis limits
                    axes[col_idx].set_xlim(0, img_data.shape[1])
                    axes[col_idx].set_ylim(img_data.shape[0], 0)
                    
                    # Add lesion overlay in white/bright color (like reference)
                    if mask_key in synthetic_files:
                        mask_data, _ = self.load_nifti_slice(synthetic_files[mask_key], reference_slice_idx)
                        if mask_data is not None:
                            mask_data = np.rot90(mask_data)
                            lesion_overlay = np.ma.masked_where(mask_data == 0, mask_data)
                            axes[col_idx].imshow(lesion_overlay, cmap='Greys_r', alpha=0.8, aspect='equal')
                    
                    axes[col_idx].set_title(title, fontsize=12, weight='bold')
                    axes[col_idx].axis('off')
                else:
                    axes[col_idx].text(0.5, 0.5, f'{title}\nLoad Failed', ha='center', va='center',
                                     transform=axes[col_idx].transAxes, fontsize=10)
                    axes[col_idx].axis('off')
            else:
                axes[col_idx].text(0.5, 0.5, f'{title}\nNot Found', ha='center', va='center',
                                 transform=axes[col_idx].transAxes, fontsize=10)
                axes[col_idx].axis('off')
        
        plt.tight_layout()
        
        # Save to method-specific folder
        output_path = method_dir / f"{technique_key}_{patient_id}_{vol_id}_{model_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Created {technique_key} visualization: {output_path}")
        return str(output_path)
    
    def create_overview_comparison(self, patient_id, vol_id, model_type="pretrained"):
        """Create overview comparison showing all methods for one case"""
        
        # Create model_type-specific directory structure
        model_vis_dir = self.vis_dir / model_type
        overview_vis_dir = model_vis_dir / "overview_comparisons"
        
        model_vis_dir.mkdir(exist_ok=True)
        overview_vis_dir.mkdir(exist_ok=True)
        
        # Calculate total columns: Normal + Mask + all conditions from all methods
        total_conditions = sum(len(self.techniques[t]['conditions']) for t in self.techniques.keys())
        n_cols = 2 + total_conditions  # Normal + Mask + all synthetic conditions
        
        fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 4))
        if n_cols == 1:
            axes = [axes]
        
        # Load original data and find reference slice
        original_img_path = self.find_original_image(patient_id, vol_id)
        original_mask_path = self.find_original_mask(patient_id, vol_id)
        reference_slice_idx = None
        
        col_idx = 0
        
        # Column 1: Normal
        if original_img_path:
            img_data, slice_idx = self.load_nifti_slice(original_img_path)
            reference_slice_idx = slice_idx
            
            if img_data is not None:
                img_data = np.rot90(img_data)
                axes[col_idx].imshow(img_data, cmap='gray', aspect='equal')
                axes[col_idx].set_title('Normal', fontsize=12, weight='bold')
                axes[col_idx].axis('off')
        else:
            axes[col_idx].text(0.5, 0.5, 'Normal\nNot Found', ha='center', va='center',
                             transform=axes[col_idx].transAxes, fontsize=10)
            axes[col_idx].axis('off')
        
        col_idx += 1
        
        # Column 2: Mask
        if original_img_path and original_mask_path:
            img_data, _ = self.load_nifti_slice(original_img_path, reference_slice_idx)
            mask_data, _ = self.load_nifti_slice(original_mask_path, reference_slice_idx)
            
            if img_data is not None and mask_data is not None:
                img_data = np.rot90(img_data)
                mask_data = np.rot90(mask_data)
                
                axes[col_idx].imshow(img_data, cmap='gray', aspect='equal')
                mask_overlay = np.ma.masked_where(mask_data == 0, mask_data)
                axes[col_idx].imshow(mask_overlay, cmap='Oranges', alpha=0.7, aspect='equal')
                
                axes[col_idx].set_title('Mask', fontsize=12, weight='bold')
                axes[col_idx].axis('off')
        else:
            axes[col_idx].text(0.5, 0.5, 'Mask\nNot Found', ha='center', va='center',
                             transform=axes[col_idx].transAxes, fontsize=10)
            axes[col_idx].axis('off')
        
        col_idx += 1
        
        # Remaining columns: All synthetic conditions from all methods
        image_counter = 1
        for technique_key, technique_info in self.techniques.items():
            for condition in technique_info['conditions']:
                synthetic_files = self.find_synthetic_data(technique_key, condition, model_type, patient_id, vol_id)
                
                # Create title with technique name and condition
                technique_name = technique_info['name']
                condition_short = condition.replace('P_P_prime', 'P→P\'').replace('P_N_prime', 'P→N\'').replace('P_N_double_prime', 'P→N\"').replace('P_P_prime_N_double_prime', 'P→P\'N\"')
                title = f"{technique_name}\n({condition_short})"
                
                if 'image' in synthetic_files:
                    img_data, _ = self.load_nifti_slice(synthetic_files['image'], reference_slice_idx)
                    
                    if img_data is not None:
                        img_data = np.rot90(img_data)
                        im = axes[col_idx].imshow(img_data, cmap='gray', aspect='equal')
                        
                        # Set consistent axis limits for uniform image sizes
                        axes[col_idx].set_xlim(0, img_data.shape[1])
                        axes[col_idx].set_ylim(img_data.shape[0], 0)
                        
                        # Add lesion overlay
                        if 'mask' in synthetic_files:
                            mask_data, _ = self.load_nifti_slice(synthetic_files['mask'], reference_slice_idx)
                            if mask_data is not None:
                                mask_data = np.rot90(mask_data)
                                lesion_overlay = np.ma.masked_where(mask_data == 0, mask_data)
                                axes[col_idx].imshow(lesion_overlay, cmap='Greys_r', alpha=0.8, aspect='equal')
                        
                        axes[col_idx].set_title(title, fontsize=10, weight='bold')
                        axes[col_idx].axis('off')
                    else:
                        axes[col_idx].text(0.5, 0.5, f'{title}\nLoad Failed', ha='center', va='center',
                                         transform=axes[col_idx].transAxes, fontsize=10)
                        axes[col_idx].axis('off')
                else:
                    axes[col_idx].text(0.5, 0.5, f'{title}\nNot Found', ha='center', va='center',
                                     transform=axes[col_idx].transAxes, fontsize=10)
                    axes[col_idx].axis('off')
                
                col_idx += 1
                image_counter += 1
        
        plt.tight_layout()
        
        # Save to overview folder
        output_path = overview_vis_dir / f"overview_{patient_id}_{vol_id}_{model_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Created overview comparison: {output_path}")
        return str(output_path)
    
    def create_multi_case_overview(self, model_type="pretrained", max_cases=5):
        """Create a multi-case overview showing several patients"""
        cases_to_use = self.available_cases[:max_cases]
        
        if not cases_to_use:
            print("❌ No available cases found")
            return None
        
        # Create model_type-specific directory structure
        model_vis_dir = self.vis_dir / model_type
        overview_vis_dir = model_vis_dir / "overview_comparisons"
        
        model_vis_dir.mkdir(exist_ok=True)
        overview_vis_dir.mkdir(exist_ok=True)
        
        # Create figure with cases as rows and methods as columns
        methods = ['Normal'] + [self.techniques[t]['name'] for t in self.techniques.keys()]
        n_rows = len(cases_to_use)
        n_cols = len(methods)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for row_idx, (patient_id, vol_id) in enumerate(cases_to_use):
            for col_idx, method in enumerate(methods):
                ax = axes[row_idx, col_idx]
                
                if method == 'Normal':
                    # Load original image
                    original_img_path = self.find_original_image(patient_id, vol_id)
                    if original_img_path:
                        img_data, _ = self.load_nifti_slice(original_img_path)
                        if img_data is not None:
                            img_data = np.rot90(img_data)
                            ax.imshow(img_data, cmap='gray', aspect='equal')
                        else:
                            ax.text(0.5, 0.5, 'Load\nFailed', ha='center', va='center', 
                                   transform=ax.transAxes, fontsize=8)
                    else:
                        ax.text(0.5, 0.5, 'Not\nFound', ha='center', va='center',
                               transform=ax.transAxes, fontsize=8)
                else:
                    # Find corresponding technique
                    technique_key = None
                    for key, info in self.techniques.items():
                        if info['name'] == method:
                            technique_key = key
                            break
                    
                    if technique_key:
                        # Use first condition for this technique
                        condition = self.techniques[technique_key]['conditions'][0]
                        synthetic_files = self.find_synthetic_data(technique_key, condition, model_type, patient_id, vol_id)
                        
                        if 'image' in synthetic_files:
                            img_data, _ = self.load_nifti_slice(synthetic_files['image'])
                            if img_data is not None:
                                img_data = np.rot90(img_data)
                                ax.imshow(img_data, cmap='gray', aspect='equal')
                                
                                # Add lesion overlay
                                if 'mask' in synthetic_files:
                                    mask_data, _ = self.load_nifti_slice(synthetic_files['mask'])
                                    if mask_data is not None:
                                        mask_data = np.rot90(mask_data)
                                        lesion_overlay = np.ma.masked_where(mask_data == 0, mask_data)
                                        ax.imshow(lesion_overlay, cmap='Reds', alpha=0.5, aspect='equal')
                            else:
                                ax.text(0.5, 0.5, 'Load\nFailed', ha='center', va='center',
                                       transform=ax.transAxes, fontsize=8)
                        else:
                            ax.text(0.5, 0.5, 'Not\nFound', ha='center', va='center',
                                   transform=ax.transAxes, fontsize=8)
                
                # Set titles for first row
                if row_idx == 0:
                    ax.set_title(method, fontsize=10, weight='bold')
                
                # Set case labels for first column
                if col_idx == 0:
                    ax.set_ylabel(f'{patient_id}\n{vol_id}', fontsize=8, weight='bold')
                
                ax.axis('off')
        
        plt.tight_layout()
        
        # Save multi-case overview
        output_path = overview_vis_dir / f"multi_case_overview_{model_type}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ Created multi-case overview: {output_path}")
        return str(output_path)
    
    def generate_all_visualizations(self, model_type="pretrained"):
        """Generate all visualizations"""
        print(f"\n🎨 Starting comprehensive visualization generation for model_type: {model_type}")
        print(f"📊 Processing {len(self.available_cases)} cases")
        
        generated_files = []
        
        # 1. Generate individual method visualizations
        print(f"\n📋 1. Generating individual method visualizations...")
        for case_idx, (patient_id, vol_id) in enumerate(self.available_cases):
            print(f"   Processing case {case_idx + 1}/{len(self.available_cases)}: {patient_id}_{vol_id}")
            
            for technique_key in self.techniques.keys():
                try:
                    output_path = self.create_method_visualization(technique_key, patient_id, vol_id, model_type)
                    generated_files.append(output_path)
                except Exception as e:
                    print(f"❌ Error creating {technique_key} visualization for {patient_id}_{vol_id}: {e}")
        
        # 2. Generate overview comparisons
        print(f"\n📋 2. Generating overview comparisons...")
        for case_idx, (patient_id, vol_id) in enumerate(self.available_cases):
            try:
                output_path = self.create_overview_comparison(patient_id, vol_id, model_type)
                generated_files.append(output_path)
            except Exception as e:
                print(f"❌ Error creating overview for {patient_id}_{vol_id}: {e}")
        
        # 3. Generate multi-case overview
        print(f"\n📋 3. Generating multi-case overview...")
        try:
            output_path = self.create_multi_case_overview(model_type, max_cases=5)
            if output_path:
                generated_files.append(output_path)
        except Exception as e:
            print(f"❌ Error creating multi-case overview: {e}")
        
        # Print summary
        print(f"\n✅ Visualization generation completed!")
        print(f"📁 Generated {len(generated_files)} visualization files")
        print(f"📁 Visualizations saved to: {self.vis_dir}/{model_type}")
        
        return generated_files

def main():
    parser = argparse.ArgumentParser(description="Generate organized LeFusion visualizations")
    parser.add_argument("--model-type", choices=["pretrained", "from_scratch", "both"], default="pretrained",
                        help="Model type to visualize")
    
    args = parser.parse_args()
    
    # Create generator
    generator = OrganizedVisualizationGenerator()
    
    # Generate visualizations for specified model types
    if args.model_type == "both":
        for model_type in ["pretrained", "from_scratch"]:
            print(f"\n{'='*60}")
            print(f"Processing {model_type.upper()} models")
            print(f"{'='*60}")
            generator.generate_all_visualizations(model_type)
    else:
        generator.generate_all_visualizations(args.model_type)

if __name__ == "__main__":
    main()