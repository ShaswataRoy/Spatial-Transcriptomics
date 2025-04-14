#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create the Mask for a Specific Experiment

Author: Shaswata Roy
Email: roy134@purdue.edu
Date: 2024/03/28

This script converts all the `_seg.npy` files (used for Cellpose annotation) in a specific experiment folder into a single TIFF file.
The directory structure is assumed to be:
    `root_folder/section_folder/experiment_folder`
where `section_folder` is either Cytoplasm or Nucleus, and `experiment_folder` refers to the experiment instance.

Example:
For the directory structure `./Cytoplasm/221218-Hela-IFNG-16h-2_1`, run the following command:
    `python create_mask.py --root ./ --section Cytoplasm --exp 221218-Hela-IFNG-16h-2_1`
The mask will be saved in the root directory as:
    `masks/221218-Hela-IFNG-16h-2_1_Cytoplasm_mask.tif`
"""

from glob import glob
import numpy as np
from tqdm import tqdm
import tifffile as tiff
import cv2
import argparse
import os
from natsort import natsorted

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate a mask from Cellpose segmentation files.")
    parser.add_argument("-r", "--root", help="Root directory", type=str, required=True)
    parser.add_argument("-e", "--exp", help="Experiment name to save the mask", type=str, required=True)
    parser.add_argument("-s", "--section", help="Cell section chosen (Cytoplasm or Nucleus)", type=str, required=True)
    parser.add_argument("-i", "--image_size", help="Image size (default: 1844)", default=1844, type=int)
    args = parser.parse_args()

    # Validate the root directory
    if not os.path.exists(args.root):
        print("Root directory not found.")
        exit()

    root_folder = args.root

    # Validate the section folder (Cytoplasm or Nucleus)
    try:
        section_folder = os.path.join(args.root, args.section)
    except:
        print("Cell section not found. Choose either Cytoplasm or Nucleus.")
        exit()

    # Validate the experiment folder
    try:
        image_folder = os.path.join(section_folder, args.exp)
    except:
        print("Chosen experiment not found.")
        exit()

    # Get all segmentation files for the experiment
    seg_files = natsorted(glob(image_folder + '/*.npy'))
    n_files = len(seg_files)

    # Initialize an empty mask array
    mask = np.zeros((n_files, args.image_size, args.image_size)).astype(np.uint8)

    # Process each segmentation file
    for i in tqdm(range(n_files), desc="Processing segmentation files"):
        # Load the segmentation file and extract the mask
        mask[i] = np.load(seg_files[i], allow_pickle=True).item()['masks'].astype(np.uint8)
        # Normalize the mask to the range [0, 255]
        mask[i] = cv2.normalize(mask[i], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Create the output folder for masks if it doesn't exist
    mask_folder = os.path.join(root_folder, 'masks')
    if not os.path.isdir(mask_folder):
        os.mkdir(mask_folder)

    # Save the mask as a TIFF file
    output_path = os.path.join(mask_folder, f"{args.exp}_{args.section}_mask.tif")
    tiff.imwrite(output_path, mask)
    print(f"Mask saved to {output_path}")