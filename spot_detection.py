import os
import numpy as np
from glob import glob
import tifffile as tiff
from natsort import natsorted
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import trackpy as tp
from skimage.segmentation import find_boundaries

# Define root directories for data
root_dir = "segmented_data"
exp_id = "221218-Hela-IFNG-16h-2_1"
spot_dir = "raw_data"
image_dir = "raw_data"

def valid_masks(mask_path):
    """
    Checks if the mask is valid or not. A mask is invalid if it touches the boundary of the image or if it touches another mask.

    :param mask_path: Path to the mask file
    :return: Valid masks and their labels
    """
    # Load the mask file
    mask = np.load(mask_path, allow_pickle=True).item()
    mask_image = mask['masks']
    masks = [mask_image == label for label in range(mask_image.max() + 1)]
    mask_boundaries = [find_boundaries(mask_image == label) for label in range(1, mask_image.max() + 1)]

    # Define the boundary of the frame
    region_boundary = np.zeros(mask_image.shape)
    region_boundary[0, :mask_image.shape[1]] = 1
    region_boundary[:mask_image.shape[0], mask_image.shape[1] - 1] = 1
    region_boundary[:mask_image.shape[0], 0] = 1
    region_boundary[mask_image.shape[0] - 1, :mask_image.shape[1]] = 1

    # Identify masks touching each other
    touching_mask_indices = set()
    for i in range(1, mask_image.max() + 1):
        for j in range(i + 1, mask_image.max() + 1):
            if np.any(np.logical_and(mask_boundaries[i - 1], mask_boundaries[j - 1])):
                touching_mask_indices.add(i)
                touching_mask_indices.add(j)

    # Identify masks touching the boundary
    touching_boundary_indices = set()
    for i in range(1, mask_image.max() + 1):
        if np.any(np.logical_and(mask_boundaries[i - 1], region_boundary)):
            touching_boundary_indices.add(i)

    # Determine valid and invalid masks
    all_indices = set(np.arange(1, mask_image.max()))
    invalid_indices = touching_boundary_indices.union(touching_mask_indices)
    valid_indices = all_indices.difference(invalid_indices)

    if len(valid_indices) == 0:
        return None, None
    valid_mask_labels = np.array([masks[i] for i in valid_indices])
    labels = np.arange(1, len(valid_mask_labels) + 1)
    valid_masks = np.einsum("i,ijk->jk", labels, valid_mask_labels)

    return valid_masks, valid_mask_labels

def find_spots(img, cyto_mask, nuc_mask, minmass_thresh=200, signal_thresh=15):
    """
    Finds the spots in the image that are within the cytoplasm and nucleus.

    :param img: Image to find the spots in
    :param cyto_mask: Cytoplasm mask
    :param nuc_mask: Nucleus mask
    :param minmass_thresh: Minimum mass threshold for spot detection
    :param signal_thresh: Signal threshold for spot detection
    :return: Counts of spots (RNAs) in cytoplasm and nucleus
    """
    # Detect spots with radius < 19px
    f = tp.locate(img, 19, invert=False, minmass=minmass_thresh)
    f = f[f.signal > signal_thresh]
    spots = f[['x', 'y']].to_numpy()

    # Initialize counts
    cyto_counts = np.zeros(cyto_mask.max() + 1)
    nuc_counts = np.zeros(nuc_mask.max() + 1)

    # Count spots in cytoplasm and nucleus
    for spot in np.round(spots).astype(int):
        if nuc_mask[spot[1], spot[0]] > 0:
            nuc_counts[nuc_mask[spot[1], spot[0]]] += 1
        elif cyto_mask[spot[1], spot[0]] > 0:
            cyto_counts[cyto_mask[spot[1], spot[0]]] += 1

    return list(cyto_counts[1:]), list(nuc_counts[1:])

def nuc_containing_cyto(cyto_mask_labels, nuc_mask_labels, nuc_mask):
    """
    Only considers the nucleus masks that are completely inside the cytoplasm mask.

    :param cyto_mask_labels: Cytoplasm mask labels
    :param nuc_mask_labels: Nucleus mask labels
    :param nuc_mask: Nucleus mask
    :return: Cytoplasm masks, nucleus masks, cytoplasm mask labels, nucleus mask labels
    """
    # Identify cytoplasm and nucleus masks containing each other
    cyto_containing_cond = np.einsum("ijk,jk->i", cyto_mask_labels, nuc_mask) > 0
    nuc_containing_cond = np.einsum("ijk,jk->i", nuc_mask_labels, cyto_mask) > 0

    # Filter masks based on conditions
    cyto_mask_labels = cyto_mask_labels[cyto_containing_cond, ...]
    cyto_labels = np.arange(1, len(cyto_mask_labels) + 1)
    cyto_masks = np.einsum("i,ijk->jk", cyto_labels, cyto_mask_labels)

    nuc_mask_labels = nuc_mask_labels[nuc_containing_cond, ...]
    nuc_labels = np.arange(1, len(nuc_mask_labels) + 1)
    nuc_masks = np.einsum("i,ijk->jk", nuc_labels, nuc_mask_labels)

    return cyto_masks, nuc_masks, cyto_mask_labels, nuc_mask_labels

if __name__ == "__main__":
    # Get experiment IDs from the spot directory
    exp_ids = [os.path.basename(path) for path in glob(spot_dir + "/*")]

    for exp_id in tqdm(exp_ids):
        # Define paths for cytoplasm, nucleus, and image data
        cyto_path = os.path.join(root_dir, "Cytoplasm", exp_id)
        nuc_path = os.path.join(root_dir, "Nucleus", exp_id)
        image_path = os.path.join(image_dir, exp_id, exp_id + "_mxtiled_corrected_stack_ch2.tif")
        cyto_seg_file_stack = natsorted(glob(cyto_path + "/*.npy"))
        nuc_seg_file_stack = natsorted(glob(nuc_path + "/*.npy"))

        # Load the image stack
        image_stack = tiff.imread(image_path)

        # Spot detection threshold
        threshold = 200

        # Initialize counts
        cyto_counts = []
        nuc_counts = []

        # Process each image in the stack
        for image_indx in tqdm(range(100), leave=False):
            try:
                img = image_stack[image_indx]

                # Load cytoplasm and nucleus masks
                cyto_mask_path = cyto_seg_file_stack[image_indx]
                nuc_mask_path = nuc_seg_file_stack[image_indx]

                cyto_mask, cyto_mask_labels = valid_masks(cyto_mask_path)
                nuc_mask, nuc_mask_labels = valid_masks(nuc_mask_path)

                if (cyto_mask is not None) and (nuc_mask is not None):
                    # Filter masks and detect spots
                    cyto_mask, nuc_mask, cyto_mask_labels, nuc_mask_labels = nuc_containing_cyto(
                        cyto_mask_labels, nuc_mask_labels, nuc_mask)
                    cyto_count, nuc_count = find_spots(img, cyto_mask, nuc_mask, threshold)
                    cyto_counts += cyto_count
                    nuc_counts += nuc_count
            except:
                # Skip errors and continue processing
                continue

        # Save the counts to files
        np.save("Counts/" + exp_id + "_cyto.npy", np.array(cyto_counts))
        np.save("Counts/" + exp_id + "_nuc.npy", np.array(nuc_counts))