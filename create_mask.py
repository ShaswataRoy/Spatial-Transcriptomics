from glob import glob
import re
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cv2
from cellpose import io,models
import tifffile as tiff
import dask.array as da
import dask_image.imread
from skimage import measure
import pandas as pd

def bgd_corrected(im_stack):
    bgd = np.median(im_stack, axis=0)
    return bgd

image_ids = os.listdir('raw_data')
# image_id = '230211-Hela-IFNG-16h-4_1'

for image_id in tqdm(image_ids,desc="Directory"):

    model_path = "Cytoplasm/train/models/cellpose_1711324891.380017"
    # model_path = "Nucleus/train/models/cellpose_1711400698.2203813"

    model = models.CellposeModel(gpu=True, 
                                pretrained_model=model_path)

    nucleus = dask_image.imread.imread('raw_data/'+image_id+'/'+image_id+'_mxtiled_corrected_stack_ch0.tif')
    cyto = dask_image.imread.imread('raw_data/'+image_id+'/'+image_id+'_mxtiled_corrected_stack_ch1.tif')
    gbp5 = dask_image.imread.imread('raw_data/'+image_id+'/'+image_id+'_mxtiled_corrected_stack_ch2.tif')

    nucleus_bgd = bgd_corrected(nucleus).compute()
    cyto_bgd = bgd_corrected(cyto).compute()
    gbp5_bgd = bgd_corrected(gbp5).compute()

    cell_folder = "segmented_data/Cytoplasm/"+image_id+"/"

    if os.path.isdir(cell_folder):
        continue
    os.mkdir(cell_folder)

    for indx in tqdm(range(nucleus.shape[0]),desc="File"):
        rgb = np.dstack((nucleus[indx].compute()-nucleus_bgd, 
                        cyto[indx].compute()-cyto_bgd, 
                        gbp5[indx].compute()-gbp5_bgd))
        
        for rgb_indx in range(rgb.shape[2]):
            rgb[:,:,rgb_indx] = cv2.normalize(rgb[:,:,rgb_indx],None, 0,255,cv2.NORM_MINMAX,cv2.CV_8U)

        masks,flows,_ = model.eval(rgb,diameter=300,\
                            flow_threshold=.4,\
                            cellprob_threshold=0,\
                            channels=[2,1])
        
        
        img =Image.fromarray(rgb.astype(np.uint8))
        
        file_name = cell_folder+str(indx).zfill(2)+".tif"
        img.save(file_name)
        io.masks_flows_to_seg(rgb,masks,flows,file_name,diams=300.)
    

# # rsync -a x-shaswata@anvil.rcac.purdue.edu:/home/x-shaswata/scratch/spatial_transcriptomics/Cytoplasm/test cellpose_data