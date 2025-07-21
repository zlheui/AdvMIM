import os
import SimpleITK as sitk
import numpy as np
import pickle
from scipy.ndimage.interpolation import zoom

import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, inv_ornt_aff, apply_orientation, aff2axcodes


def load_nii_file(nii_image):
    image = sitk.ReadImage(nii_image)
    image_array = sitk.GetArrayFromImage(image)
    return image_array


# cwd = '/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT2MR_image'
# cwd = '/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT2MR_label'
cwd = '/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/MRI/Training/MRI_3d'
count = 0
for file in os.listdir(cwd):
        
    if '.nii.gz' not in file:
        continue

    image = nib.load(os.path.join(cwd, file))

    image = image.get_fdata().astype(np.float32)
    image = np.transpose(image, (2, 0, 1))


    if cwd in ['/media/mount/zhulei/FLARE-MedFM/FLARE-Task3-DomainAdaption/CT/CT2MR_label']:
        image[image==128] = 0
        image[image==129] = 0
        image[image==130] = 0
        image[image==131] = 0
        image[image==139] = 0
        image[image==141] = 0
        image[image==137] = 0
        image[image==14] = 0

        if np.max(image) > 13 or np.min(image) < 0:
            print('error: ', file, np.max(image), np.min(image))
            print(np.unique(image))
            exit(0)
    
    
    # print(image.shape)
    # exit(0)    
    for idx in range(0, image.shape[0]):
        image_slice = image[idx,:,:]
        slice_no = "{:04d}".format(idx)

        # count_class = []
        # for cid in range(0, 14):
        #     count_class.append(np.sum(image_slice==cid))

        # count_class = np.array(count_class)
        # print(count_class / np.sum(count_class))

        x, y = image_slice.shape

        image_slice = zoom(image_slice, (256 / x, 256 / y), order=0)

        # save by slice
        if not os.path.exists(os.path.join(cwd, 'slice')):
            os.makedirs(os.path.join(cwd, 'slice'))
        np.savez(os.path.join(cwd, 'slice', file.replace('.nii.gz', '') + '_slice_' + slice_no), image=image_slice)
        # exit(0)
    print('process ', count)
    count += 1

  