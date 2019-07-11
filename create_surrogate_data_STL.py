# this one corresponds to the dataset035
# stl10 images -- 32x32
# samples from gaussian (adding the value)
# anchor conditioned by edges

# Same as before but introducing a correctio in the anchor coordinates
# Another correction introduced: gausian for the translation based on the output size, not on the input

from __future__ import division
import os
from os.path import join as join
import numpy as np

from torchvision import transforms
from PIL import Image
from skimage import color
from skimage.transform import rotate as pad_rotation
import time
import sys

from tqdm import tqdm

np.random.seed(42)

# General variables
root_path = './surrogate_dataset/'
classes_path = join(root_path, 'classes_folder_16000_set')
mask_path = join(root_path, 'class_folder_16000_set_edge') # mask_folder
nb_classes = 16000   # this will be the number of classes

#nb_samples = 125   # This correspond to the number of transformations
# I reduced to 112 to reduce the generation and training time... The split will be 90/10
nb_samples = 112


# importing transformations
sys.path.append('./')
from transformations import Rotate_and_flip, \
HSV_contrast_2, HSV_color, PCA_rand_always, Get_coord_from_mask, \
anchor_crop_gauss, Scale_images_and_anchor

# initializing transformations
output_size = 32
max_scale = 1.4 # measure for STL10
translation = 0.2 # 0.2 # measure for STL10

pca_mult = (0.5, 2)
max_rot = 20

hsv_power = (0.25, 4)
hsv_factor = (0.7, 1.4)
hsv_addition = (-0.1, 0.1)

hsv_add_color = (-0.1, 0.1)

# initializing conditional cropping
a_crop = anchor_crop_gauss(output_size)

# initializing other transformations
pca_rand = PCA_rand_always(pca_mult)
rot_and_flip = Rotate_and_flip(max_rot)
hsv_contrast = HSV_contrast_2(hsv_power, hsv_factor, hsv_addition)
hsv_color = HSV_color(hsv_add_color)

# create the samples...
labels = os.listdir(classes_path)
labels.sort()

print ('%d classes loaded...' %len(labels))
st = time.time()

# Here start a for loop to go through all the classes
for label_ind in tqdm(labels):
    directory = join(root_path, 'data_35', str(label_ind).split('.')[0].zfill(len(str(nb_classes))))
    if not os.path.exists(directory):
        os.makedirs(directory)
    image = Image.open(classes_path + '/' + str(label_ind).zfill(len(str(nb_classes))))
    mask = Image.open(mask_path + '/' + str(label_ind).zfill(len(str(nb_classes)))[:-4] + '_edge.png')

    anchor = Get_coord_from_mask(mask, 1)
    min_scale = 0.7#output_size/np.min(image.size) # 0.7 # measure for STL10 images (96x96)
    scales = (min_scale, max_scale)
    scaling = Scale_images_and_anchor(output_size, scales)

    # here a for to compute all the transformations on a class:
    for transf_idx in range(nb_samples):
        (image_reesc, coord_resc) = scaling(image, anchor)

        croped_im = a_crop(image_reesc, coord_resc, translation)
        transf = transforms.Compose([pca_rand,
                                     rot_and_flip,
                                     hsv_contrast,
                                     hsv_color
                                     ])

        image_transf = transf(croped_im)
        image_path = join(directory, str(transf_idx).zfill(len(str(nb_samples))))
        image_transf.save(image_path + '.png')
    # finish the for loop that goes through the transformations
# finish the loop that goes through the images
nd = time.time()
print (str(nd-st) + 'seconds')
print "Done!"
