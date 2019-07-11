# this one corresponds to the clustered_dataset_056
# stl10 images -- 32x32
# first dataset with larger clusters
# aglomerative clustering done in the first half of the 50K images
# and 3 nearest images for the single images (from the whole 100k set)

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

def generate_samples(image, anchor, images, scaling, translation, pca_rand, \
                     rot_and_flip, hsv_contrast, hsv_color, nb_samples, directory, im_idx):
    for transf_idx in range(nb_samples):
        (image_reesc, coord_resc) = scaling(image, anchor)
        croped_im = a_crop(image_reesc, coord_resc, translation)
        transf = transforms.Compose([pca_rand,
                                     rot_and_flip,
                                     hsv_contrast,
                                     hsv_color
                                     ])

        image_transf = transf(croped_im)
        image_path = join(directory, str(im_idx).zfill(len(str(nb_samples*len(images)))))
        image_transf.save(image_path + '.png')
        im_idx += 1
    return im_idx


# General variables
name = 'data_57'
root_path = './surrogate_dataset/unlab_dataset_cluster_058/'
path_clusters = root_path + 'original_data/'
path_masks = root_path + 'edges/'

cluster_list = os.listdir(path_clusters)
cluster_list.sort()

nb_clusters = len(cluster_list)   # this will be the number of classes
nb_samples = 112
# I reduced to 112 to reduce the generation and training time... The split will be 90/10
# This correspond to the number of transformations per image


# importing transformations
sys.path.append('./')
from transformations_24 import Rotate_and_flip, \
HSV_contrast_2, HSV_color, PCA_rand_always, Get_coord_from_mask, \
anchor_crop_gauss, Scale_images_and_anchor

# initializing transformations
output_size = 32
max_scale = 1.4
translation = 0.2

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


print ('%d clusters loaded...' %len(cluster_list))
st = time.time()

# Here start a for loop to go through all the classes

for cluster in tqdm(cluster_list):
    images = os.listdir(join(path_clusters, cluster))
    images.sort()
    masks = os.listdir(join(path_masks, cluster))
    masks.sort()

    directory = join(root_path, name, cluster)
    if not os.path.exists(directory):
        os.makedirs(directory)

    im_idx = 0
    for idx, img in enumerate(images):
        image = Image.open(join(path_clusters, cluster, img))
        mask = Image.open(join(path_masks, cluster, masks[idx]))

        anchor = Get_coord_from_mask(mask, 1)
        min_scale = 0.7 # output_size/np.min(image.size)
        scales = (min_scale, max_scale)
        scaling = Scale_images_and_anchor(output_size, scales)

        # here a for to compute all the transformations on a class:

        im_idx = generate_samples(image, anchor, images, scaling, translation, pca_rand,
                         rot_and_flip, hsv_contrast, hsv_color, nb_samples, directory, im_idx)


nd = time.time()
print (str(nd-st) + 'seconds')
print "Done!"
