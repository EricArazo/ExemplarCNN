import os
import numpy as np
from PIL import Image
from skimage import color
from skimage.filters import sobel
from tqdm import tqdm
from os.path import join as join


# Converting the class images into "edge masks"
#out_path = './class_folder_edge/'
#in_path = '../../surrogate_dataset_imagenet/classes_folder/'

out_path = '../unlab_dataset_cluster_058/edges/'
in_path = '../unlab_dataset_cluster_058/original_data/'

clusters = os.listdir(in_path)
clusters.sort()

try:
    os.mkdir(out_path)
except:
    pass

for cluster in tqdm(clusters):
    images = os.listdir(join(in_path, cluster))
    images.sort()
    try:
        os.mkdir(join(out_path, cluster))
    except:
        pass
    for img in images:
        image = np.asarray(Image.open(join(in_path, cluster, img)))
        edge_mask = sobel(color.rgb2gray(image))
        im = Image.fromarray(np.uint8(edge_mask*255))
        im.save(join(out_path, cluster, img[:-4] + '_edge.png'))
