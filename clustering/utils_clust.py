from __future__ import print_function
import numpy as np
import h5py
import matplotlib.pyplot as plt
from sklearn import preprocessing
from PIL import Image

def normalizing_samples_L2(samples):
    # normalizing the features
    #samples_norm = np.linalg.norm(samples, axis=1)
    #samples_L2 = np.divide(samples,samples_norm[:,None])    # this None is to divide each raw from the 
                                                            # matrix by one element from the vector
        
    samples_L2 = preprocessing.normalize(samples, norm = 'l2')
    return samples_L2
  
def loading_images(features_path):
    # loading images
    f_h5py = h5py.File(features_path, 'r')
    nb_images = len(f_h5py['features'])
    features_dim = len(f_h5py['features'][0])
    print ("Dimension of the features:", end = '')
    print (features_dim)
    print ("Number of samples:", end = '')
    print (nb_images)

    samples = np.zeros((nb_images, features_dim))
    for num, sample in enumerate(f_h5py['features']):
        samples[num] = sample
        if num == nb_images-1:
            break
    f_h5py.close()
    
    return samples
    
    
def searching_similar_images(query_number, number_of_images, samples, images_path, plotting):
    images_path = images_path
    
    length_nb_images = len(str(len(samples)))

    feats_im = samples[query_number]
    distances = np.dot(samples, feats_im)
    abs_dist = np.absolute(distances)
    args = abs_dist.argsort()
    args_inv = args[::-1]   # this way the indices corresponding to 
                            # images closer tothe query are at the beginning of the array
    
    if plotting:
        # plotting
        plt.figure(figsize=(3,3))
        plt.title('Query:')
        image = Image.open(images_path + str(args_inv[0]).zfill(length_nb_images) + '.png')
        plt.imshow(np.asarray(image))
        plt.axis('off')
        plt.show()
        plt.figure(figsize=(15,5) )

        for num, im in enumerate(args_inv[1:number_of_images + 1]):
            image = Image.open(images_path + str(im).zfill(length_nb_images) + '.png')
            #plt.subplot(int(number_of_images/5),int(number_of_images/2), num+1)
            plt.subplot(2,5, num+1)
            plt.imshow(np.asarray(image))
            plt.title('Image ' + str(im) + '\nSimilarity with query: ' + str(round(distances[im],3)))
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    return args_inv[1:number_of_images + 1]    