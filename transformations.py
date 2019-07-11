from __future__ import division
import os
import numpy as np
import skimage
from skimage import transform as tf
from skimage import exposure
from torchvision import transforms
from PIL import Image
from skimage import color
from skimage.transform import rotate as pad_rotation
import time
import sys
import cv2

import torch

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter



class Rotate_and_flip(object):
    """Randomply rotates PIL images plus random fliping
    The rotated images are ass well padded with the reflected image"""
    def __init__(self, angle_range):
        self.angle_range = angle_range

    def __call__(self, image):
        angle = np.random.uniform(-self.angle_range, self.angle_range)
        if np.random.uniform(-1, 1) > 0:
            im_np = np.asarray(image)
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

            im_H_flip = im_np[:,::-1,:]
            image = Image.fromarray(im_H_flip)
        if np.random.uniform(-1, 1) < 2:
            im_np = np.asarray(image)
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

            angle = self.angle_range * np.random.uniform(-1,1)
            im_end = pad_rotation(im_np, angle = angle, mode = 'reflect')
            return Image.fromarray(np.uint8(im_end*255))
        return image

class Translate_and_scale(object):
    """Randomply scales a PIL image plus random crop
    """

    def __init__(self, image_size, max_scaling):
        #self.angle_range = angle_range
        self.im_size = image_size
        self.max_scaling = max_scaling

    def __call__(self, image):

        scaled_size = int(self.im_size*np.random.uniform(1, self.max_scaling))

        # randomly define the coordinates to crop the image
        x1 = np.random.randint(0, (scaled_size-self.im_size)+1) # the '+1' is to avoid problems if
                                                                # there is no scaling ... in that case
                                                                # the randint is between 0 and 0. That
                                                                # raises a ValueError
        y1 = np.random.randint(0, (scaled_size-self.im_size)+1)

        if np.random.uniform(-1, 1) < 0.5:
            im_np = np.asarray(image)
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

            im_s = tf.resize(im_np, output_shape = (scaled_size,scaled_size), mode = 'reflect')
            im_crop = im_s[x1:x1+self.im_size, y1:y1+self.im_size]
            return Image.fromarray(np.uint8(im_crop*255))
        return image

class Translate_and_scale_large_images(object):
    """Randomply scales a PIL image plus random crop. Now from random sizes images.
    There is no requirement for the lower scaling, the function picks the higher
    between the given value andthe minimum allowed.
    """

    def __init__(self, out_size, scaling = (1, 2)):
        #self.angle_range = angle_range
        self.im_size = out_size
        self.higher_scaling = scaling[1]
        self.lower_scaling = scaling[0]

    def __call__(self, image):
        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

        shorter_edge = np.min(image.size)
        self.lower_scaling = max(self.lower_scaling, self.im_size/float(shorter_edge))

        scaler = np.random.uniform(self.lower_scaling, self.higher_scaling)
        scaled_size_x = im_np.shape[0]*scaler
        scaled_size_y = im_np.shape[1]*scaler

        # randomly define the coordinates to crop the image
        x1 = np.random.randint(0, (scaled_size_x-self.im_size)+1)
        y1 = np.random.randint(0, (scaled_size_y-self.im_size)+1)

        im_s = tf.resize(im_np, output_shape = (int(im_np.shape[0]*scaler), int(im_np.shape[1]*scaler)), mode = 'reflect')
        im_crop = im_s[x1:x1+self.im_size, y1:y1+self.im_size]
        return Image.fromarray(np.uint8(im_crop*255))


class Exposure_gam_log_sig(object):
    """Randomply applies gamma, logarithmic and sigmoid exposure corection
    -> gamma_corrected = exposure.adjust_gamma(img, 2)
            http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.adjust_gamma
    -> logarithmic_corrected = exposure.adjust_log(img, 1)
            http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.adjust_log
    -> sigmoid_corrected = exposure.adjust_sigmoid(img, 0.5,1) # normal image with cutoff = 0.5 and gaain = 5
            http://scikit-image.org/docs/dev/api/skimage.exposure.html#skimage.exposure.adjust_gamma

    Initial evaluations:

        g_c = skimage.exposure.adjust_gamma(im_np, 0.2) # from 0.2 to 1.8
        l_c = skimage.exposure.adjust_log(im_np, 0.4) #  from 0.4 to 1
        s_c = skimage.exposure.adjust_sigmoid(im_np, cutoff = 0.7, gain = 7) # cutoff from 0.2 to 0.7, gain from 2 to 7

    """
    def __init__(self, gamma = (0.2, 1.8), log = (0.4, 1), sig_cut = (0.2, 0.7), sig_gain = (2, 7)):
        self.gamma = gamma
        self.log = log
        self.sig_cut = sig_cut
        self.sig_gain = sig_gain

    def __call__(self, image):
        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

        if np.random.uniform(-1, 1) > 0.5:
            return image
        if np.random.uniform(-1, 1) > 0:
            im_np = skimage.exposure.adjust_gamma(im_np, np.random.uniform(self.gamma[0],
                                                                            self.gamma[1]))

        if np.random.uniform(-1, 1) > 0:
            im_np = skimage.exposure.adjust_log(im_np, np.random.uniform(self.log[0],
                                                                     self.log[1]))

        if np.random.uniform(-1, 1) > 2:
            im_np = skimage.exposure.adjust_sigmoid(im_np,
                                                  np.random.uniform(self.sig_cut[0],self.sig_cut[1]),
                                                  np.random.uniform(self.sig_gain[0],self.sig_gain[1]))
        return Image.fromarray(np.uint8(im_np))

class Lab_rand(object):
    """Randomply applies transformations to the images in the Lab color space

    First randomly applies a transformation to the L chanel of the Lab space
    (which means modify the luminosity of the image)

    Then, randomly as well, it multiplies each of the color channels (a and b)
    by a number passed as a parrameter.

    Check the constrains in this codeline:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/color/colorconv.py#L979

    """
    def __init__(self, ctL = (1, 2), cta = (-2, 2), ctb = (-2, 2)):
        self.ctL = ctL
        self.cta = cta
        self.ctb = ctb

    def __call__(self, image):
        im_rgb = image.convert(mode = "RGB")
        im_array = np.asarray(im_rgb)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped
        im_lab = color.rgb2lab(im_array)

        if np.random.uniform(-1, 1) > 0:
            L = np.random.uniform(self.ctL[0], self.ctL[1])
            im_lab_0 = im_lab[:,:,0]*L
        else:
            im_lab_0 = im_lab[:,:,0]

        if np.random.uniform(-1, 1) > 0:
            a = np.random.uniform(self.cta[0], self.cta[1])
            b = np.random.uniform(self.ctb[0], self.ctb[1])
            im_lab_1 = im_lab[:,:,1]*a
            im_lab_2 = im_lab[:,:,2]*b
        else:
            im_lab_1 = im_lab[:,:,1]
            im_lab_2 = im_lab[:,:,2]

        im_lab_t = np.stack((im_lab_0, im_lab_1, im_lab_2), axis = 2)
        im_t = color.lab2rgb(im_lab_t)

        # Postprocessing to avoid saturating pixels
        im_end = im_t-im_t.min()
        im_end = im_end/im_end.max()

        return Image.fromarray((im_end*255).astype('uint8'), "RGB")

class HSV_rand(object):
    """Randomply transforms the HSV channels.
    Power transformations for the three channels.
    """
    def __init__(self, h_power = (-0.1, 0.1), s_power = (0.1, 5), v_power = (0.3, 1)):
        self.h_power = h_power
        self.s_power = s_power
        self.v_power = v_power

    def __call__(self, image):
        if np.random.uniform(-1, 1) > 2:
            return image

        else:
            h = np.random.uniform(self.h_power[0], self.h_power[1])
            s = np.random.uniform(self.s_power[0], self.s_power[1])
            v = np.random.uniform(self.v_power[0], self.v_power[1])

            im_np = np.asarray(image)
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

            im_hsv = color.rgb2hsv(im_np)

            im_hsv_t_h = [x+h for x in im_hsv[:,:,0]] # from -0.1 to 0.1
            im_hsv_t_s = [x**s for x in im_hsv[:,:,1]]# from 0 to 5
            im_hsv_t_v = [x**v for x in im_hsv[:,:,2]] # from 0.3 to 1

            im_hsv_t = np.stack((im_hsv_t_h, im_hsv_t_s, im_hsv_t_v), axis = 2)
            im_rgb = color.hsv2rgb(im_hsv_t)

            # Postprocessing to avoid saturating pixels
            im_end = im_rgb-im_rgb.min()
            im_end = im_end/im_end.max()

            return Image.fromarray(np.uint8(im_end*255))

class HSV_contrast_2(object):
    """Randomply transforms the HSV channels.
    Power transformations for the three channels.
    """
    def __init__(self, power = (0.25, 4), factor = (0.7, 1.4), addition = (-0.1, 0.1)):
        self.power = power
        self.factor = factor
        self.addition = addition

    def __call__(self, image):
        power = np.random.uniform(self.power[0], self.power[1])
        factor = np.random.uniform(self.factor[0], self.factor[1])
        addition = np.random.uniform(self.addition[0], self.addition[1])

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped
        im_hsv = color.rgb2hsv(im_np)

        im_hsv_t_h = [x for x in im_hsv[:,:,0]]
        im_hsv_t_s = [(((x**power)*factor) + addition) for x in im_hsv[:,:,1]]
        im_hsv_t_v = [(((x**power)*factor) + addition) for x in im_hsv[:,:,2]]

        im_hsv_t = np.stack((im_hsv_t_h, im_hsv_t_s, im_hsv_t_v), axis = 2)
        im_rgb = color.hsv2rgb(im_hsv_t)

        # Postprocessing to avoid saturating pixels
        im_end = im_rgb-im_rgb.min()
        im_end = im_end/im_end.max()

        return Image.fromarray(np.uint8(im_end*255))

class HSV_color(object):
    """Randomply transforms the HSV channels.
    Power transformations for the three channels.
    """
    def __init__(self,addition = (-0.1, 0.1)):
        self.addition = addition

    def __call__(self, image):
        addition = np.random.uniform(self.addition[0], self.addition[1])

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped
        im_hsv = color.rgb2hsv(im_np)

        im_hsv_t_h = [(x+addition) for x in im_hsv[:,:,0]]
        im_hsv_t_s = [x for x in im_hsv[:,:,1]]
        im_hsv_t_v = [x for x in im_hsv[:,:,2]]

        im_hsv_t = np.stack((im_hsv_t_h, im_hsv_t_s, im_hsv_t_v), axis = 2)
        im_rgb = color.hsv2rgb(im_hsv_t)

        # Postprocessing to avoid saturating pixels
        im_end = im_rgb-im_rgb.min()
        im_end = im_end/im_end.max()

        return Image.fromarray(np.uint8(im_end*255))


class HSV_rand_always(object):
    """Randomply transforms the HSV channels.
    Power transformations for the three channels.
    """
    def __init__(self, h_power = (-0.1, 0.1), s_power = (0.1, 5), v_power = (0.3, 1)):
        self.h_power = h_power
        self.s_power = s_power
        self.v_power = v_power

    def __call__(self, image):
        h = np.random.uniform(self.h_power[0], self.h_power[1])
        s = np.random.uniform(self.s_power[0], self.s_power[1])
        v = np.random.uniform(self.v_power[0], self.v_power[1])

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

        im_hsv = color.rgb2hsv(im_np)

        im_hsv_t_h = [x+h for x in im_hsv[:,:,0]] # from -0.1 to 0.1
        im_hsv_t_s = [x**s for x in im_hsv[:,:,1]]# from 0 to 5
        im_hsv_t_v = [x**v for x in im_hsv[:,:,2]] # from 0.3 to 1

        im_hsv_t = np.stack((im_hsv_t_h, im_hsv_t_s, im_hsv_t_v), axis = 2)
        im_rgb = color.hsv2rgb(im_hsv_t)

        # Postprocessing to avoid saturating pixels
        im_end = im_rgb-im_rgb.min()
        im_end = im_end/im_end.max()

        return Image.fromarray(np.uint8(im_end*255))


# PCA transformation
class PCA(object):
    def __init__(self, n=1):
        self.n = n

    def fit(self, X_input):
        # compute mean of the data and store
        self.mean = X_input.mean(axis=0)

        # subtract the mean
        X_input = X_input - self.mean

        # we want each pixel as a vector of 3 components (RGB)
        X = np.reshape(X_input, (X_input.shape[0]*X_input.shape[1], X_input.shape[2]))    # EAS

        # compute the empirical covariance matrix
        covariance = np.dot(X.T, X)

        # compute eigenvalues and eigenvectors
        vals, vecs = np.linalg.eig(covariance)
        # sort eigenvalues and vectors by eigenvalue
        indices = np.argsort(-vals)
        vals = vals[indices]
        vecs = vecs[:, indices]

        # store eigenvalues and principal components
        self.eigenvalues = vals
        self.components = vecs[:, :self.n]
        return self

    def transform(self, X):
        return np.dot(X - self.mean, self.components)

    def inverse_transform(self, X):
        return np.dot(X, self.components.T) + self.mean


class PCA_rand(object):
    """Randomply applies a transformation in the PCA space.
    """
    def __init__(self, mult = (0.2, 1.2)):
        self.mult = mult

    def __call__(self, image):
        if np.random.uniform(-1, 1) > 0.5:
            return image

        else:
            ch1 = np.random.uniform(self.mult[0], self.mult[1])
            ch2 = np.random.uniform(self.mult[0], self.mult[1])
            ch3 = np.random.uniform(self.mult[0], self.mult[1])

            im_np = np.asarray(image)/255.0
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)


            if len(im_np.shape) < 3:
                print 'PCA not done'
                return image

            # define pca for three dimensions
            pca = PCA(3)

            # fit pca
            pca.fit(im_np)

            # transform image to PCA space
            im_pca = pca.transform(im_np)

            # image transformation
            im_0 = im_pca[:,:,0]*ch1    # each of them multiplied by a number in the reange (0.5 to 2)
            im_1 = im_pca[:,:,1]*ch2
            im_2 = im_pca[:,:,2]*ch3

            im_pca_tf = np.stack((im_0, im_1, im_2), axis = 2)

            # transform back the image
            im_end = pca.inverse_transform(im_pca_tf)
            # Postprocessing to avoid saturating pixels
            im_end = im_end-im_end.min()
            np.seterr(divide='ignore', invalid='ignore')
            im_end = im_end/im_end.max()

            return Image.fromarray(np.uint8(im_end*255))

class PCA_rand_always(object):
    """Randomply applies a transformation in the PCA space.
    """
    def __init__(self, mult = (0.2, 1.2)):
        self.mult = mult

    def __call__(self, image):
        ch1 = np.random.uniform(self.mult[0], self.mult[1])
        ch2 = np.random.uniform(self.mult[0], self.mult[1])
        ch3 = np.random.uniform(self.mult[0], self.mult[1])

        im_np = np.asarray(image)/255.0
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)


        if len(im_np.shape) < 3:
            print 'PCA not done'
            return image

        # define pca for three dimensions
        pca = PCA(3)

        # fit pca
        pca.fit(im_np)

        # transform image to PCA space
        im_pca = pca.transform(im_np)

        # image transformation
        im_0 = im_pca[:,:,0]*ch1    # each of them multiplied by a number in the reange (0.5 to 2)
        im_1 = im_pca[:,:,1]*ch2
        im_2 = im_pca[:,:,2]*ch3

        im_pca_tf = np.stack((im_0, im_1, im_2), axis = 2)

        # transform back the image
        im_end = pca.inverse_transform(im_pca_tf)
        # Postprocessing to avoid saturating pixels
        im_end = im_end-im_end.min()
        np.seterr(divide='ignore', invalid='ignore')
        im_end = im_end/im_end.max()

        return Image.fromarray(np.uint8(im_end*255))

def Get_coord_from_mask(mask, nb_points):
    # normalize the mask
    maks_np = np.asarray(mask)
    #maks_np = maks_np.transpose(1,0)    # when we go from pil to numpy array the W and L dimensions are swaped
    maks_np_norm = maks_np/float(np.sum(maks_np))

    # "vectorize" the mask
    mask_vec = np.ravel(maks_np_norm)

    # cumulative distribution from the vector
    mask_vec_cum = np.cumsum(mask_vec)

    # here we get the index of 30 points... they should be arouund the most salient region...
    indexes = []
    for dot in np.random.rand(nb_points):
        indexes.append(np.argmax(mask_vec_cum>dot))

    # from index to coordinates in the mask
    return np.unravel_index(indexes, maks_np_norm.shape)


def Shi_tomasi_anchor(image):
    image_np = np.asarray(image)
    gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray,1,0.01,10)
    corners = np.int0(corners)
    return corners.ravel()

class Cond_crop(object):
    """Crop an image from the given coordinates.

    """

    def __init__(self, out_size):
        #self.angle_range = angle_range
        self.out_size = out_size

    def __call__(self, image, center_coord = (100, 100)):
        x_coord = center_coord[0]
        y_coord = center_coord[1]

        if image.size[0] < self.out_size:
            factor =  self.out_size/image.size[0]
            scale = (factor, factor)
            scaling = Scale_images_and_coord_always(self.out_size, scale)
            (image, center_coord) = scaling(image, center_coord)
            x_coord = center_coord[0]
            y_coord = center_coord[1]

        if image.size[1] < self.out_size:
            factor =  self.out_size/image.size[1]
            scale = (factor, factor)
            scaling = Scale_images_and_coord_always(self.out_size, scale)
            (image, center_coord) = scaling(image, center_coord)
            x_coord = center_coord[0]
            y_coord = center_coord[1]

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

        # The following if and elses are to avoid croping out of the original image.
        if (x_coord > (im_np.shape[0] - self.out_size/2)):# or (x_coord < (self.out_size/2)):
            x_coord = round(im_np.shape[0] - self.out_size/2)
        elif x_coord < (self.out_size/2):
            x_coord = round(self.out_size/2)
        else:
            x_coord = round(x_coord)

        if (y_coord > (im_np.shape[1] - self.out_size/2)):# or (y_coord < (self.out_size/2)):
            y_coord = round(im_np.shape[1] - self.out_size/2)
        elif y_coord < (self.out_size/2):
            y_coord = round(self.out_size/2)
        else:
            y_coord = round(y_coord)

        half_out = round(self.out_size/2)

        im_crop = im_np[x_coord-half_out:(x_coord-half_out) + self.out_size,
                        y_coord-half_out:(y_coord-half_out) + self.out_size,:]

        return Image.fromarray(np.uint8(im_crop))

class Scale_images_and_coord(object):
    """Randomply scales a PIL image and its asociated coordinates.

    """

    def __init__(self, out_size, scaling = (0.5, 2)):
        #self.angle_range = angle_range
        self.out_size = out_size
        self.max_scaling = scaling[1]
        self.min_scaling = scaling[0]

    def __call__(self, image, coord):
        if np.random.uniform(-1, 1) < 0.5:
            scaling_factor = np.random.uniform(self.min_scaling, self.max_scaling)

            scaled_size_x = round(image.size[0]*scaling_factor)
            scaled_size_y = round(image.size[1]*scaling_factor)
            coord_0_resized = round(coord[0]*scaling_factor)
            coord_1_resized = round(coord[1]*scaling_factor)


            if (scaled_size_x < self.out_size) or (scaled_size_y < self.out_size):
                scaling_factor = self.out_size/float(np.min(image.size))
                scaled_size_x = round(image.size[0]*scaling_factor)
                scaled_size_y = round(image.size[1]*scaling_factor)
                coord_0_resized = round(coord[0]*scaling_factor)
                coord_1_resized = round(coord[1]*scaling_factor)

            im_np = np.asarray(image)
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped
            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis],
                                        im_np[:,:, np.newaxis],
                                        im_np[:,:, np.newaxis]), axis = 2)

            im_s = tf.resize(im_np, output_shape = (scaled_size_y, scaled_size_x), mode = 'reflect')
        else:
            return image, coord

        return Image.fromarray(np.uint8(im_s*255)), (coord_0_resized, coord_1_resized)
class Scale_images_and_coord_always(object):
    """Randomply scales a PIL image and its asociated coordinates.

    """

    def __init__(self, out_size, scaling = (0.5, 2)):
        #self.angle_range = angle_range
        self.out_size = out_size
        self.max_scaling = scaling[1]
        self.min_scaling = scaling[0]

    def __call__(self, image, coord):
        if np.random.uniform(-1, 1) < 2:
            scaling_factor = np.random.uniform(self.min_scaling, self.max_scaling)

            scaled_size_x = round(image.size[0]*scaling_factor)
            scaled_size_y = round(image.size[1]*scaling_factor)
            coord_0_resized = round(coord[0]*scaling_factor)
            coord_1_resized = round(coord[1]*scaling_factor)


            if (scaled_size_x < self.out_size) or (scaled_size_y < self.out_size):
                scaling_factor = self.out_size/float(np.min(image.size))
                scaled_size_x = round(image.size[0]*scaling_factor)
                scaled_size_y = round(image.size[1]*scaling_factor)
                coord_0_resized = round(coord[0]*scaling_factor)
                coord_1_resized = round(coord[1]*scaling_factor)

            im_np = np.asarray(image)
            #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped
            if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
                im_np = np.concatenate((im_np[:,:, np.newaxis],
                                        im_np[:,:, np.newaxis],
                                        im_np[:,:, np.newaxis]), axis = 2)

            im_s = tf.resize(im_np, output_shape = (scaled_size_y, scaled_size_x), mode = 'reflect')
        else:
            return image, coord

        return Image.fromarray(np.uint8(im_s*255)), (coord_0_resized, coord_1_resized)

class Scale_images_and_anchor(object):
    """Randomly scales a PIL image and its asociated anchor.

    The scalling factor passed to the function should be:
    f > X/S
    where X is the output size and S the input size.

    In the case that the S < X we scale the image to the
    minimum and then apply a random zoom

    """

    def __init__(self, out_size, scaling = (0.5, 2)):
        #self.angle_range = angle_range
        self.out_size = out_size
        self.max_scaling = scaling[1]
        self.min_scaling = scaling[0]

    def __call__(self, image, anchor):
        if (image.size[0] < self.out_size) or (image.size[1] < self.out_size) or (self.min_scaling > 1):
            scaling_factor_correct = self.out_size/np.min(image.size)
            scaling_factor = np.random.uniform(scaling_factor_correct, self.max_scaling*scaling_factor_correct)
        else:
            scaling_factor = np.random.uniform(self.min_scaling, self.max_scaling)

        scaled_size_x = round(image.size[0]*scaling_factor)
        scaled_size_y = round(image.size[1]*scaling_factor)
        coord_0_resized = round(anchor[0]*scaling_factor)
        coord_1_resized = round(anchor[1]*scaling_factor)

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped
        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis],
                                    im_np[:,:, np.newaxis],
                                    im_np[:,:, np.newaxis]), axis = 2)

        im_s = tf.resize(im_np, output_shape = (scaled_size_y, scaled_size_x), mode = 'reflect')
        return Image.fromarray(np.uint8(im_s*255)), (coord_0_resized, coord_1_resized)


def sampling_anchor_uniform(in_size, out_size, anchor, translation):
    # helper to generate the cropping coordinateds from an anchor
    # It checkes that the box does not fall out of the image

    in_x = in_size[0]
    in_y = in_size[1]

    # Sampling around the anchor
    x_coord = anchor[0] + np.random.uniform(-translation*out_size, translation*out_size)
    y_coord = anchor[1] + np.random.uniform(-translation*out_size, translation*out_size)

    if (x_coord > (in_x - out_size/2)):
        x_coord = round(in_x - out_size/2)
    elif x_coord < (out_size/2):
        x_coord = round(out_size/2)
    else:
        x_coord = round(x_coord)

    if (y_coord > (in_y - out_size/2)):
        y_coord = round(in_y - out_size/2)
    elif y_coord < (out_size/2):
        y_coord = round(out_size/2)
    else:
        y_coord = round(y_coord)

    return (int(x_coord), int(y_coord))

def sampling_anchor_gauss(in_size, out_size, anchor, translation):
    # helper to generate the cropping coordinateds from an anchor
    # It checkes that the box does not fall out of the image

    in_x = in_size[0]
    in_y = in_size[1]

    # Sampling around the anchor
    x_coord = anchor[0]+np.random.normal(0, translation*out_size)
    y_coord = anchor[1]+np.random.normal(0, translation*out_size)

    if (x_coord > (in_x - out_size/2)):
        x_coord = round(in_x - out_size/2)
    elif x_coord < (out_size/2):
        x_coord = round(out_size/2)
    else:
        x_coord = round(x_coord)

    if (y_coord > (in_y - out_size/2)):
        y_coord = round(in_y - out_size/2)
    elif y_coord < (out_size/2):
        y_coord = round(out_size/2)
    else:
        y_coord = round(y_coord)

    return (int(x_coord), int(y_coord))

class anchor_crop_uniform(object):
    """
    Return a crop around a given anchor

    """

    def __init__(self, out_size):
        #self.angle_range = angle_range
        self.out_size = out_size

    def __call__(self, image, anchor = (100, 100), translation = 0.2):

        (x_coord, y_coord) = sampling_anchor_uniform(image.size, self.out_size, anchor, translation)

        (x_coord, y_coord) = (y_coord, x_coord) # when we go from pil to numpy array the W and L dimensions are swaped

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

        half_out = int(self.out_size/2)

        im_crop = im_np[x_coord-half_out:(x_coord-half_out) + self.out_size,
                        y_coord-half_out:(y_coord-half_out) + self.out_size,:]
        return Image.fromarray(np.uint8(im_crop))

class anchor_crop_gauss(object):
    """
    Return a crop around a given anchor

    """

    def __init__(self, out_size):
        #self.angle_range = angle_range
        self.out_size = out_size

    def __call__(self, image, anchor = (100, 100), translation = 0.2):

        (x_coord, y_coord) = sampling_anchor_gauss(image.size, self.out_size, anchor, translation)

        (x_coord, y_coord) = (y_coord, x_coord) # when we go from pil to numpy array the W and L dimensions are swaped

        im_np = np.asarray(image)
        #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

        if len(im_np.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
            im_np = np.concatenate((im_np[:,:, np.newaxis], im_np[:,:, np.newaxis], im_np[:,:, np.newaxis]), axis = 2)

        half_out = int(self.out_size/2)

        im_crop = im_np[x_coord-half_out:(x_coord-half_out) + self.out_size,
                        y_coord-half_out:(y_coord-half_out) + self.out_size,:]


        return Image.fromarray(np.uint8(im_crop))

def elastic_transform(image, alpha_param, sigma_param, alpha_affine_param, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    """
    image = np.asarray(image)
    #im_np = im_np.transpose(1,0,2)    # when we go from pil to numpy array the W and L dimensions are swaped

    if len(image.shape) < 3: # if there is less than 3 channels (black&white) we triplicate the image
        image = np.concatenate((image[:,:, np.newaxis], image[:,:, np.newaxis], image[:,:, np.newaxis]), axis = 2)

    alpha = image.shape[1] * alpha_param
    sigma = image.shape[1] * sigma_param
    alpha_affine = image.shape[1] * alpha_affine_param

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    im_elastic = map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

    return Image.fromarray(np.uint8(im_elastic))

class Cutout(object):
    """

    From here: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - int(self.length / 2), 0, h)
            y2 = np.clip(y + int(self.length / 2), 0, h)
            x1 = np.clip(x - int(self.length / 2), 0, w)
            x2 = np.clip(x + int(self.length / 2), 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
