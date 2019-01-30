
# Pre-processing the images for SSD object detection
import tensorflow as tf
from enum import Enum, IntEnum
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from configs.config_common import *
slim = tf.contrib.slim



class Preprocessing(object):

    def __init__(self):
        # RGB mean values:
        self.rgb_mean = [123., 117., 104.]


    # This function preprocesses the input image.
    # Inputs:
    #     image: A Tensor representing the image 
    #     labels: labels of bounding boxes
    #     bboxes: bounding boxes
    #     mode: training / testing
    # Outputs:
    #     the preprocessed image
    def preprocess(self, image, labels, bboxes, mode):
        if image.get_shape().ndims != 3:
            raise ValueError('Input image must have 3 channels.')
        if mode=='train':
            return self.preprocess_train(image, labels, bboxes)
        else:
            return self.preprocess_test(image, labels, bboxes)

        

    # This function preprocesses the input image for training.
    # Inputs:
    #     image: input image
    #     labels: labels of bounding boxes
    #     bboxes: bounding boxes
    # Outputs:
    #     preprocessed image, labels, bounding boxes
    def preprocess_train(self, image, labels, bboxes):
        # convert image to float & scaled to [0, 1].
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        # Generate a single randomly distorted bounding box for an image.
        image, labels, bboxes = self.generate_bounding_box_for_image(image, labels, bboxes)
        # filterout the overlapped bounding boxes 
        labels, bboxes = self.filterout_overlapped_bboxes(labels, bboxes)
        # resize the image to 300*300 using bilinear interpolation.
        image = self.image_resize(image, (FLAGS.image_size, FLAGS.image_size), method=tf.image.ResizeMethod.BILINEAR)
        # Randomly flip the image horizontally.
        image, bboxes = self.flip_image_bounding_boxes(image, bboxes)
        # Randomly distort the colors.
        image = self.image_augmentation(image, lambda x, ordering: self.color_augmentation(x, ordering))
        # scale to [0, 255]
        image = image * 255.
        # subtract mean of image
        image = image - tf.constant(self.rgb_mean, dtype=image.dtype)
        return image, labels, bboxes, None



    # This function preprocesses the input image for testing.
    # Inputs:
    #     image: input image
    #     labels: labels of bounding boxes
    #     bboxes: bounding boxes
    # Outputs:
    #     preprocessed image, labels, bounding boxes
    def preprocess_test(self, image, labels, bboxes):
        # convert image to float
        image = tf.to_float(image)
        # subtract mean of image
        image = image - tf.constant(self.rgb_mean, dtype=image.dtype)
        # define image rectangle corners (scaled values [0,0,1,1]) 
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        # Add image rectangle corners to bboxes.
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)
        # resize the image to 300*300 using bilinear interpolation.
        image = self.image_resize(image, (FLAGS.image_size, FLAGS.image_size), method=tf.image.ResizeMethod.BILINEAR)
        return image, labels, bboxes[1:], bboxes[0]

    

    # This function resizes the image to 300*300 using interpolation
    # Inputs:
    #     image: input image 
    #     size: desired size
    #     method: interpolation method for resizing
    # Outputs:
    #     resized image
    def image_resize(self, image, size, method):
        image = tf.expand_dims(image, 0)
        image = tf.image.resize_images(image, size, method, False)
        image = tf.reshape(image, tf.stack([size[0], size[1], 3]))
        return image



    # This function performs the image augmentation
    # Inputs:
    #     x: input Tensor.
    #     func: Python function to apply.
    #     num_cases: Python int32, number of cases to sample sel from.
    # Outputs:
    #     The result of func(x, sel).
    def image_augmentation(self, img, func):
        rand_uniform = tf.random_uniform([], maxval=4, dtype=tf.int32)
        return control_flow_ops.merge([func(control_flow_ops.switch(img, tf.equal(rand_uniform, case))[1], case)
                for case in range(4)])[0]



    # This function performs 4 color augmentations: hue, saturation, contrast, brightness
    # Inputs:
    #     image: image in [0, 1].
    #     aug_color_ordering: order of augmentation 
    # Outputs:
    #     color-distorted image on range [0, 1]
    def color_augmentation(self, image, aug_color_order=0, scope=None):
        def brightness(image):  return tf.image.random_brightness(image, max_delta=32. / 255.)
        def saturation(image):  return tf.image.random_saturation(image, lower=0.5, upper=1.5)
        def hue(image):         return tf.image.random_hue(image, max_delta=0.2)
        def contrast(image):    return tf.image.random_contrast(image, lower=0.5, upper=1.5)
        if aug_color_order == 0:   image = contrast(hue(saturation(brightness(image))))                
        elif aug_color_order == 1: image = hue(contrast(brightness(saturation(image))))                
        elif aug_color_order == 2: image = saturation(brightness(hue(contrast(image))))                
        elif aug_color_order == 3: image = brightness(contrast(saturation(hue(image))))                
        return tf.clip_by_value(image, 0.0, 1.0)



    # This function generates a single randomly distorted bounding box for an image.
    # Inputs:
    #     image: input image 
    #     labels: labels of bounding boxes 
    #     bbox: bounding boxes
    # Outputs:
    #     cropped_image and the distorted bbox
    def generate_bounding_box_for_image(self, image, labels, bboxes):
        bbox_begin, bbox_size, distort_bbox = tf.image.sample_distorted_bounding_box(
                    tf.shape(image), bounding_boxes=tf.expand_dims(bboxes, 0),
                    min_object_covered=0.25, aspect_ratio_range=(0.6, 1.67),
                    area_range=(0.1, 1.0), max_attempts=200, use_image_if_no_bounding_boxes=True)
        distort_bbox = distort_bbox[0, 0]
        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        # Restore the shape since the dynamic slice loses 3rd dimension.
        cropped_image.set_shape([None, None, 3])
        # Update bounding boxes: resize and filter out.
        bboxes = self.bboxes_resize(distort_bbox, bboxes)
        return cropped_image, labels, bboxes



    # Filter out overlapped bounding boxes 
    # Inputs:
    #     labels & bounding boxes
    # Outputs:
    #     filterout labels, bboxes
    def filterout_overlapped_bboxes(self, labels, bboxes):
        scores = self.calculate_bboxes_intersection(tf.constant([0, 0, 1, 1], bboxes.dtype), bboxes)
        mask = scores > 0.5
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)
        return labels, bboxes



    # Compute intersection between a reference box and a collection of bounding boxes. 
    # Inputs:
    #     bbox_ref, bboxes
    # Outputs:
    #     Tensor with relative intersection.
    def calculate_bboxes_intersection(self, bbox_ref, bboxes):
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        inter_vol =  tf.maximum(int_ymax - int_ymin, 0.) * tf.maximum(int_xmax - int_xmin, 0.)
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = tf.where(math_ops.greater(bboxes_vol, 0), math_ops.divide(inter_vol, bboxes_vol),
                          tf.zeros_like(inter_vol))
        return scores

     

    # Resize bounding boxes based on a reference bounding box,
    # Inputs:
    #     bounding boxes
    # Outputs:
    #     bounding boxes   
    def bboxes_resize(self, bbox_ref, bboxes):
        if isinstance(bboxes, dict):
            d_bboxes = {}
            for c in bboxes.keys():
                d_bboxes[c] = self.bboxes_resize(bbox_ref, bboxes[c])
            return d_bboxes
        bboxes = bboxes - tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        s = tf.stack([bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0], bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s
        return bboxes



    # This function flips left-right bounding boxes.
    # Inputs:
    #      bounding boxes
    # Outputs:
    #      flipped bounding bboxes
    def flip_bboxes(self, bboxes):
        bboxes = tf.stack([bboxes[:, 0], 1 - bboxes[:, 3], bboxes[:, 2], 1 - bboxes[:, 1]], axis=-1)
        return bboxes


        
    # This function flips left-right of an image and its bounding boxes.
    # Inputs:
    #     labels & bounding boxes
    # Outputs:
    #      flipped images & bboxes
    def flip_image_bounding_boxes(self, image, bboxes, seed=None):
        image = ops.convert_to_tensor(image)
        uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
        mirror_cond = math_ops.less(uniform_random, .5)
        # Flip image.
        flipped_image = control_flow_ops.cond(mirror_cond, lambda: array_ops.reverse_v2(image, [1]), lambda: image)
        flipped_image.set_shape(image.get_shape())
        # Flip bboxes.
        flipped_bboxes = control_flow_ops.cond(mirror_cond, lambda: self.flip_bboxes(bboxes), lambda: bboxes)
        return flipped_image, flipped_bboxes












