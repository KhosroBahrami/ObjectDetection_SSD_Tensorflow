
# Postprocessing
import numpy as np
import tensorflow as tf
from configs.config_test import *



class Postprocessing(object):

    def __init__(self):      
        a=1


    # This function selects top bounding boxes from network output layers.
    # Inputs:
    #      predictions_net: List of SSD prediction layers;
    #      localizations_net: List of localization layers;
    # Outputs:
    #      d_scores, d_bboxes
    def select_top_bboxes(self, predictions_net, localizations_net, scope=None):

        with tf.name_scope(scope, 'ssd_bboxes_select', [predictions_net, localizations_net]):
            l_scores = []
            l_bboxes = []
            for i in range(len(predictions_net)):
                scores, bboxes = self.select_top_bboxes_layer(predictions_net[i], localizations_net[i])
                l_scores.append(scores)
                l_bboxes.append(bboxes)
            # Concat results.
            d_scores = {}
            d_bboxes = {}
            for c in l_scores[0].keys():
                ls = [s[c] for s in l_scores]
                lb = [b[c] for b in l_bboxes]
                d_scores[c] = tf.concat(ls, axis=1)
                d_bboxes[c] = tf.concat(lb, axis=1)
            return d_scores, d_bboxes



    # Select top bounding boxes from features in one layer.
    # Inputs:
    #      predictions_layer: A SSD prediction layer
    #      localizations_layer: A SSD localization layer
    # Outputs:
    #      d_scores, d_bboxes
    def select_top_bboxes_layer(self, predictions_layer, localizations_layer, scope=None):
        select_threshold = 0.0 if FLAGS.test_select_threshold is None else FLAGS.test_select_threshold
        p_shape = self.get_shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer, tf.stack([p_shape[0], -1, p_shape[-1]]))
        l_shape = self.get_shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer, tf.stack([l_shape[0], -1, l_shape[-1]]))
        d_scores = {}
        d_bboxes = {}
        for c in range(1, FLAGS.num_classes):
            # Remove boxes under the threshold.
            scores = predictions_layer[:, :, c]
            fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
            scores = scores * fmask
            bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)
            # Append to dictionary.
            d_scores[c] = scores
            d_bboxes[c] = bboxes
        return d_scores, d_bboxes




    # Sort bounding boxes 
    # Inputs:
    #      scores, bboxes: 
    # Outputs:
    #      scores, bboxes
    def sort_bboxes(self, scores, bboxes):

        if isinstance(scores, dict) or isinstance(bboxes, dict):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                    d_scores[c], d_bboxes[c] = self.sort_bboxes(scores[c], bboxes[c])
            return d_scores, d_bboxes
        scores, idxes = tf.nn.top_k(scores, k=FLAGS.test_sort_top_k, sorted=True)
        r = tf.map_fn(lambda x: [tf.gather(x[0], x[1])], [bboxes, idxes], dtype=[bboxes.dtype],
                          parallel_iterations=10, back_prop=False, swap_memory=False, infer_shape=True)
        bboxes = r[0]
        return scores, bboxes




    # Apply non-maximum selection to bounding boxes. 
    # Inputs:
    #      scores, bboxes
    # Outputs:
    #      scores, bboxes 
    def bboxes_nms(self, scores, bboxes):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores, FLAGS.test_nms_top_k, FLAGS.test_matching_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)
        # Pad results.
        scores = self.pad_axis(scores, 0, FLAGS.test_nms_top_k, axis=0)
        bboxes = self.pad_axis(bboxes, 0, FLAGS.test_nms_top_k, axis=0)
        return scores, bboxes



    # Apply non-maximum selection to bounding boxes. 
    # Inputs:
    #      scores, bboxes
    # Outputs:
    #      scores, bboxes 
    def non_maximum_supression_bboxes(self, scores, bboxes):
        # Dictionaries as inputs:
        if isinstance(scores, dict) or isinstance(bboxes, dict):
                d_scores = {}
                d_bboxes = {}
                for c in scores.keys():
                    d_scores[c], d_bboxes[c] = self.non_maximum_supression_bboxes(scores[c], bboxes[c])
                return d_scores, d_bboxes
        # Tensors as inputs:
        r = tf.map_fn(lambda x: self.bboxes_nms(x[0], x[1]),
                          (scores, bboxes), dtype=(scores.dtype, bboxes.dtype), parallel_iterations=10,
                          back_prop=False, swap_memory=False, infer_shape=True)
        scores, bboxes = r
        return scores, bboxes




    # Pad a tensor on an axis, with a given offset and output size. The tensor is padded with zero. 
    # Inputs:
    #      x: Tensor to pad;
    #      offset: Offset to add on the dimension chosen;
    #      size: Final size of the dimension.
    # Outputs:
    #      Padded tensor 
    def pad_axis(self, x, offset, size, axis=0, name=None):
        with tf.name_scope(name, 'pad_axis'):
            shape = self.get_shape(x)
            rank = len(shape)
            new_size = tf.maximum(size-offset-shape[axis], 0)
            pad1 = tf.stack([0]*axis + [offset] + [0]*(rank-axis-1))
            pad2 = tf.stack([0]*axis + [new_size] + [0]*(rank-axis-1))
            paddings = tf.stack([pad1, pad2], axis=1)
            x = tf.pad(x, paddings, mode='CONSTANT')
            shape[axis] = size
            x = tf.reshape(x, tf.stack(shape))
            return x



    # Returns the dimensions of a Tensor as list of integers or scale tensors.
    # Inputs:
    #      x: N-d Tensor;
    # Outputs:
    #      A list of dimensions of input tensor. 
    def get_shape(self, x, rank=None):
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape = x.get_shape()
            if rank is None:
                static_shape = static_shape.as_list()
                rank = len(static_shape)
            else:
                static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(x), rank)
            return [s if s is not None else d
                    for s, d in zip(static_shape, dynamic_shape)]




 
