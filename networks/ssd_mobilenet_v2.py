
# SSD VGG_based network
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from collections import namedtuple
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import nn
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from configs.config_common import *

import copy
import functools

import tensorflow as tf

import tensorflow.contrib.slim.nets as nets


import networks.utils.conv_blocks as ops
import networks.utils.mobilenet_v2_funcs as lib

slim = tf.contrib.slim



op = lib.op

expand_input = ops.expand_input_by_factor

# pyformat: disable
# Architecture: https://arxiv.org/abs/1801.04381
V2_DEF = dict(
    defaults={
        # Note: these parameters of batch norm affect the architecture
        # that's why they are here and not in training_scope.
        (slim.batch_norm,): {'center': True, 'scale': True},
        (slim.conv2d, slim.fully_connected, slim.separable_conv2d): {
            'normalizer_fn': slim.batch_norm, 'activation_fn': tf.nn.relu6
        },
        (ops.expanded_conv,): {
            'expansion_size': expand_input(6),
            'split_expansion': 1,
            'normalizer_fn': slim.batch_norm,
            'residual': True
        },
        (slim.conv2d, slim.separable_conv2d): {'padding': 'SAME'}
    },
    spec=[
        op(slim.conv2d, stride=2, num_outputs=32, kernel_size=[3, 3]),
        op(ops.expanded_conv,
           expansion_size=expand_input(1, divisible_by=1),
           num_outputs=16),
        op(ops.expanded_conv, stride=2, num_outputs=24),
        op(ops.expanded_conv, stride=1, num_outputs=24),
        op(ops.expanded_conv, stride=2, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=1, num_outputs=32),
        op(ops.expanded_conv, stride=2, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=64),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        op(ops.expanded_conv, stride=1, num_outputs=96),
        #op(ops.expanded_conv, stride=2, num_outputs=160),
        #op(ops.expanded_conv, stride=1, num_outputs=160),
        #op(ops.expanded_conv, stride=1, num_outputs=160),
        #op(ops.expanded_conv, stride=1, num_outputs=320),
        #op(slim.conv2d, stride=1, kernel_size=[1, 1], num_outputs=1280)
    ],
)
# pyformat: enable







class SSD_network(object):

    def __init__(self): 
        self.img_shape=(FLAGS.image_size, FLAGS.image_size)
        self.num_classes=FLAGS.num_classes
        self.no_annotation_label=FLAGS.num_classes
        # MobileNet_V2
        self.feat_layers=['layer_7', 'layer_14', 'block12', 'block13', 'block14', 'block15']
        self.feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
        self.anchor_size_bounds=[0.15, 0.90]
        self.anchor_sizes=[(21., 45.), (45., 99.), (99., 153.), (153., 207.), (207., 261.), (261., 315.)]
        self.anchor_ratios=[[2, .5], [2, .5, 3, 1./3], [2, .5, 3, 1./3], [2, .5, 3, 1./3], [2, .5], [2, .5]]
        self.anchor_steps=[8, 16, 32, 64, 100, 300]
        self.anchor_offset=0.5
        self.normalizations=[20, -1, -1, -1, -1, -1]
        self.prior_scaling=[0.1, 0.1, 0.2, 0.2]





    # The multibox layer of SSD
    # Inputs:
    #     output of network layers
    # Outputs:
    #     predictions & localizations of objects
    def ssd_multibox_layer(self, end_points):
        predictions = []
        logits = []
        localisations = []

        for e in end_points:
            print('--> : ', e)

        
        print('\n  len of anchor size, anchor ratio: ')
        for i, layer in enumerate(self.feat_layers):
            with tf.variable_scope(layer + '_box'):
                net = end_points[layer]
                if self.normalizations[i] > 0:
                    net = self.spatial_normalization(net)
                # Number of anchors.
                num_anchors = len(self.anchor_sizes[i]) + len(self.anchor_ratios[i])
                print(' ',layer,'_box:', len(self.anchor_sizes[i]) , len(self.anchor_ratios[i]) )
                # Location prediction:
                num_loc_pred = num_anchors * 4
                loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
                loc_pred = tf.reshape(loc_pred, self.tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
                # Class prediction:
                num_cls_pred = num_anchors * FLAGS.num_classes
                cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
                cls_pred = tf.reshape(cls_pred, self.tensor_shape(cls_pred, 4)[:-1]+[num_anchors, FLAGS.num_classes])

            predictions.append(slim.softmax(cls_pred))
            logits.append(cls_pred)
            localisations.append(loc_pred)
        return predictions, localisations, logits, end_points
        




###########################
###########################


 

    @slim.add_arg_scope
    def mobilenet_v2(self, input_tensor,
                  num_classes=1001,
                  depth_multiplier=1.0,
                  scope='MobilenetV2',
                  conv_defs=None,
                  finegrain_classification_mode=False,
                  min_depth=None,
                  divisible_by=None,
                  activation_fn=None,
                  **kwargs):
      """Creates mobilenet V2 network.

      Inference mode is created by default. To create training use training_scope
      below.

      with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
         logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

      Args:
        input_tensor: The input tensor
        num_classes: number of classes
        depth_multiplier: The multiplier applied to scale number of
        channels in each layer. Note: this is called depth multiplier in the
        paper but the name is kept for consistency with slim's model builder.
        scope: Scope of the operator
        conv_defs: Allows to override default conv def.
        finegrain_classification_mode: When set to True, the model
        will keep the last layer large even for small multipliers. Following
        https://arxiv.org/abs/1801.04381
        suggests that it improves performance for ImageNet-type of problems.
          *Note* ignored if final_endpoint makes the builder exit earlier.
        min_depth: If provided, will ensure that all layers will have that
        many channels after application of depth multiplier.
        divisible_by: If provided will ensure that all layers # channels
        will be divisible by this number.
        activation_fn: Activation function to use, defaults to tf.nn.relu6 if not
          specified.
        **kwargs: passed directly to mobilenet.mobilenet:
          prediction_fn- what prediction function to use.
          reuse-: whether to reuse variables (if reuse set to true, scope
          must be given).
      Returns:
        logits/endpoints pair

      Raises:
        ValueError: On invalid arguments
      """
      if conv_defs is None:
        conv_defs = V2_DEF
      if 'multiplier' in kwargs:
        raise ValueError('mobilenetv2 doesn\'t support generic '
                         'multiplier parameter use "depth_multiplier" instead.')
      if finegrain_classification_mode:
        conv_defs = copy.deepcopy(conv_defs)
        if depth_multiplier < 1:
          conv_defs['spec'][-1].params['num_outputs'] /= depth_multiplier
      if activation_fn:
        conv_defs = copy.deepcopy(conv_defs)
        defaults = conv_defs['defaults']
        conv_defaults = (
            defaults[(slim.conv2d, slim.fully_connected, slim.separable_conv2d)])
        conv_defaults['activation_fn'] = activation_fn

      depth_args = {}
      # NB: do not set depth_args unless they are provided to avoid overriding
      # whatever default depth_multiplier might have thanks to arg_scope.
      if min_depth is not None:
        depth_args['min_depth'] = min_depth
      if divisible_by is not None:
        depth_args['divisible_by'] = divisible_by

      with slim.arg_scope((lib.depth_multiplier,), **depth_args):
        logits, end_points = lib.mobilenet(
            input_tensor,
            num_classes=num_classes,
            conv_defs=conv_defs,
            scope=scope,
            multiplier=depth_multiplier,
            **kwargs)

      r = self.ssd_multibox_layer(end_points)

      return r
    #mobilenet.default_image_size = 224


    def wrapped_partial(self, func, *args, **kwargs):
      partial_func = functools.partial(func, *args, **kwargs)
      functools.update_wrapper(partial_func, func)
      return partial_func


    # Wrappers for mobilenet v2 with depth-multipliers. Be noticed that
    # 'finegrain_classification_mode' is set to True, which means the embedding
    # layer will not be shrinked when given a depth-multiplier < 1.0.
    
    #mobilenet_v2_140 = wrapped_partial(mobilenet, depth_multiplier=1.4)
    #mobilenet_v2_050 = wrapped_partial(mobilenet, depth_multiplier=0.50, finegrain_classification_mode=True)
    #mobilenet_v2_035 = wrapped_partial(mobilenet, depth_multiplier=0.35, finegrain_classification_mode=True)


    @slim.add_arg_scope
    def mobilenet_base(self, input_tensor, depth_multiplier=1.0, **kwargs):
      """Creates base of the mobilenet (no pooling and no logits) ."""
      return mobilenet(input_tensor,
                       depth_multiplier=depth_multiplier,
                       base_only=True, **kwargs)


    def training_scope(self, **kwargs):
      """Defines MobilenetV2 training scope.

      Usage:
         with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
           logits, endpoints = mobilenet_v2.mobilenet(input_tensor)

      with slim.

      Args:
        **kwargs: Passed to mobilenet.training_scope. The following parameters
        are supported:
          weight_decay- The weight decay to use for regularizing the model.
          stddev-  Standard deviation for initialization, if negative uses xavier.
          dropout_keep_prob- dropout keep probability
          bn_decay- decay for the batch norm moving averages.

      Returns:
        An `arg_scope` to use for the mobilenet v2 model.
      """
      return lib.training_scope(**kwargs)


    __all__ = ['training_scope', 'mobilenet_base', 'mobilenet', 'V2_DEF']





##############################
##############################






    # This function performs spatial normalization on every feature 
    # Inputs:
    #     output of network as 4-D tensor with dimensions [batch_size, height, width, channels].
    # Outputs:
    #     normalized features
    @add_arg_scope
    def spatial_normalization(self, inputs):
        with variable_scope.variable_scope(None, 'L2Normalization', [inputs], reuse=None) as sc:
            inputs_shape = inputs.get_shape()
            inputs_rank = inputs_shape.ndims
            norm_dim = tf.range(inputs_rank-1, inputs_rank)
            params_shape = inputs_shape[-1:]
            # Normalize along spatial dimensions.
            outputs = nn.l2_normalize(inputs, norm_dim, epsilon=1e-12)
            # Additional scaling.
            scale_collections = utils.get_variable_collections(None, 'scale')
            scale = variables.model_variable('gamma',
                                                 shape=params_shape,
                                                 dtype=inputs.dtype.base_dtype,
                                                 initializer=init_ops.ones_initializer(),
                                                 collections=scale_collections,
                                                 trainable=True)
            outputs = tf.multiply(outputs, scale)
            return utils.collect_named_outputs(None, sc.original_name_scope, outputs)



    # This function adds symmetric padding to H and W dimensions
    # Inputs:
    #     4D input tensor 
    # Outputs:
    #     padded tensor
    @add_arg_scope
    def pad2d(self, inputs, pad=(0, 0)):
        paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        net = tf.pad(inputs, paddings, mode='CONSTANT')
        return net
  

    
    #
    # Args:
    #      weight_decay: The l2 regularization coefficient.
    # Returns:
    #  An arg_scope.
    def arg_scope(self, weight_decay=0.0005, data_format='NHWC'):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d], padding='SAME', data_format=data_format):
                with slim.arg_scope([self.pad2d, self.spatial_normalization]) as sc:
                     return sc
    

   
    # This function generates the anchor boxes.  
    # Inputs:
    #     feat_shape: Feature shape, used for computing relative position grids;
    # Output:
    #      y, x, h, w: Relative x and y grids, and height and width.
    def generate_anchors(self):
        layers_anchors = []
        for i, feat_shape in enumerate(self.feat_shapes):
            sizes = self.anchor_sizes[i]
            ratios = self.anchor_ratios[i]
            step = self.anchor_steps[i]
            offset = self.anchor_offset
            dtype=np.float32
                      
            y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
            y = (y.astype(dtype) + offset) * step / self.img_shape[0]
            x = (x.astype(dtype) + offset) * step / self.img_shape[1]
            # Expand dims to support easy broadcasting.
            y = np.expand_dims(y, axis=-1)
            x = np.expand_dims(x, axis=-1)
            # Compute relative height and width.
            num_anchors = len(sizes) + len(ratios)
            h = np.zeros((num_anchors, ), dtype=dtype)
            w = np.zeros((num_anchors, ), dtype=dtype)
            # Add first anchor boxes with ratio=1.
            h[0] = sizes[0] / self.img_shape[0]
            w[0] = sizes[0] / self.img_shape[1]
            di = 1
            if len(sizes) > 1:
                h[1] = math.sqrt(sizes[0] * sizes[1]) / self.img_shape[0]
                w[1] = math.sqrt(sizes[0] * sizes[1]) / self.img_shape[1]
                di += 1
            for i, r in enumerate(ratios):
                h[i+di] = sizes[0] / self.img_shape[0] / math.sqrt(r)
                w[i+di] = sizes[0] / self.img_shape[1] * math.sqrt(r)
            anchor_bboxes = y, x, h, w
            layers_anchors.append(anchor_bboxes)
        return layers_anchors



    # This function reshape a list
    # Inputs:
    #     a list 
    # Outputs:
    #     reshaped list
    def reshape_list(self, l, shape=None):
        r = []
        if shape is None: # Flatten everything.
            for a in l:
                if isinstance(a, (list, tuple)):
                    r = r + list(a)
                else:
                    r.append(a)
        else: # Reshape to list of list.
            i = 0
            for s in shape:
                if s == 1:
                    r.append(l[i])
                else:
                    r.append(l[i:i+s])
                i += s
        return r



    # This function generates batches of testing images
    # Inputs:
    #     images 
    # Outputs:
    #     batches of images
    def generate_test_batches(self, image, gt_labels, gt_bboxes, gt_difficult_objects,
              gt_bbox_img, gclasses, glocalisations, gscores, batch_shape,
                  batch_size=1, num_threads=1, capacity=5, dynamic_pad=False):
        r = tf.train.batch(
                self.reshape_list([image, gt_labels, gt_bboxes, gt_difficult_objects, gt_bbox_img,
                gclasses, glocalisations, gscores]), batch_size = 1, num_threads = 1, capacity = 5, dynamic_pad=True)
        b_image, b_glabels, b_gbboxes, b_gdifficults, b_gbbox_img, b_gclasses, b_glocalisations, b_gscores = self.reshape_list(r, batch_shape)
        return b_image, b_glabels, b_gbboxes, b_gdifficults, b_gbbox_img, b_gclasses, b_glocalisations, b_gscores



       
    # This function generates batches of training images
    # Inputs:
    #     images 
    # Outputs:
    #     batches of images
    def generate_train_batches(self, image, gclasses, glocalisations, gscores, batch_shape,
                  batch_size=1, num_threads=1, capacity=5, dynamic_pad=False):
        r = tf.train.batch(
                self.reshape_list([image, gclasses, glocalisations, gscores]),
                batch_size=1, num_threads=1, capacity=5 * 1)
        b_image, b_gclasses, b_glocalisations, b_gscores = self.reshape_list(r, batch_shape)
        batch_queue = slim.prefetch_queue.prefetch_queue(
                self.reshape_list([b_image, b_gclasses, b_glocalisations, b_gscores]), capacity=2 * 1)
        return b_image, b_gclasses, b_glocalisations, b_gscores, batch_queue
    



    # This function returns the dimensions of a tensor.
    # Inputs:
    #     A Tensor 
    # Outputs:
    #      list of dimensions. 
    def tensor_shape(self, x, rank):
        if x.get_shape().is_fully_defined():
            return x.get_shape().as_list()
        else:
            static_shape = x.get_shape().with_rank(rank).as_list()
            dynamic_shape = tf.unstack(tf.shape(x), rank)
            return [s if s is not None else d for s, d in zip(static_shape, dynamic_shape)]



    # This function returns the dimensions of a Tensor as list of integers or scale tensors.
    # Inputs:
    #      x: N-d Tensor;
    #      rank: Rank of the Tensor. 
    # Outputs:
    #      A list of dimensions of the tensor.         
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



    # Smoothed absolute function. Useful to compute an L1 smooth error.
    #    Define as:
    #        x^2 / 2         if abs(x) < 1
    #        abs(x) - 0.5    if abs(x) > 1
    def abs_smooth(self, x):
        absx = tf.abs(x)
        minx = tf.minimum(absx, 1)
        r = 0.5 * ((absx - 1) * minx + absx)
        return r



    
    # This function implements the losses of the network
    # Inputs:
    #     image: A N-D Tensor of shape.
    # Outputs:
    #      A list of dimensions. 
    def losses(self, logits, localisations, gclasses, glocalisations, gscores,
                   match_threshold=0.5, negative_ratio=3., alpha=1., label_smoothing=0., scope='ssd_losses'):
        with tf.name_scope(scope, 'ssd_losses'):
            lshape = self.get_shape(logits[0], 5)
            num_classes = lshape[-1]
            batch_size = lshape[0]

            flogits = []
            fgclasses = []
            fgscores = []
            flocalisations = []
            fglocalisations = []
            for i in range(len(logits)):
                flogits.append(tf.reshape(logits[i], [-1, num_classes]))
                fgclasses.append(tf.reshape(gclasses[i], [-1]))
                fgscores.append(tf.reshape(gscores[i], [-1]))
                flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
                fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))
            # concat 
            logits = tf.concat(flogits, axis=0)
            gclasses = tf.concat(fgclasses, axis=0)
            gscores = tf.concat(fgscores, axis=0)
            localisations = tf.concat(flocalisations, axis=0)
            glocalisations = tf.concat(fglocalisations, axis=0)
            dtype = logits.dtype
            # Compute positive matching mask...
            pmask = gscores > match_threshold
            fpmask = tf.cast(pmask, dtype)
            n_positives = tf.reduce_sum(fpmask)
            # Hard negative mining...
            no_classes = tf.cast(pmask, tf.int32)
            predictions = slim.softmax(logits)
            nmask = tf.logical_and(tf.logical_not(pmask), gscores > -0.5)
            fnmask = tf.cast(nmask, dtype)
            nvalues = tf.where(nmask, predictions[:, 0], 1. - fnmask)
            nvalues_flat = tf.reshape(nvalues, [-1])
            # Number of negative entries to select.
            max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
            n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
            n_neg = tf.minimum(n_neg, max_neg_entries)

            val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
            max_hard_pred = -val[-1]
            # Final negative mask.
            nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
            fnmask = tf.cast(nmask, dtype)

            # cross-entropy loss.
            with tf.name_scope('cross_entropy_pos'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=gclasses)
                loss = tf.div(tf.reduce_sum(loss * fpmask), batch_size, name='value')
                tf.losses.add_loss(loss)
            with tf.name_scope('cross_entropy_neg'):
                loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=no_classes)
                loss = tf.div(tf.reduce_sum(loss * fnmask), batch_size, name='value')
                tf.losses.add_loss(loss)

            # localization loss: smooth L1, L2, ...
            with tf.name_scope('localization'):
                # Weights Tensor: positive mask + random negative.
                weights = tf.expand_dims(alpha * fpmask, axis=-1)
                loss = self.abs_smooth(localisations - glocalisations)
                loss = tf.div(tf.reduce_sum(loss * weights), batch_size, name='value')
                tf.losses.add_loss(loss)
    



    # Generate groundtruth labels and bounding boxes using SSD net anchors.
    # Inputs:
    #     labels: 1D Tensor(int64) containing groundtruth labels;
    #     bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
    #     anchors: List of Numpy array with layer anchors;
    # Outputs:
    #     target_labels, target_localizations, target_scores:
    def generate_groundtruth_bboxes(self, labels, bboxes, anchors):

        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            t_labels, t_loc, t_scores = self.generate_groundtruth_bboxes_layer(labels, bboxes, anchors_layer)
            target_labels.append(t_labels)
            target_localizations.append(t_loc)
            target_scores.append(t_scores)
        return target_labels, target_localizations, target_scores




    # Generate groundtruth labels and bounding boxes for a layer
    # Inputs:
    #      labels: 1D Tensor(int64) containing groundtruth labels;
    #      bboxes: Nx4 Tensor(float) with bboxes relative coordinates;
    #      anchors_layer: Numpy array with layer anchors;
    # Outputs:
    #      target_labels, target_localizations, target_scores
    def generate_groundtruth_bboxes_layer(self, labels, bboxes, anchors_layer):

        ignore_threshold=0.5
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        dtype=tf.float32
        # Anchors coordinates and volume.
        yref, xref, href, wref = anchors_layer
        ymin = yref - href / 2.
        xmin = xref - wref / 2.
        ymax = yref + href / 2.
        xmax = xref + wref / 2.
        vol_anchors = (xmax - xmin) * (ymax - ymin)
        # Initialize tensors...
        shape = (yref.shape[0], yref.shape[1], href.size)
        feat_labels = tf.zeros(shape, dtype=tf.int64)
        feat_scores = tf.zeros(shape, dtype=dtype)
        feat_ymin = tf.zeros(shape, dtype=dtype)
        feat_xmin = tf.zeros(shape, dtype=dtype)
        feat_ymax = tf.ones(shape, dtype=dtype)
        feat_xmax = tf.ones(shape, dtype=dtype)
        
        def jaccard_with_anchors(bbox):
            """Compute jaccard score between a box and the anchors.
            """
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w
            union_vol = vol_anchors - inter_vol + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            jaccard = tf.div(inter_vol, union_vol)
            return jaccard

        def intersection_with_anchors(bbox):
            """Compute intersection between score a box and the anchors.
            """
            int_ymin = tf.maximum(ymin, bbox[0])
            int_xmin = tf.maximum(xmin, bbox[1])
            int_ymax = tf.minimum(ymax, bbox[2])
            int_xmax = tf.minimum(xmax, bbox[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            inter_vol = h * w
            scores = tf.div(inter_vol, vol_anchors)
            return scores
        
        def condition(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            r = tf.less(i, tf.shape(labels))
            return r[0]

        def body(i, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax):
            # Jaccard score.
            label = labels[i]
            bbox = bboxes[i]
            jaccard = jaccard_with_anchors(bbox)
            # Mask: check threshold + scores + no annotations + num_classes.
            mask = tf.greater(jaccard, feat_scores)
            mask = tf.logical_and(mask, feat_scores > -0.5)
            mask = tf.logical_and(mask, label < self.num_classes)
            imask = tf.cast(mask, tf.int64)
            fmask = tf.cast(mask, dtype)
            # Update values using mask.
            feat_labels = imask * label + (1 - imask) * feat_labels
            feat_scores = tf.where(mask, jaccard, feat_scores)
            feat_ymin = fmask * bbox[0] + (1 - fmask) * feat_ymin
            feat_xmin = fmask * bbox[1] + (1 - fmask) * feat_xmin
            feat_ymax = fmask * bbox[2] + (1 - fmask) * feat_ymax
            feat_xmax = fmask * bbox[3] + (1 - fmask) * feat_xmax
            return [i+1, feat_labels, feat_scores, feat_ymin, feat_xmin, feat_ymax, feat_xmax]
      
        # Main loop definition.
        i = 0
        [i, feat_labels, feat_scores,
         feat_ymin, feat_xmin,
         feat_ymax, feat_xmax] = tf.while_loop(condition, body,
                                               [i, feat_labels, feat_scores,
                                                feat_ymin, feat_xmin, feat_ymax, feat_xmax])
        # Transform to center / size.
        feat_cy = (feat_ymax + feat_ymin) / 2.
        feat_cx = (feat_xmax + feat_xmin) / 2.
        feat_h = feat_ymax - feat_ymin
        feat_w = feat_xmax - feat_xmin
        # Encode features.
        feat_cy = (feat_cy - yref) / href / prior_scaling[0]
        feat_cx = (feat_cx - xref) / wref / prior_scaling[1]
        feat_h = tf.log(feat_h / href) / prior_scaling[2]
        feat_w = tf.log(feat_w / wref) / prior_scaling[3]
        # Use SSD ordering: x / y / w / h instead of ours.
        feat_localizations = tf.stack([feat_cx, feat_cy, feat_w, feat_h], axis=-1)
        return feat_labels, feat_localizations, feat_scores


    
    # Compute the relative bounding boxes from the SSD net features and reference anchors bounding boxes.
    # Inputs:
    #     feat_localizations: List of Tensors containing localization features.
    #     anchors: List of numpy array containing anchor boxes.
    # Outputs:
    #      List of Tensors Nx4: ymin, xmin, ymax, xmax
    def generate_relative_bboxes(self, feat_localizations, anchors, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(self.generate_relative_bboxes_layer(feat_localizations[i], anchors_layer, prior_scaling))
        return bboxes
            


    # Compute the relative bounding boxes from the layer features and reference anchor bounding boxes.
    # Inputs:
    #      feat_localizations: Tensor containing localization features.
    #      anchors: List of numpy array containing anchor boxes.
    # Outputs:
    #      Tensor Nx4: ymin, xmin, ymax, xmax
    def generate_relative_bboxes_layer(self, feat_localizations, anchors_layer,
                                   prior_scaling=[0.1, 0.1, 0.2, 0.2]):
        yref, xref, href, wref = anchors_layer
        # Compute center, height and width
        cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
        cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
        w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
        h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
        # Boxes coordinates.
        ymin = cy - h / 2.
        xmin = cx - w / 2.
        ymax = cy + h / 2.
        xmax = cx + w / 2.
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
        return bboxes



















