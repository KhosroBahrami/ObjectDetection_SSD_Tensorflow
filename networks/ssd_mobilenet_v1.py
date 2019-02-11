
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
slim = tf.contrib.slim

import functools


# Conv and DepthSepConv namedtuple define layers of the MobileNet architecture
# Conv defines 3x3 convolution layers
# DepthSepConv defines 3x3 depthwise convolution followed by 1x1 convolution.
# stride is the stride of the convolution
# depth is the number of channels or filters in a layer
Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['kernel', 'stride', 'depth'])

# MOBILENETV1_CONV_DEFS specifies the MobileNet body
MOBILENETV1_CONV_DEFS = [
    Conv(kernel=[3, 3], stride=2, depth=32),
    DepthSepConv(kernel=[3, 3], stride=1, depth=64),
    DepthSepConv(kernel=[3, 3], stride=2, depth=128),
    DepthSepConv(kernel=[3, 3], stride=1, depth=128),
    DepthSepConv(kernel=[3, 3], stride=2, depth=256),
    DepthSepConv(kernel=[3, 3], stride=1, depth=256),
    DepthSepConv(kernel=[3, 3], stride=2, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512),
    DepthSepConv(kernel=[3, 3], stride=1, depth=512)
]



class SSD_network(object):

    def __init__(self): 
        self.img_shape=(FLAGS.image_size, FLAGS.image_size)
        self.num_classes=FLAGS.num_classes
        self.no_annotation_label=FLAGS.num_classes
        # MobileNet_V1
        self.feat_layers=['Conv2d_5_pointwise', 'Conv2d_11_pointwise', 'block12', 'block13', 'block14', 'block15']
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



    def mobilenet_v1_base(self, inputs,
                          final_endpoint='Conv2d_13_pointwise',
                          min_depth=8,
                          depth_multiplier=1.0,
                          conv_defs=None,
                          output_stride=None,
                          use_explicit_padding=False,
                          scope=None):


      depth = lambda d: max(int(d * depth_multiplier), min_depth)
      end_points = {}

      # Used to find thinned depths for each layer.
      if depth_multiplier <= 0:
          raise ValueError('depth_multiplier is not greater than zero.')

      if conv_defs is None:
          conv_defs = MOBILENETV1_CONV_DEFS

      if output_stride is not None and output_stride not in [8, 16, 32]:
          raise ValueError('Only allowed output_stride values are 8, 16, 32.')

      padding = 'SAME'
      if use_explicit_padding:
          padding = 'VALID'
      with tf.variable_scope(scope, 'MobilenetV1', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.separable_conv2d], padding=padding):
          # The current_stride variable keeps track of the output stride of the
          # activations, i.e., the running product of convolution strides up to the
          # current network layer. This allows us to invoke atrous convolution
          # whenever applying the next convolution would result in the activations
          # having output stride larger than the target output_stride.
          current_stride = 1

          # The atrous convolution rate parameter.
          rate = 1

          net = inputs
          for i, conv_def in enumerate(conv_defs):
              end_point_base = 'Conv2d_%d' % i

              if output_stride is not None and current_stride == output_stride:
                  # If we have reached the target output_stride, then we need to employ
                  # atrous convolution with stride=1 and multiply the atrous rate by the
                  # current unit's stride for use in subsequent layers.
                  layer_stride = 1
                  layer_rate = rate
                  rate *= conv_def.stride
              else:
                  layer_stride = conv_def.stride
                  layer_rate = 1
                  current_stride *= conv_def.stride

              if isinstance(conv_def, Conv): # Convolution
                  end_point = end_point_base
                  if use_explicit_padding:
                      net = _fixed_padding(net, conv_def.kernel)
                  net = slim.conv2d(net, depth(conv_def.depth), conv_def.kernel, stride=conv_def.stride,
                                    scope=end_point)
                  end_points[end_point] = net
                  if end_point == final_endpoint:
                      return net, end_points

              elif isinstance(conv_def, DepthSepConv): # Depthwise separable convolution
                  end_point = end_point_base + '_depthwise'
                  # By passing filters=None
                  # Depthwise convolution: separable_conv2d produces only a depthwise convolution layer
                  if use_explicit_padding:
                      net = _fixed_padding(net, conv_def.kernel, layer_rate)
                  net = slim.separable_conv2d(net, None, conv_def.kernel, depth_multiplier=1,
                                              stride=layer_stride, rate=layer_rate, scope=end_point)

                  #print('  block: ',i, ' : ',net.shape)
                  end_points[end_point] = net
                  #if end_point == final_endpoint:
                  #    return net, end_points
                    
                  # Pointwise convolution
                  end_point = end_point_base + '_pointwise'
                  net = slim.conv2d(net, depth(conv_def.depth), [1, 1], stride=1, scope=end_point)
                  print(end_point, ' : ',net.shape)
                  end_points[end_point] = net

                  #if end_point == final_endpoint:
                  #    return net, end_points
              else:
                  raise ValueError('Unknown convolution type %s for layer %d' % (conv_def.ltype, i))


          # Block 8: 1 conv (1*1), 1 conv (3*3)
          block = 'block12'
          with tf.variable_scope(block):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = self.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
          print(block,net.shape)
          end_points[block] = net
          # Block 9: 1 conv (1*1), 1 conv (3*3) 
          block = 'block13'
          with tf.variable_scope(block):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = self.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
          print(block,net.shape)
          end_points[block] = net
          # Block 10: 1 conv (1*1), 1 conv (3*3) 
          block = 'block14'
          with tf.variable_scope(block):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
          print(block,net.shape)
          end_points[block] = net
          # Block 11: 1 conv (1*1), 1 conv (3*3) 
          block = 'block15'
          with tf.variable_scope(block):
                net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
                net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
          print(block,net.shape)
          end_points[block] = net

          # khosro
          return net, end_points
        



    def net(self, inputs,
    #def mobilenet_v1(self, inputs,
                     num_classes=1000,
                     dropout_keep_prob=0.999,
                     is_training=True,
                     min_depth=8,
                     depth_multiplier=1.0,
                     conv_defs=None,
                     prediction_fn=tf.contrib.layers.softmax,
                     spatial_squeeze=True,
                     reuse=None,
                     scope='MobilenetV1',
                     global_pool=False):
      """Mobilenet v1 model for classification.
       Raises:
        ValueError: Input rank is invalid.
      """
      print('  input: ',inputs.shape)
      input_shape = inputs.get_shape().as_list()
      if len(input_shape) != 4:
          raise ValueError('Invalid input tensor rank, expected 4, was: %d' % len(input_shape))

      with tf.variable_scope(scope, 'MobilenetV1', [inputs], reuse=reuse) as scope:
          with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
              net, end_points = self.mobilenet_v1_base(inputs, scope=scope, min_depth=min_depth,
                                                  depth_multiplier=depth_multiplier, conv_defs=conv_defs)

      r = self.ssd_multibox_layer(end_points)
      return r





    def _fixed_padding(self, inputs, kernel_size, rate=1):
      """Pads the input along the spatial dimensions independently of input size.

      Pads the input such that if it was used in a convolution with 'VALID' padding,
      the output would have the same dimensions as if the unpadded input was used
      in a convolution with 'SAME' padding.

      Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
        rate: An integer, rate for atrous convolution.

      Returns:
        output: A tensor of size [batch, height_out, width_out, channels] with the
          input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
      """
      kernel_size_effective = [kernel_size[0] + (kernel_size[0] - 1) * (rate - 1),
                               kernel_size[0] + (kernel_size[0] - 1) * (rate - 1)]
      pad_total = [kernel_size_effective[0] - 1, kernel_size_effective[1] - 1]
      pad_beg = [pad_total[0] // 2, pad_total[1] // 2]
      pad_end = [pad_total[0] - pad_beg[0], pad_total[1] - pad_beg[1]]
      padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg[0], pad_end[0]],
                                      [pad_beg[1], pad_end[1]], [0, 0]])
      return padded_inputs



    def wrapped_partial(self, func, *args, **kwargs):
      partial_func = functools.partial(func, *args, **kwargs)
      functools.update_wrapper(partial_func, func)
      return partial_func





    def _reduced_kernel_size_for_small_input(self, input_tensor, kernel_size):
      """Define kernel size which is automatically reduced for small input.

      If the shape of the input images is unknown at graph construction time this
      function assumes that the input images are large enough.

      Args:
        input_tensor: input tensor of size [batch_size, height, width, channels].
        kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

      Returns:
        a tensor with the kernel size.
      """
      shape = input_tensor.get_shape().as_list()
      if shape[1] is None or shape[2] is None:
        kernel_size_out = kernel_size
      else:
        kernel_size_out = [min(shape[1], kernel_size[0]),
                           min(shape[2], kernel_size[1])]
      return kernel_size_out


    def mobilenet_v1_arg_scope(self, 
        is_training=True,
        weight_decay=0.00004,
        stddev=0.09,
        regularize_depthwise=False,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=0.001,
        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
        normalizer_fn=slim.batch_norm):
      """Defines the default MobilenetV1 arg scope.

      Args:
        is_training: Whether or not we're training the model. If this is set to
          None, the parameter is not added to the batch_norm arg_scope.
        weight_decay: The weight decay to use for regularizing the model.
        stddev: The standard deviation of the trunctated normal weight initializer.
        regularize_depthwise: Whether or not apply regularization on depthwise.
        batch_norm_decay: Decay for batch norm moving average.
        batch_norm_epsilon: Small float added to variance to avoid dividing by zero
          in batch norm.
        batch_norm_updates_collections: Collection for the update ops for
          batch norm.
        normalizer_fn: Normalization function to apply after convolution.

      Returns:
        An `arg_scope` to use for the mobilenet v1 model.
      """
      batch_norm_params = {
          'center': True,
          'scale': True,
          'decay': batch_norm_decay,
          'epsilon': batch_norm_epsilon,
          'updates_collections': batch_norm_updates_collections,
      }
      if is_training is not None:
        batch_norm_params['is_training'] = is_training

      # Set weight_decay for weights in Conv and DepthSepConv layers.
      weights_init = tf.truncated_normal_initializer(stddev=stddev)
      regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
      if regularize_depthwise:
        depthwise_regularizer = regularizer
      else:
        depthwise_regularizer = None
      with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                          weights_initializer=weights_init,
                          activation_fn=tf.nn.relu6, normalizer_fn=normalizer_fn):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):
            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer) as sc:
              return sc




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



















