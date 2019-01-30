
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


from networks.utils import resnet_utils
resnet_arg_scope = resnet_utils.resnet_arg_scope



class SSD_network(object):

    def __init__(self): 
        self.img_shape=(FLAGS.image_size, FLAGS.image_size)
        self.num_classes=FLAGS.num_classes
        self.no_annotation_label=FLAGS.num_classes
        # Resnet_V2
        self.feat_layers=['resnet_v2/block2', 'resnet_v2/block3', 'resnet_v2/block4', 'block13', 'block14', 'block15']
        self.feat_shapes=[(19, 19), (10, 10), (10, 10), (5, 5), (3, 3), (1, 1)]

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
    def bottleneck(self, inputs, depth, depth_bottleneck, stride, rate=1,
                   outputs_collections=None, scope=None):
      """Bottleneck residual unit variant with BN before convolutions.

      This is the full preactivation residual unit variant proposed in [2]. See
      Fig. 1(b) of [2] for its definition. Note that we use here the bottleneck
      variant which has an extra bottleneck layer.

      When putting together two consecutive ResNet blocks that use this unit, one
      should use stride = 2 in the last unit of the first block.

      Args:
        inputs: A tensor of size [batch, height, width, channels].
        depth: The depth of the ResNet unit output.
        depth_bottleneck: The depth of the bottleneck layers.
        stride: The ResNet unit's stride. Determines the amount of downsampling of
          the units output compared to its input.
        rate: An integer, rate for atrous convolution.
        outputs_collections: Collection to add the ResNet unit output.
        scope: Optional variable_scope.

      Returns:
        The ResNet unit's output.
      """
      with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
          shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
          shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                 normalizer_fn=None, activation_fn=None,
                                 scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1, scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride, rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections, sc.name, output)


    def resnet_v2(self, inputs,
                  blocks,
                  num_classes=None,
                  is_training=True,
                  global_pool=True,
                  output_stride=None,
                  include_root_block=True,
                  spatial_squeeze=True,
                  reuse=None,
                  scope=None):
      """Generator for v2 (preactivation) ResNet models.

      This function generates a family of ResNet v2 models. See the resnet_v2_*()
      methods for specific model instantiations, obtained by selecting different
      block instantiations that produce ResNets of various depths.

      Training for image classification on Imagenet is usually done with [224, 224]
      inputs, resulting in [7, 7] feature maps at the output of the last ResNet
      block for the ResNets defined in [1] that have nominal stride equal to 32.
      However, for dense prediction tasks we advise that one uses inputs with
      spatial dimensions that are multiples of 32 plus 1, e.g., [321, 321]. In
      this case the feature maps at the ResNet output will have spatial shape
      [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]
      and corners exactly aligned with the input image corners, which greatly
      facilitates alignment of the features to the image. Using as input [225, 225]
      images results in [8, 8] feature maps at the output of the last ResNet block.

      For dense prediction tasks, the ResNet needs to run in fully-convolutional
      (FCN) mode and global_pool needs to be set to False. The ResNets in [1, 2] all
      have nominal stride equal to 32 and a good choice in FCN mode is to use
      output_stride=16 in order to increase the density of the computed features at
      small computational and memory overhead, cf. http://arxiv.org/abs/1606.00915.

      Args:
        inputs: A tensor of size [batch, height_in, width_in, channels].
        blocks: A list of length equal to the number of ResNet blocks. Each element
          is a resnet_utils.Block object describing the units in the block.
        num_classes: Number of predicted classes for classification tasks.
          If 0 or None, we return the features before the logit layer.
        is_training: whether batch_norm layers are in training mode.
        global_pool: If True, we perform global average pooling before computing the
          logits. Set to True for image classification, False for dense prediction.
        output_stride: If None, then the output will be computed at the nominal
          network stride. If output_stride is not None, it specifies the requested
          ratio of input to output spatial resolution.
        include_root_block: If True, include the initial convolution followed by
          max-pooling, if False excludes it. If excluded, `inputs` should be the
          results of an activation-less convolution.
        spatial_squeeze: if True, logits is of shape [B, C], if false logits is
            of shape [B, 1, 1, C], where B is batch_size and C is number of classes.
            To use this parameter, the input images must be smaller than 300x300
            pixels, in which case the output logit layer does not contain spatial
            information and can be removed.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.


      Returns:
        net: A rank-4 tensor of size [batch, height_out, width_out, channels_out].
          If global_pool is False, then height_out and width_out are reduced by a
          factor of output_stride compared to the respective height_in and width_in,
          else both height_out and width_out equal one. If num_classes is 0 or None,
          then net is the output of the last ResNet block, potentially after global
          average pooling. If num_classes is a non-zero integer, net contains the
          pre-softmax activations.
        end_points: A dictionary from components of the network to the corresponding
          activation.

      Raises:
        ValueError: If the target output_stride is not valid.
      """
      with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, self.bottleneck, resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
          with slim.arg_scope([slim.batch_norm], is_training=is_training):
            net = inputs
            if include_root_block:
              if output_stride is not None:
                if output_stride % 4 != 0:
                  raise ValueError('The output_stride needs to be a multiple of 4.')
                output_stride /= 4
              # We do not include batch normalization or activation functions in
              # conv1 because the first ResNet unit will perform these. Cf.
              # Appendix of [2].
              with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                net = resnet_utils.conv2d_same(net, 64, 7, stride=2, scope='conv1')
              net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
            net = resnet_utils.stack_blocks_dense(net, blocks, output_stride)
            # This is needed because the pre-activation variant does not have batch
            # normalization or activation functions in the residual unit output. See
            # Appendix of [2].
            net = slim.batch_norm(net, activation_fn=tf.nn.relu, scope='postnorm')
            # Convert end_points_collection into a dictionary of end_points.
            end_points = slim.utils.convert_collection_to_dict(end_points_collection)

            print(' --: ',net.shape)

            '''
            if global_pool:
              # Global average pooling.
              net = tf.reduce_mean(net, [1, 2], name='pool5', keep_dims=True)
              end_points['global_pool'] = net
            if num_classes:
              net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                normalizer_fn=None, scope='logits')
              end_points[sc.name + '/logits'] = net
              if spatial_squeeze:
                net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                end_points[sc.name + '/spatial_squeeze'] = net
              end_points['predictions'] = slim.softmax(net, scope='predictions')
            '''
            
            print(' --: ',net.shape)

            '''
            # Block 8: 1 conv (1*1), 1 conv (3*3)
            block = 'block12'
            with tf.variable_scope(block):
                net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
                net = self.pad2d(net, pad=(1, 1))
                net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
            print(block,net.shape)
            end_points[block] = net
            '''
            
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

            r = self.ssd_multibox_layer(end_points)
            return r
              
            #return net, end_points
    #resnet_v2.default_image_size = 224


    def resnet_v2_block(self, scope, base_depth, num_units, stride):
      """Helper function for creating a resnet_v2 bottleneck block.

      Args:
        scope: The scope of the block.
        base_depth: The depth of the bottleneck layer for each unit.
        num_units: The number of units in the block.
        stride: The stride of the block, implemented as a stride in the last unit.
          All other units have stride=1.

      Returns:
        A resnet_v2 bottleneck block.
      """
      return resnet_utils.Block(scope, self.bottleneck, [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': 1
      }] * (num_units - 1) + [{
          'depth': base_depth * 4,
          'depth_bottleneck': base_depth,
          'stride': stride
      }])
    #resnet_v2.default_image_size = 224


    def resnet_v2_50(self, inputs,
                     num_classes=None,
                     is_training=True,
                     global_pool=True,
                     output_stride=None,
                     spatial_squeeze=True,
                     reuse=None,
                     scope='resnet_v2_50'):
      """ResNet-50 model of [1]. See resnet_v2() for arg and return description."""
      blocks = [
          resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
          resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
          resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
      ]
      return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       reuse=reuse, scope=scope)
    #resnet_v2_50.default_image_size = resnet_v2.default_image_size


    def resnet_v2_101(self, inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='resnet_v2_101'):
      """ResNet-101 model of [1]. See resnet_v2() for arg and return description."""
      blocks = [
          resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
          resnet_v2_block('block3', base_depth=256, num_units=23, stride=2),
          resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
      ]
      return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       reuse=reuse, scope=scope)
    #resnet_v2_101.default_image_size = resnet_v2.default_image_size


    def resnet_v2_152(self, inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='resnet_v2_152'):
      """ResNet-152 model of [1]. See resnet_v2() for arg and return description."""
      blocks = [
          resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v2_block('block2', base_depth=128, num_units=8, stride=2),
          resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
          resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
      ]
      return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       reuse=reuse, scope=scope)
    #resnet_v2_152.default_image_size = resnet_v2.default_image_size


    def resnet_v2_200(self, inputs,
                      num_classes=None,
                      is_training=True,
                      global_pool=True,
                      output_stride=None,
                      spatial_squeeze=True,
                      reuse=None,
                      scope='resnet_v2_200'):
      """ResNet-200 model of [2]. See resnet_v2() for arg and return description."""
      blocks = [
          resnet_v2_block('block1', base_depth=64, num_units=3, stride=2),
          resnet_v2_block('block2', base_depth=128, num_units=24, stride=2),
          resnet_v2_block('block3', base_depth=256, num_units=36, stride=2),
          resnet_v2_block('block4', base_depth=512, num_units=3, stride=1),
      ]
      return resnet_v2(inputs, blocks, num_classes, is_training=is_training,
                       global_pool=global_pool, output_stride=output_stride,
                       include_root_block=True, spatial_squeeze=spatial_squeeze,
                       reuse=reuse, scope=scope)
    #resnet_v2_200.default_image_size = resnet_v2.default_image_size



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



















