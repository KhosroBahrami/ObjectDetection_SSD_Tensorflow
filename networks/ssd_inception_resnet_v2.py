
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





class SSD_network(object):

    def __init__(self): 
        self.img_shape=(FLAGS.image_size, FLAGS.image_size)
        self.num_classes=FLAGS.num_classes
        self.no_annotation_label=FLAGS.num_classes
        # InceptionResnet_V2
        self.feat_layers=['Conv2d_4a_3x3', 'MaxPool_5a_3x3', 'Mixed_6a', 'Mixed_7a', 'block13', 'block14']
        self.feat_shapes=[(71, 71), (35, 35), (17, 17), (8, 8), (4, 4), (2, 2)]
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



    def block35(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
      """Builds the 35x35 resnet block."""
      with tf.variable_scope(scope, 'Block35', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 32, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 32, 3, scope='Conv2d_0b_3x3')
        with tf.variable_scope('Branch_2'):
          tower_conv2_0 = slim.conv2d(net, 32, 1, scope='Conv2d_0a_1x1')
          tower_conv2_1 = slim.conv2d(tower_conv2_0, 48, 3, scope='Conv2d_0b_3x3')
          tower_conv2_2 = slim.conv2d(tower_conv2_1, 64, 3, scope='Conv2d_0c_3x3')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_1, tower_conv2_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')
        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
          # Use clip_by_value to simulate bandpass activation.
          scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
          net = activation_fn(net)
      return net


    def block17(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
      """Builds the 17x17 resnet block."""
      with tf.variable_scope(scope, 'Block17', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 128, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 160, [1, 7],
                                      scope='Conv2d_0b_1x7')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 192, [7, 1],
                                      scope='Conv2d_0c_7x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
          # Use clip_by_value to simulate bandpass activation.
          scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
          net = activation_fn(net)
      return net


    def block8(self, net, scale=1.0, activation_fn=tf.nn.relu, scope=None, reuse=None):
      """Builds the 8x8 resnet block."""
      with tf.variable_scope(scope, 'Block8', [net], reuse=reuse):
        with tf.variable_scope('Branch_0'):
          tower_conv = slim.conv2d(net, 192, 1, scope='Conv2d_1x1')
        with tf.variable_scope('Branch_1'):
          tower_conv1_0 = slim.conv2d(net, 192, 1, scope='Conv2d_0a_1x1')
          tower_conv1_1 = slim.conv2d(tower_conv1_0, 224, [1, 3],
                                      scope='Conv2d_0b_1x3')
          tower_conv1_2 = slim.conv2d(tower_conv1_1, 256, [3, 1],
                                      scope='Conv2d_0c_3x1')
        mixed = tf.concat(axis=3, values=[tower_conv, tower_conv1_2])
        up = slim.conv2d(mixed, net.get_shape()[3], 1, normalizer_fn=None,
                         activation_fn=None, scope='Conv2d_1x1')

        scaled_up = up * scale
        if activation_fn == tf.nn.relu6:
          # Use clip_by_value to simulate bandpass activation.
          scaled_up = tf.clip_by_value(scaled_up, -6.0, 6.0)

        net += scaled_up
        if activation_fn:
          net = activation_fn(net)
      return net


    def inception_resnet_v2_base(self, inputs,
                                 final_endpoint='Conv2d_7b_1x1',
                                 output_stride=16,
                                 align_feature_maps=False,
                                 scope=None,
                                 activation_fn=tf.nn.relu):
      """Inception model from  http://arxiv.org/abs/1602.07261.

      Constructs an Inception Resnet v2 network from inputs to the given final
      endpoint. This method can construct the network up to the final inception
      block Conv2d_7b_1x1.

      Args:
        inputs: a tensor of size [batch_size, height, width, channels].
        final_endpoint: specifies the endpoint to construct the network up to. It
          can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
          'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
          'Mixed_5b', 'Mixed_6a', 'PreAuxLogits', 'Mixed_7a', 'Conv2d_7b_1x1']
        output_stride: A scalar that specifies the requested ratio of input to
          output spatial resolution. Only supports 8 and 16.
        align_feature_maps: When true, changes all the VALID paddings in the network
          to SAME padding so that the feature maps are aligned.
        scope: Optional variable_scope.
        activation_fn: Activation function for block scopes.

      Returns:
        tensor_out: output tensor corresponding to the final_endpoint.
        end_points: a set of activations for external use, for example summaries or
                    losses.

      Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
          or if the output_stride is not 8 or 16, or if the output_stride is 8 and
          we request an end point after 'PreAuxLogits'.
      """
      if output_stride != 8 and output_stride != 16:
        raise ValueError('output_stride must be 8 or 16.')

      padding = 'SAME' if align_feature_maps else 'VALID'

      end_points = {}

      def add_and_check_final(name, net):
        end_points[name] = net
        return name == final_endpoint

      with tf.variable_scope(scope, 'InceptionResnetV2', [inputs]):
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
          # 149 x 149 x 32
          net = slim.conv2d(inputs, 32, 3, stride=2, padding=padding, scope='Conv2d_1a_3x3')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('Conv2d_1a_3x3', net): return net, end_points

          # 147 x 147 x 32
          net = slim.conv2d(net, 32, 3, padding=padding, scope='Conv2d_2a_3x3')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('Conv2d_2a_3x3', net): return net, end_points
          # 147 x 147 x 64
          net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('Conv2d_2b_3x3', net): return net, end_points
          # 73 x 73 x 64
          net = slim.max_pool2d(net, 3, stride=2, padding=padding, scope='MaxPool_3a_3x3')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('MaxPool_3a_3x3', net): return net, end_points
          # 73 x 73 x 80
          net = slim.conv2d(net, 80, 1, padding=padding, scope='Conv2d_3b_1x1')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('Conv2d_3b_1x1', net): return net, end_points
          # 71 x 71 x 192
          net = slim.conv2d(net, 192, 3, padding=padding, scope='Conv2d_4a_3x3')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('Conv2d_4a_3x3', net): return net, end_points
          # 35 x 35 x 192
          net = slim.max_pool2d(net, 3, stride=2, padding=padding, scope='MaxPool_5a_3x3')
          print(net) 
          print(' : ',net.shape)
          if add_and_check_final('MaxPool_5a_3x3', net): return net, end_points

          # 35 x 35 x 320
          with tf.variable_scope('Mixed_5b'):
            with tf.variable_scope('Branch_0'):
              tower_conv = slim.conv2d(net, 96, 1, scope='Conv2d_1x1')
            with tf.variable_scope('Branch_1'):
              tower_conv1_0 = slim.conv2d(net, 48, 1, scope='Conv2d_0a_1x1')
              tower_conv1_1 = slim.conv2d(tower_conv1_0, 64, 5, scope='Conv2d_0b_5x5')
            with tf.variable_scope('Branch_2'):
              tower_conv2_0 = slim.conv2d(net, 64, 1, scope='Conv2d_0a_1x1')
              tower_conv2_1 = slim.conv2d(tower_conv2_0, 96, 3, scope='Conv2d_0b_3x3')
              tower_conv2_2 = slim.conv2d(tower_conv2_1, 96, 3, scope='Conv2d_0c_3x3')
            with tf.variable_scope('Branch_3'):
              tower_pool = slim.avg_pool2d(net, 3, stride=1, padding='SAME', scope='AvgPool_0a_3x3')
              tower_pool_1 = slim.conv2d(tower_pool, 64, 1, scope='Conv2d_0b_1x1')
            net = tf.concat([tower_conv, tower_conv1_1, tower_conv2_2, tower_pool_1], 3)

          if add_and_check_final('Mixed_5b', net): return net, end_points
          # TODO(alemi): Register intermediate endpoints
          net = slim.repeat(net, 10, self.block35, scale=0.17, activation_fn=activation_fn)
          print(net) 
          print(' : ',net.shape)

          # 17 x 17 x 1088 if output_stride == 8,
          # 33 x 33 x 1088 if output_stride == 16
          use_atrous = output_stride == 8

          with tf.variable_scope('Mixed_6a'):
            with tf.variable_scope('Branch_0'):
              tower_conv = slim.conv2d(net, 384, 3, stride=1 if use_atrous else 2,
                                       padding=padding,
                                       scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              tower_conv1_0 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
              tower_conv1_1 = slim.conv2d(tower_conv1_0, 256, 3,
                                          scope='Conv2d_0b_3x3')
              tower_conv1_2 = slim.conv2d(tower_conv1_1, 384, 3,
                                          stride=1 if use_atrous else 2,
                                          padding=padding,
                                          scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              tower_pool = slim.max_pool2d(net, 3, stride=1 if use_atrous else 2,
                                           padding=padding,
                                           scope='MaxPool_1a_3x3')
            net = tf.concat([tower_conv, tower_conv1_2, tower_pool], 3)
            print(net) 
            print(' : ',net.shape)

          if add_and_check_final('Mixed_6a', net): return net, end_points

          # TODO(alemi): register intermediate endpoints
          with slim.arg_scope([slim.conv2d], rate=2 if use_atrous else 1):
            net = slim.repeat(net, 20, self.block17, scale=0.10, activation_fn=activation_fn)
          if add_and_check_final('PreAuxLogits', net): return net, end_points

          if output_stride == 8:
            # TODO(gpapan): Properly support output_stride for the rest of the net.
            raise ValueError('output_stride==8 is only supported up to the '
                             'PreAuxlogits end_point for now.')

          # 8 x 8 x 2080
          with tf.variable_scope('Mixed_7a'):
            with tf.variable_scope('Branch_0'):
              tower_conv = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
              tower_conv_1 = slim.conv2d(tower_conv, 384, 3, stride=2,
                                         padding=padding,
                                         scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_1'):
              tower_conv1 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
              tower_conv1_1 = slim.conv2d(tower_conv1, 288, 3, stride=2,
                                          padding=padding,
                                          scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_2'):
              tower_conv2 = slim.conv2d(net, 256, 1, scope='Conv2d_0a_1x1')
              tower_conv2_1 = slim.conv2d(tower_conv2, 288, 3,
                                          scope='Conv2d_0b_3x3')
              tower_conv2_2 = slim.conv2d(tower_conv2_1, 320, 3, stride=2,
                                          padding=padding,
                                          scope='Conv2d_1a_3x3')
            with tf.variable_scope('Branch_3'):
              tower_pool = slim.max_pool2d(net, 3, stride=2,
                                           padding=padding,
                                           scope='MaxPool_1a_3x3')
            net = tf.concat([tower_conv_1, tower_conv1_1, tower_conv2_2, tower_pool], 3)
            print(net) 
            print(' : ',net.shape)

          if add_and_check_final('Mixed_7a', net): return net, end_points

          # TODO(alemi): register intermediate endpoints
          net = slim.repeat(net, 9, self.block8, scale=0.20, activation_fn=activation_fn)
          net = self.block8(net, activation_fn=None)

          # 8 x 8 x 1536
          net = slim.conv2d(net, 1536, 1, scope='Conv2d_7b_1x1')
          if add_and_check_final('Conv2d_7b_1x1', net): return net, end_points

        raise ValueError('final_endpoint (%s) not recognized', final_endpoint)





    #def inception_resnet_v2(self, inputs, num_classes=1001, is_training=True,
    def net(self, inputs, num_classes=1001, is_training=True,
                            dropout_keep_prob=0.8,
                            reuse=None,
                            scope='InceptionResnetV2',
                            create_aux_logits=True,
                            activation_fn=tf.nn.relu):
      """Creates the Inception Resnet V2 model.

      Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
          Dimension batch_size may be undefined. If create_aux_logits is false,
          also height and width may be undefined.
        num_classes: number of predicted classes. If 0 or None, the logits layer
          is omitted and the input features to the logits layer (before  dropout)
          are returned instead.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        reuse: whether or not the network and its variables should be reused. To be
          able to reuse 'scope' must be given.
        scope: Optional variable_scope.
        create_aux_logits: Whether to include the auxilliary logits.
        activation_fn: Activation function for conv2d.

      Returns:
        net: the output of the logits layer (if num_classes is a non-zero integer),
          or the non-dropped-out input to the logits layer (if num_classes is 0 or
          None).
        end_points: the set of end_points from the inception model.
      """
      end_points = {}

      with tf.variable_scope(scope, 'InceptionResnetV2', [inputs], reuse=reuse) as scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):

          net, end_points = self.inception_resnet_v2_base(inputs, scope=scope, activation_fn=activation_fn)

          print(' --: ',net.shape)

          ''' 
          if create_aux_logits and num_classes:
            with tf.variable_scope('AuxLogits'):
              aux = end_points['PreAuxLogits']
              aux = slim.avg_pool2d(aux, 5, stride=3, padding='VALID', scope='Conv2d_1a_3x3')
              aux = slim.conv2d(aux, 128, 1, scope='Conv2d_1b_1x1')
              aux = slim.conv2d(aux, 768, aux.get_shape()[1:3], padding='VALID', scope='Conv2d_2a_5x5')
              aux = slim.flatten(aux)
              aux = slim.fully_connected(aux, num_classes, activation_fn=None, scope='Logits')
              end_points['AuxLogits'] = aux

          with tf.variable_scope('Logits'):
            # TODO(sguada,arnoegw): Consider adding a parameter global_pool which
            # can be set to False to disable pooling here (as in resnet_*()).
            kernel_size = net.get_shape()[1:3]
            if kernel_size.is_fully_defined():
              net = slim.avg_pool2d(net, kernel_size, padding='VALID', scope='AvgPool_1a_8x8')
            else:
              net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='global_pool')
            end_points['global_pool'] = net
            if not num_classes:
              return net, end_points
            net = slim.flatten(net)
            net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope='Dropout')
            end_points['PreLogitsFlatten'] = net
            logits = slim.fully_connected(net, num_classes, activation_fn=None, scope='Logits')
            end_points['Logits'] = logits
            end_points['Predictions'] = tf.nn.softmax(logits, name='Predictions')
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
        '''
        # Block 11: 1 conv (1*1), 1 conv (3*3) 
        block = 'block15'
        with tf.variable_scope(block):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        print(block,net.shape)
        end_points[block] = net
        '''

        r = self.ssd_multibox_layer(end_points)
        return r


        #return logits, end_points
    #inception_resnet_v2.default_image_size = 299




    def inception_resnet_v2_arg_scope(self, 
        weight_decay=0.00004,
        batch_norm_decay=0.9997,
        batch_norm_epsilon=0.001,
        activation_fn=tf.nn.relu,
        batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS,
        batch_norm_scale=False):
      """Returns the scope with the default parameters for inception_resnet_v2.

      Args:
        weight_decay: the weight decay for weights variables.
        batch_norm_decay: decay for the moving average of batch_norm momentums.
        batch_norm_epsilon: small float added to variance to avoid dividing by zero.
        activation_fn: Activation function for conv2d.
        batch_norm_updates_collections: Collection for the update ops for
          batch norm.
        batch_norm_scale: If True, uses an explicit `gamma` multiplier to scale the
          activations in the batch normalization layer.

      Returns:
        a arg_scope with the parameters needed for inception_resnet_v2.
      """
      # Set weight_decay for weights in conv2d and fully_connected layers.
      with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          biases_regularizer=slim.l2_regularizer(weight_decay)):

        batch_norm_params = {
            'decay': batch_norm_decay,
            'epsilon': batch_norm_epsilon,
            'updates_collections': batch_norm_updates_collections,
            'fused': None,  # Use fused batch norm if possible.
            'scale': batch_norm_scale,
        }
        # Set activation_fn and parameters for batch_norm.
        with slim.arg_scope([slim.conv2d], activation_fn=activation_fn,
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params) as scope:
          return scope




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



















