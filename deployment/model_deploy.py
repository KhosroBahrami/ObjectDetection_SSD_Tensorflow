# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Deploy Slim models across multiple clones and replicas.



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops

slim = tf.contrib.slim


__all__ = ['create_clones',
           'optimize_clones'
           ]


# Namedtuple used to represent a clone during deployment.
Clone = collections.namedtuple('Clone',
                               ['outputs',  # Whatever model_fn() returned.
                                'scope',    # The scope used to create it.
                                'device',   # The device used to create.
                                ])

# Namedtuple used to represent a DeployedModel, returned by deploy().
DeployedModel = collections.namedtuple('DeployedModel',
                                       ['train_op',  # The `train_op`
                                        'summary_op',  # The `summary_op`
                                        'total_loss',  # The loss `Tensor`
                                        'clones',  # A list of `Clones` tuples.
                                        ])

# Default parameters for DeploymentConfig
_deployment_params = {'num_clones': 1,
                      'clone_on_cpu': False,
                      'fake_multiple_gpus': False,
                      'replica_id': 0,
                      'num_replicas': 1,
                      'num_ps_tasks': 0,
                      'worker_job_name': 'worker',
                      'ps_job_name': 'ps'}



def _gather_clone_loss(clone, num_clones, regularization_losses):
    """Gather the loss for a single clone.

    Args:
        clone: A Clone namedtuple.
        num_clones: The number of clones being deployed.
        regularization_losses: Possibly empty list of regularization_losses
            to add to the clone losses.

    Returns:
        A tensor for the total loss for the clone.  Can be None.
    """
    # The return value.
    sum_loss = None
    # Individual components of the loss that will need summaries.
    clone_loss = None
    regularization_loss = None
    # Compute and aggregate losses on the clone device.
    #with tf.device(clone.device):
    with tf.device(''):
        all_losses = []
        #clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, clone.scope)
        clone_losses = tf.get_collection(tf.GraphKeys.LOSSES, '')
        if clone_losses:
            clone_loss = tf.add_n(clone_losses, name='clone_loss')
            if num_clones > 1:
                clone_loss = tf.div(clone_loss, 1.0 * num_clones,
                                    name='scaled_clone_loss')
            all_losses.append(clone_loss)
        if regularization_losses:
            regularization_loss = tf.add_n(regularization_losses,
                                           name='regularization_loss')
            all_losses.append(regularization_loss)
        if all_losses:
            sum_loss = tf.add_n(all_losses)
    # Add the summaries out of the clone device block.
    if clone_loss is not None:
        tf.summary.scalar('clone_loss', clone_loss)
        # tf.summary.scalar(clone.scope + '/clone_loss', clone_loss)
    if regularization_loss is not None:
        tf.summary.scalar('regularization_loss', regularization_loss)
    return sum_loss


def _optimize_clone(optimizer, clone, num_clones, regularization_losses,
                                        **kwargs):
    """Compute losses and gradients for a single clone.

    Args:
        optimizer: A tf.Optimizer  object.
        clone: A Clone namedtuple.
        num_clones: The number of clones being deployed.
        regularization_losses: Possibly empty list of regularization_losses
            to add to the clone losses.
        **kwargs: Dict of kwarg to pass to compute_gradients().

    Returns:
        A tuple (clone_loss, clone_grads_and_vars).
            - clone_loss: A tensor for the total loss for the clone.  Can be None.
            - clone_grads_and_vars: List of (gradient, variable) for the clone.
                Can be empty.
    """
    sum_loss = _gather_clone_loss(clone, num_clones, regularization_losses)
    clone_grad = None
    if sum_loss is not None:
        #with tf.device(clone.device):
        with tf.device(''):
            clone_grad = optimizer.compute_gradients(sum_loss, **kwargs)
    return sum_loss, clone_grad


def optimize_clones(clones, optimizer,
                    regularization_losses=None,
                    **kwargs):
    """Compute clone losses and gradients for the given list of `Clones`.

    Note: The regularization_losses are added to the first clone losses.

    Args:
      clones: List of `Clones` created by `create_clones()`.
      optimizer: An `Optimizer` object.
      regularization_losses: Optional list of regularization losses. If None it
         will gather them from tf.GraphKeys.REGULARIZATION_LOSSES. Pass `[]` to
         exclude them.
      **kwargs: Optional list of keyword arguments to pass to `compute_gradients`.

    Returns:
      A tuple (total_loss, grads_and_vars).
        - total_loss: A Tensor containing the average of the clone losses
            including the regularization loss.
        - grads_and_vars: A List of tuples (gradient, variable) containing the
            sum of the gradients for each variable.

    """
    grads_and_vars = []
    clones_losses = []
    num_clones = len(clones)
    if regularization_losses is None:
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    for clone in clones:
        with tf.name_scope(''):
        #with tf.name_scope(clone.scope):
            clone_loss, clone_grad = _optimize_clone(
                    optimizer, clone, num_clones, regularization_losses, **kwargs)
            if clone_loss is not None:
                clones_losses.append(clone_loss)
                grads_and_vars.append(clone_grad)
            # Only use regularization_losses for the first clone
            regularization_losses = None
    # Compute the total_loss summing all the clones_losses.
    total_loss = tf.add_n(clones_losses, name='total_loss')
    # Sum the gradients accross clones.
    grads_and_vars = _sum_clones_gradients(grads_and_vars)
    return total_loss, grads_and_vars



def _sum_clones_gradients(clone_grads):
    """Calculate the sum gradient for each shared variable across all clones.

    This function assumes that the clone_grads has been scaled appropriately by
    1 / num_clones.

    Args:
      clone_grads: A List of List of tuples (gradient, variable), one list per
        `Clone`.

    Returns:
      List of tuples of (gradient, variable) where the gradient has been summed
        across all clones.
    """
    sum_grads = []
    for grad_and_vars in zip(*clone_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad_var0_clone0, var0), ... (grad_varN_cloneN, varN))
        grads = []
        var = grad_and_vars[0][1]
        for g, v in grad_and_vars:
            assert v == var
            if g is not None:
                grads.append(g)
        if grads:
            if len(grads) > 1:
                sum_grad = tf.add_n(grads, name=var.op.name + '/sum_grads')
            else:
                sum_grad = grads[0]
            sum_grads.append((sum_grad, var))
    return sum_grads








