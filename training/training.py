
# Generic training script that trains a SSD model using a given dataset

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from deployment import model_deploy
from networks import network_factory
DATA_FORMAT = 'NHWC'
from configs.config_train import *
import tensorflow.contrib.slim.nets 
from configs.config_general import backbone_network


class Training(object):

        def __init__(self):
                a=1
                
        
        def training(self, network, b_image, b_gclasses, b_glocalisations, b_gscores, batch_queue, batch_shape, dataset, global_step):

                gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.train_gpu_memory_fraction)
                config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
            
                summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

                b_image, b_gclasses, b_glocalisations, b_gscores = \
                        network.reshape_list(batch_queue.dequeue(), batch_shape)

                
                # Construct SSD network
                arg_scope = network.arg_scope(weight_decay=FLAGS.train_weight_decay, data_format=DATA_FORMAT)
                # Network inference               
                with slim.arg_scope(arg_scope):

                        if backbone_network == 'ssd_vgg_300': # VGG
                                predictions, localisations, logits, outputs = network.net(b_image, training=True)

                        elif backbone_network == 'ssd_mobilenet_v1': # MobilenetV1
                                predictions, localisations, logits, outputs = network.net(b_image, is_training=True)

                        elif backbone_network == 'ssd_mobilenet_v2': # MobilenetV2
                                predictions, localisations, logits, outputs = network.net(b_image, is_training=True)

                        elif backbone_network == 'ssd_resnet_v1': # ResnetV1
                                block = network.resnet_v1_block
                                blocks = [
                                block('block1', base_depth=1, num_units=2, stride=2),
                                block('block2', base_depth=2, num_units=2, stride=2),
                                block('block3', base_depth=4, num_units=2, stride=2),
                                block('block4', base_depth=8, num_units=2, stride=1),]
                                predictions, localisations, logits, outputs = network.net(b_image, blocks, is_training=True)

                        elif backbone_network == 'ssd_resnet_v2': # ResnetV2
                                block = network.resnet_v2_block
                                blocks = [
                                block('block1', base_depth=1, num_units=2, stride=2),
                                block('block2', base_depth=2, num_units=2, stride=2),
                                block('block3', base_depth=4, num_units=2, stride=2),
                                block('block4', base_depth=8, num_units=2, stride=1),]
                                predictions, localisations, logits, outputs = network.net(b_image, blocks, is_training=True)

                        elif backbone_network == 'ssd_inception_v4': # InceptionV4
                                predictions, localisations, logits, outputs = network.net(b_image, is_training=True)

                        elif backbone_network == 'ssd_inception_resnet_v2': # InceptionResnetV2
                                predictions, localisations, logits, outputs = network.net(b_image, is_training=True)



                  # Add loss to network
                network.losses(logits, localisations, b_gclasses, b_glocalisations, b_gscores,
                                   match_threshold=FLAGS.train_match_threshold, negative_ratio=FLAGS.train_negative_ratio,
                                   alpha=FLAGS.train_loss_alpha, label_smoothing=FLAGS.train_label_smoothing)
                
                clones=[] 
                clones.append((outputs, '', ''))
                first_clone_scope=''
                
                # Gather update_ops from the first clone. 
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)
               
                # Add summaries for end_points.
                end_points = outputs
              
                for end_point in end_points:
                    x = end_points[end_point]
                    summaries.add(tf.summary.histogram('activations/' + end_point, x))
                    summaries.add(tf.summary.scalar('sparsity/' + end_point, tf.nn.zero_fraction(x)))
                # Add summaries for losses 
                for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
                    summaries.add(tf.summary.scalar(loss.op.name, loss))
                # Add summaries for extra loss    
                for loss in tf.get_collection('EXTRA_LOSSES', first_clone_scope):
                    summaries.add(tf.summary.scalar(loss.op.name, loss))
                # Add summaries for variables.
                for variable in slim.get_model_variables():
                    summaries.add(tf.summary.histogram(variable.op.name, variable))

                moving_average_variables, variable_averages = None, None

                # Configure the optimization procedure.
                learning_rate = self.configure_learning_rate(dataset.num_samples, global_step)
                optimizer = self.configure_optimizer(FLAGS, learning_rate)
                summaries.add(tf.summary.scalar('learning_rate', learning_rate))

                # Variables to train.
                variables_to_train = tf.trainable_variables()
                
                # and returns a train_tensor and summary_op
                total_loss, clones_gradients = model_deploy.optimize_clones(
                    clones, optimizer, var_list=variables_to_train)
                
                # Add summaries for total_loss 
                summaries.add(tf.summary.scalar('total_loss', total_loss))

                # Create gradient updates
                grad_updates = optimizer.apply_gradients(clones_gradients, global_step=global_step)
                update_ops.append(grad_updates)
                update_op = tf.group(*update_ops)
                train_tensor = control_flow_ops.with_dependencies([update_op], total_loss, name='train_op')
                
                # Add  summaries from the first clone
                summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES, first_clone_scope))
                # Merge all summaries together.
                summary_op = tf.summary.merge(list(summaries), name='summary_op')
            
                
                saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=1.0, write_version=2, pad_step_number=False)
                # training function of slim 
                slim.learning.train(train_tensor, logdir=FLAGS.train_dir, master='', is_chief=True,
                    init_fn=self.get_init_fn(FLAGS), summary_op=summary_op,
                    number_of_steps=FLAGS.train_max_number_of_steps,
                    log_every_n_steps=FLAGS.train_log_every_n_steps,
                    save_summaries_secs=FLAGS.train_save_summaries_secs, saver=saver,
                    save_interval_secs=FLAGS.train_save_interval_secs, session_config=config,
                    sync_optimizer=None)
              




        # Configuration of the learning rate.
        # Inputs:
        #      num_samples_per_epoch: The number of samples in each epoch of training.
        #      global_step: The global_step tensor.
        # Outputs:
        #      A `Tensor` representing the learning rate.
        def configure_learning_rate(self, num_samples_per_epoch, global_step):

            decay_steps = int(num_samples_per_epoch / FLAGS.train_batch_size * FLAGS.train_num_epochs_per_decay)

            if FLAGS.train_learning_rate_decay_type == 'exponential':
                return tf.train.exponential_decay(FLAGS.train_learning_rate, global_step, decay_steps,
                                                  FLAGS.train_learning_rate_decay_factor, staircase=True,
                                                  name='exponential_decay_learning_rate')
            elif FLAGS.train_learning_rate_decay_type == 'fixed':
                return tf.constant(flags.train_learning_rate, name='fixed_learning_rate')
            elif FLAGS.train_learning_rate_decay_type == 'polynomial':
                return tf.train.polynomial_decay(FLAGS.train_learning_rate, global_step, decay_steps,
                                                 FLAGS.train_end_learning_rate, power=1.0, cycle=False,
                                                 name='polynomial_decay_learning_rate')



        # Configures the optimizer used for training.
        # Inputs:
        #      learning_rate: A scalar or `Tensor` learning rate.
        # Outputs:
        #      An instance of an optimizer.
        def configure_optimizer(self, flags, learning_rate):

            if flags.train_optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(learning_rate, rho=flags.adadelta_rho,
                    epsilon=flags.opt_epsilon)
            elif flags.train_optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(learning_rate,
                    initial_accumulator_value=flags.adagrad_initial_accumulator_value)
            elif flags.train_optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate, beta1=flags.train_adam_beta1,
                    beta2=flags.train_adam_beta2, epsilon=flags.train_opt_epsilon)
            elif flags.train_optimizer == 'ftrl':
                optimizer = tf.train.FtrlOptimizer(learning_rate,
                    learning_rate_power=flags.train_ftrl_learning_rate_power,
                    initial_accumulator_value=flags.train_ftrl_initial_accumulator_value,
                    l1_regularization_strength=flags.train_ftrl_l1,
                    l2_regularization_strength=flags.train_ftrl_l2)
            elif flags.train_optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=flags.train_momentum,
                    name='Momentum')
            elif flags.train_optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=flags.train_rmsprop_decay,
                    momentum=flags.train_rmsprop_momentum, epsilon=flags.opt_epsilon)
            elif flags.train_optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            return optimizer



        def add_variables_summaries(self, learning_rate):
            summaries = []
            for variable in slim.get_model_variables():
                summaries.append(tf.summary.histogram(variable.op.name, variable))
            summaries.append(tf.summary.scalar('training/Learning Rate', learning_rate))
            return summaries



  
        # Returns a function run by the chief worker to warm-start the training.
        #    Note that the init_fn is only run when initializing the model during the very
        #    first global step.
        #    Returns:
        #      An init function run by the supervisor.
        def get_init_fn(self, flags):
            if flags.train_checkpoint_path is None:
                return None
            if tf.train.latest_checkpoint(flags.train_dir):
                tf.logging.info('Ignoring --checkpoint_path because a checkpoint already exists in %s'
                    % flags.train_dir)
                return None
        
            exclusions = []
            if flags.train_checkpoint_exclude_scopes:
                exclusions = [scope.strip()
                              for scope in flags.train_checkpoint_exclude_scopes.split(',')]
                print('\n exclusions:')
                print(exclusions)
                print('\n\n')
                
            variables_to_restore = []
            for var in slim.get_model_variables():
                excluded = False
                for exclusion in exclusions:
                    if var.op.name.startswith(exclusion):
                        excluded = True
                        break
                if not excluded:
                    variables_to_restore.append(var)

            # Change model scope if necessary.
            if flags.train_checkpoint_model_scope is not None:
                variables_to_restore = \
                    {var.op.name.replace(flags.model_name, flags.train_checkpoint_model_scope): var
                     for var in variables_to_restore}
                    
            if tf.gfile.IsDirectory(flags.train_checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(flags.train_checkpoint_path)
            else:
                checkpoint_path = flags.train_checkpoint_path
            tf.logging.info('Fine-tuning from %s. Ignoring missing vars: %s' % (checkpoint_path, flags.train_ignore_missing_vars))

            return slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore,
                ignore_missing_vars=flags.train_ignore_missing_vars)




        # Returns a list of variables to train.
        #    Returns:
        #      A list of variables to train by the optimizer.
        def get_variables_to_train(self, flags):
            if flags.train_trainable_scopes is None:
                return tf.trainable_variables()
            else:
                scopes = [scope.strip() for scope in flags.train_trainable_scopes.split(',')]

                print('\n scopes:')
                print(scopes)
                print('\n')
                
            variables_to_train = []
            for scope in scopes:
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                variables_to_train.extend(variables)
            return variables_to_train







