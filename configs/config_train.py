
# Config of training
import tensorflow as tf
from configs.config_general import training_mode
from configs.config_general import backbone_network
slim = tf.contrib.slim



# Fine tuning on existing SSD model (VGG)
if training_mode == 'finetune_on_existing_ssd_model': 
    tf.app.flags.DEFINE_string('train_dir', './tmp/tfmodel/', 'Directory of checkpoints.')
    tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/ssd_vgg/ssd_300_vgg.ckpt', 'path of checkpoint.')
    tf.app.flags.DEFINE_float('train_learning_rate', 0.001, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('train_learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
    tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes', None,
        'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
    tf.app.flags.DEFINE_string('train_checkpoint_model_scope', None,
        'Model scope in the checkpoint. None if the same as the trained model.')
    tf.app.flags.DEFINE_string('train_trainable_scopes', None,
        'Comma-separated list of scopes to filter the set of variables to train.By default, None would train all the variables.')


# Fine tuning on new SSD model (based on other baseline networks, such as MobileNet, ResNet, ...)
elif training_mode == 'finetune_on_new_ssd_model_step1': 

    tf.app.flags.DEFINE_string('train_dir', './tmp/tfmodel/', 'Directory of checkpoints.')

    if backbone_network == 'ssd_mobilenet_v1':
        tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/mobilenet_v1/mobilenet_v1_1.0_224.ckpt', 'path of checkpoint.')
    if backbone_network == 'ssd_mobilenet_v2':
        tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/mobilenet_v2/mobilenet_v2_1.0_224.ckpt', 'path of checkpoint.')
    if backbone_network == 'ssd_resnet_v1':
        tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/resnet_v1/resnet_v1_152.ckpt', 'path of checkpoint.')
    if backbone_network == 'ssd_resnet_v2':
        tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/resnet_v2/resnet_v2_152.ckpt', 'path of checkpoint.')
    if backbone_network == 'ssd_inception_v4':
        tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/inception_v4/inception_v4.ckpt', 'path of checkpoint.')
    if backbone_network == 'ssd_inception_resnet_v2':
        tf.app.flags.DEFINE_string('train_checkpoint_path', './checkpoints/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt', 'path of checkpoint.')


    tf.app.flags.DEFINE_float('train_learning_rate', 0.001, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('train_learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
    tf.app.flags.DEFINE_string('train_checkpoint_model_scope', 'MobilenetV2',
            'Model scope in the checkpoint. None if the same as the trained model.')

    if backbone_network == 'ssd_mobilenet_v1': # MobilenetV1
        tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes',
                                   'MobilenetV1/Conv2d_6_depthwise,'
                                   'MobilenetV1/Conv2d_6_pointwise,'
                                   'MobilenetV1/Conv2d_7_depthwise,'
                                   'MobilenetV1/Conv2d_7_pointwise,'
                                   'MobilenetV1/Conv2d_8_depthwise,'
                                   'MobilenetV1/Conv2d_8_pointwise,'
                                   'MobilenetV1/Conv2d_9_depthwise,'
                                   'MobilenetV1/Conv2d_9_pointwise,'
                                   'MobilenetV1/Conv2d_10_depthwise,'
                                   'MobilenetV1/Conv2d_10_pointwise,'
                                   'MobilenetV1/Conv2d_11_depthwise,'
                                   'MobilenetV1/Conv2d_11_pointwise,'
                                   'MobilenetV1/Conv2d_5_depthwise_box,'
                                   'MobilenetV1/Conv2d_5_pointwise_box,'
                                   'MobilenetV1/Conv2d_11_depthwise_box,'
                                   'MobilenetV1/Conv2d_11_pointwise_box,'
                                   'MobilenetV1/block12_box,'
                                   'MobilenetV1/block13_box,'
                                   'MobilenetV1/block14_box,'
                                   'MobilenetV1/block15_box',
                                   'list of variables to exclude when restoring from a checkpoint.')
        tf.app.flags.DEFINE_string('train_trainable_scopes',
                                   'MobilenetV1/Conv2d_6_depthwise,'
                                   'MobilenetV1/Conv2d_6_pointwise,'
                                   'MobilenetV1/Conv2d_7_depthwise,'
                                   'MobilenetV1/Conv2d_7_pointwise,'
                                   'MobilenetV1/Conv2d_8_depthwise,'
                                   'MobilenetV1/Conv2d_8_pointwise,'
                                   'MobilenetV1/Conv2d_9_depthwise,'
                                   'MobilenetV1/Conv2d_9_pointwise,'
                                   'MobilenetV1/Conv2d_10_depthwise,'
                                   'MobilenetV1/Conv2d_10_pointwise,'
                                   'MobilenetV1/Conv2d_11_depthwise,'
                                   'MobilenetV1/Conv2d_11_pointwise,'
                                   'MobilenetV1/Conv2d_5_depthwise_box,'
                                   'MobilenetV1/Conv2d_5_pointwise_box,'
                                   'MobilenetV1/Conv2d_11_depthwise_box,'
                                   'MobilenetV1/Conv2d_11_pointwise_box,'
                                   'MobilenetV1/block12_box,'
                                   'MobilenetV1/block13_box,'
                                   'MobilenetV1/block14_box,'
                                   'MobilenetV1/block15_box',
            'list of scopes to filter the set of variables to train.By default, None would train all the variables.')


    if backbone_network == 'ssd_mobilenet_v2': #MobilenetV2
        tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes',
                                   'MobilenetV2/Logits,'
                                   'MobilenetV2/expanded_conv_6,'
                                   'MobilenetV2/expanded_conv_7,'
                                   'MobilenetV2/expanded_conv_8,'
                                   'MobilenetV2/expanded_conv_9,'
                                   'MobilenetV2/expanded_conv_10,'
                                   'MobilenetV2/expanded_conv_11,'
                                   'MobilenetV2/expanded_conv_12,'
                                   'MobilenetV2/expanded_conv_13,'
                                   'MobilenetV2/expanded_conv_14,'
                                   'MobilenetV2/expanded_conv_15,'
                                   'MobilenetV2/expanded_conv_16,'
                                   'MobilenetV2/layer_7_box,'
                                   'MobilenetV2/layer_14_box,'
                                   'MobilenetV2/block12_box,'
                                   'MobilenetV2/block13_box,'
                                   'MobilenetV2/block14_box,'
                                   'MobilenetV2/block15_box',
                                   'list of variables to exclude when restoring from a checkpoint.')
        tf.app.flags.DEFINE_string('train_trainable_scopes',
                                   'MobilenetV2/Logits,'
                                   'MobilenetV2/expanded_conv_6,'
                                   'MobilenetV2/expanded_conv_7,'
                                   'MobilenetV2/expanded_conv_8,'
                                   'MobilenetV2/expanded_conv_9,'
                                   'MobilenetV2/expanded_conv_10,'
                                   'MobilenetV2/expanded_conv_11,'
                                   'MobilenetV2/expanded_conv_12,'
                                   'MobilenetV2/expanded_conv_13,'
                                   'MobilenetV2/expanded_conv_14,'
                                   'MobilenetV2/expanded_conv_15,'
                                   'MobilenetV2/expanded_conv_16,'
                                   'MobilenetV2/layer_7_box,'
                                   'MobilenetV2/layer_14_box,'
                                   'MobilenetV2/block12_box,'
                                   'MobilenetV2/block13_box,'
                                   'MobilenetV2/block14_box,'
                                   'MobilenetV2/block15_box',
            'list of scopes to filter the set of variables to train.By default, None would train all the variables.')


    if backbone_network == 'ssd_resnet_v1': #ResnetV1
        tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes',
                                   'resnet_v1_152/block1,'
                                   'resnet_v1_152/block2,'
                                   'resnet_v1_152/block3,'
                                   'resnet_v1_152/block4,'
                                   'resnet_v1_152/conv1,'
                                   'resnet_v1_152/logits',
                                   'list of variables to exclude when restoring from a checkpoint.')
        tf.app.flags.DEFINE_string('train_trainable_scopes',
                                   'resnet_v1_152/block1,'
                                   'resnet_v1_152/block2,'
                                   'resnet_v1_152/block3,'
                                   'resnet_v1_152/block4,'
                                   'resnet_v1_152/conv1,'
                                   'resnet_v1_152/logits',
            'list of scopes to filter the set of variables to train.By default, None would train all the variables.')

    if backbone_network == 'ssd_resnet_v2': #ResnetV2
        tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes',
                                   'resnet_v2_152/block1,'
                                   'resnet_v2_152/block2,'
                                   'resnet_v2_152/block3,'
                                   'resnet_v2_152/block4,'
                                   'resnet_v2_152/conv1,'
                                   'resnet_v2_152/logits',
                                   'list of variables to exclude when restoring from a checkpoint.')
        tf.app.flags.DEFINE_string('train_trainable_scopes',
                                   'resnet_v2_152/block1,'
                                   'resnet_v2_152/block2,'
                                   'resnet_v2_152/block3,'
                                   'resnet_v2_152/block4,'
                                   'resnet_v2_152/conv1,'
                                   'resnet_v2_152/logits',
            'list of scopes to filter the set of variables to train.By default, None would train all the variables.')

    if backbone_network == 'ssd_inception_v4': #InceptionV4
        tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes',
                                   'InceptionV4/Mixed_7a,'
                                   'InceptionV4/Mixed_7b,'
                                   'InceptionV4/Mixed_7c,'
                                   'InceptionV4/Mixed_7d,'
                                   'InceptionV4/Logits',
                                   'list of variables to exclude when restoring from a checkpoint.')
        tf.app.flags.DEFINE_string('train_trainable_scopes',
                                   'InceptionV4/Mixed_7a,'
                                   'InceptionV4/Mixed_7b,'
                                   'InceptionV4/Mixed_7c,'
                                   'InceptionV4/Mixed_7d,'
                                   'InceptionV4/Logits',
            'list of scopes to filter the set of variables to train.By default, None would train all the variables.')

    if backbone_network == 'ssd_inception_resnet_v2': #InceptionResnetV2
        tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes',
                                   'InceptionResnetV2/Repeat_1,'
                                   'InceptionResnetV2/Repeat_2',
                                   'list of variables to exclude when restoring from a checkpoint.')
        tf.app.flags.DEFINE_string('train_trainable_scopes',
                                   'InceptionResnetV2/Repeat_1,'
                                   'InceptionResnetV2/Repeat_2',
            'list of scopes to filter the set of variables to train.By default, None would train all the variables.')


# Fine tuning on new SSD model (based on other baseline networks, such as ResNet, ...)
elif training_mode == 'finetune_on_new_ssd_model_step2': 
    tf.app.flags.DEFINE_string('train_dir', './tmp/tfmodel_finetune/', 'Directory of checkpoints.')
    tf.app.flags.DEFINE_string('train_checkpoint_path', './tmp/tfmodel/model.ckpt', 'path of checkpoint.')
    tf.app.flags.DEFINE_float('train_learning_rate', 0.00001, 'Initial learning rate.')
    tf.app.flags.DEFINE_float('train_learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
    tf.app.flags.DEFINE_string('train_checkpoint_exclude_scopes', None,
        'Comma-separated list of scopes of variables to exclude when restoring from a checkpoint.')
    tf.app.flags.DEFINE_string('train_checkpoint_model_scope', None,
        'Model scope in the checkpoint. None if the same as the trained model.')
    tf.app.flags.DEFINE_string('train_trainable_scopes', None,
        'Comma-separated list of scopes to filter the set of variables to train.By default, None would train all the variables.')




tf.app.flags.DEFINE_string('train_dataset_path', './tfrecords_train/', 'The path of train dataset.')
tf.app.flags.DEFINE_string('train_dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')
tf.app.flags.DEFINE_integer('train_batch_size', 32, 'The number of samples in each batch.')
tf.app.flags.DEFINE_float('train_gpu_memory_fraction', 0.8, 'GPU memory fraction to use.')
tf.app.flags.DEFINE_integer('train_num_readers', 4, 'The number of parallel readers.')
tf.app.flags.DEFINE_integer('train_common_queue_capacity', 640, 'The capacity of the common queue')
tf.app.flags.DEFINE_boolean('train_shuffle', True, 'shuffle the data sources.')
tf.app.flags.DEFINE_integer('train_num_preprocessing_threads', 4,'The number of threads.')
tf.app.flags.DEFINE_float('train_loss_alpha', 1., 'Alpha parameter in the loss function.')
tf.app.flags.DEFINE_float('train_negative_ratio', 3., 'Negative ratio in the loss function.')
tf.app.flags.DEFINE_float('train_match_threshold', 0.5, 'Matching threshold in the loss function.')
tf.app.flags.DEFINE_float('train_label_smoothing', 0.0, 'The amount of label smoothing.')
tf.app.flags.DEFINE_integer('train_log_every_n_steps', 10, 'The frequency with which logs are print.')
tf.app.flags.DEFINE_integer('train_save_summaries_secs', 60, 'The frequency with which summaries are saved')
tf.app.flags.DEFINE_integer('train_save_interval_secs', 600,'The frequency with which the model is saved.')

# Optimization 
tf.app.flags.DEFINE_float('train_weight_decay', 0.0005, 'The weight decay on the model weights.')
tf.app.flags.DEFINE_string('train_optimizer', 'adam', 'adadelta, adagrad, adam, ftrl, momentum, sgd, rmsprop')
tf.app.flags.DEFINE_float('train_adam_beta1', 0.9, 'The exponential decay rate for the 1st moment estimates.')
tf.app.flags.DEFINE_float('train_adam_beta2', 0.999, 'The exponential decay rate for the 2nd moment estimates.')
tf.app.flags.DEFINE_float('train_opt_epsilon', 1.0, 'Epsilon term for the optimizer.')
tf.app.flags.DEFINE_float('train_ftrl_learning_rate_power', -0.5, 'The learning rate power.')
tf.app.flags.DEFINE_float('train_ftrl_initial_accumulator_value', 0.1,'Starting value for the FTRL accumulators.')
tf.app.flags.DEFINE_float('train_ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')
tf.app.flags.DEFINE_float('train_ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')
tf.app.flags.DEFINE_float('train_momentum', 0.9,'The momentum for the MomentumOptimizer and RMSPropOptimizer.')
tf.app.flags.DEFINE_float('train_rmsprop_momentum', 0.9, 'Momentum.')
tf.app.flags.DEFINE_float('train_rmsprop_decay', 0.9, 'Decay term for RMSProp.')

# Learning Rate Flags.
tf.app.flags.DEFINE_string('train_learning_rate_decay_type', 'exponential','fixed, exponential or polynomial')
tf.app.flags.DEFINE_float('train_end_learning_rate', 0.0001, 'The minimal end learning rate')
tf.app.flags.DEFINE_float('train_num_epochs_per_decay', 2.0, 'Number of epochs after which learning rate decays.')
tf.app.flags.DEFINE_integer('train_max_number_of_steps', None, 'The maximum number of training steps.')
tf.app.flags.DEFINE_boolean('train_ignore_missing_vars', True,'ignore missing variables.')

FLAGS = tf.app.flags.FLAGS



