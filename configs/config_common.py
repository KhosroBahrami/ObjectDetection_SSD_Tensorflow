
# Config for common parameters of training & testing

import tensorflow as tf
from configs.config_general import backbone_network
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('train_or_test', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_integer('image_size', 300, 'size of input image')

tf.app.flags.DEFINE_integer('num_classes', 21, 'Number of classes to use in the dataset.')

tf.app.flags.DEFINE_string('model_name', backbone_network, 'The name of the architecture to evaluate.')


FLAGS = tf.app.flags.FLAGS


