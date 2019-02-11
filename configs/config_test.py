
# Config of test
import tensorflow as tf
from configs.config_general import backbone_network
slim = tf.contrib.slim

# Dataset  
tf.app.flags.DEFINE_string('test_eval_dir', './tmp/tfmodel/', 'Directory where the results are saved to.')
tf.app.flags.DEFINE_string('test_dataset_path', './tfrecords_test/', 'path of test dataset.')
tf.app.flags.DEFINE_string('test_dataset_name', 'pascalvoc_2007', 'The name of the dataset to load.')


if backbone_network == 'ssd_vgg_300':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_vgg/ssd_300_vgg.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_mobilenet_v1':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_mobilenet_v1/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_mobilenet_v2':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_mobilenet_v2/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_resnet_v1':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_resnet_v1/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_resnet_v2':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_resnet_v2/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_inception_v4':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_inception_v4/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_inception_resent_v2':
   tf.app.flags.DEFINE_string('test_checkpoint_path', './checkpoints/ssd_inception_resent_v2/model.ckpt','path of checkpoint file.')



tf.app.flags.DEFINE_float('test_gpu_memory_fraction', 0.1, 'GPU memory fraction to use.')

# Data reading 
tf.app.flags.DEFINE_integer('test_common_queue_capacity', 2, 'The capacity of the common queue')
tf.app.flags.DEFINE_integer('test_batch_size', 1, 'The number of samples in each batch.')
tf.app.flags.DEFINE_boolean('test_shuffle', False, 'shuffle the data sources and common queue when reading.')
tf.app.flags.DEFINE_integer('test_num_readers', 1, 'The number of parallel readers to use')

# Processing 
tf.app.flags.DEFINE_integer('test_num_preprocessing_threads', 4, 'The number of threads')
tf.app.flags.DEFINE_integer('test_num_classes', 21, 'Number of classes to use in the dataset.')
tf.app.flags.DEFINE_float('test_select_threshold', 0.01, 'Selection threshold.')
tf.app.flags.DEFINE_integer('test_max_num_batches', 50, 'Max number of batches to evaluate by default use all.')
tf.app.flags.DEFINE_float('test_matching_threshold', 0.5, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer('test_sort_top_k', 400, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_integer('test_nms_top_k', 200, 'Matching threshold with groundtruth objects.')
tf.app.flags.DEFINE_boolean('test_discard_difficult_objects', True, 'discard difficult objects from evaluation.')

# Sataset sizes
tf.app.flags.DEFINE_integer('pascalvoc_2007_train_size', 5011, 'size of train dataset of voc 2007')
tf.app.flags.DEFINE_integer('pascalvoc_2007_test_size', 4952, 'size of test dataset of voc 2007')
tf.app.flags.DEFINE_integer('pascalvoc_2012_train_size', 17125, 'size of train dataset of voc 2012')
tf.app.flags.DEFINE_integer('pascalvoc_2012_test_size', 0, 'size of test dataset of voc 2012')


FLAGS = tf.app.flags.FLAGS


