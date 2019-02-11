
# Config for demo of SSD object detection
import tensorflow as tf
from configs.config_general import backbone_network
slim = tf.contrib.slim

tf.app.flags.DEFINE_float('demo_selection_threshold', 0.5, 'threshold to select image block')

tf.app.flags.DEFINE_float('demo_nms_threshold', 0.45, 'threshold for nms (non-maximum selection)')

tf.app.flags.DEFINE_string('demo_path_of_demo_images', 'demo_images/', 'path of demo images')

tf.app.flags.DEFINE_string('demo_image_name', 'person.jpg', 'name of demo image')

tf.app.flags.DEFINE_string('demo_data_format', 'NHWC', 'number of classes')

if backbone_network == 'ssd_vgg_300':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_vgg/ssd_300_vgg.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_mobilenet_v1':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_mobilenet_v1/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_mobilenet_v2':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_mobilenet_v2/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_resnet_v1':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_resnet_v1/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_resnet_v2':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_resnet_v2/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_inception_v4':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_inception_v4/model.ckpt','path of checkpoint file.')
elif backbone_network == 'ssd_inception_resent_v2':
   tf.app.flags.DEFINE_string('demo_checkpoint_path', './checkpoints/ssd_inception_resent_v2/model.ckpt','path of checkpoint file.')

FLAGS = tf.app.flags.FLAGS









































