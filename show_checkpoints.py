
# Show checkpoints
from tensorflow.python import pywrap_tensorflow
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets 


print('\n vgg:')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/vgg/ssd_300_vgg.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


print('\n\n mobilenetV1 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/mobilenet_v1/mobilenet_v1_1.0_224.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)

'''
print('\n\n ssd_mobilenetV1 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/ssd_mobilenet_v1/model.ckpt-10518')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)
'''

print('\n\n mobilenetV2 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/mobilenet_v2/mobilenet_v2_1.0_224.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


print('\n\n resnet_v1 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/resnet_v1/resnet_v1_152.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


print('\n\n resnet_v2 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/resnet_v2/resnet_v2_152.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


print('\n\n inception_v4 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/inception_v4/inception_v4.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)



print('\n\n inception_resnet_v2 :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/inception_resnet_v2/inception_resnet_v2_2016_08_30.ckpt')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)



print('\n\n squeeznet :')
reader = pywrap_tensorflow.NewCheckpointReader('checkpoints/squeezeNet/squeezenet_v1.1.pkl')
var_to_shape_map = reader.get_variable_to_shape_map()
#saver = tf.train.Saver()
print('\n')
for v in sorted(var_to_shape_map):
   print(v)


   
