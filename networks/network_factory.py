
# This module is used to create SSD model based on a backbone network.
import functools
import tensorflow as tf
from networks import ssd_vgg_300
from networks import ssd_mobilenet_v1
from networks import ssd_mobilenet_v2
from networks import ssd_resnet_v1
from networks import ssd_resnet_v2
from networks import ssd_inception_v4
from networks import ssd_inception_resnet_v2


slim = tf.contrib.slim

networks = {
                'ssd_vgg_300': ssd_vgg_300, 
                'ssd_mobilenet_v1': ssd_mobilenet_v1,
                'ssd_mobilenet_v2': ssd_mobilenet_v2,
                'ssd_resnet_v1': ssd_resnet_v1,
                'ssd_resnet_v2': ssd_resnet_v2,
                'ssd_inception_v4': ssd_inception_v4,
                'ssd_inception_resnet_v2': ssd_inception_resnet_v2,
            }


# This function loads the network 
# Inputs:
#     name: network name
# Output:
#     loaded network   
def load_network(name):
    return networks[name].SSD_network()






