
# Demo of object detection using SSD method
import sys
import os
import math
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from enum import Enum, IntEnum
from preprocessing.preprocessing import *
from networks import network_factory
from postprocessing.demo_postprocessing import * 
from visualization.visualization import * 
from configs.config_demo import *


# main module for demo
def main():
   
   gpu_options = tf.GPUOptions(allow_growth=True)
   config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)

   with tf.Session() as sess:

     
      #*** 1) Define a placeholder for the input image 
      with tf.name_scope(None, "Input_image") as scope:
         print('\n1) Input image...')
         input_image = tf.placeholder(tf.uint8, shape=(None, None, 3))
      

      #*** 2) Preprocessing step
      with tf.name_scope(None, "Preprocessing") as scope:
         print('\n2) Preprocessing...')
         oPreprocess = Preprocessing()
         image, _, _, bbox_img = oPreprocess.preprocess(input_image, None, None, 'test')
         image = tf.expand_dims(image, 0)


      #*** 3) Create SSD model
      with tf.name_scope(None, "SSD_model") as scope:
         print('\n3) SSD model...')
         # Create an object of SSD VGG network class
         print('  loaded model: ',FLAGS.model_name)
         network = network_factory.load_network(FLAGS.model_name)
         with slim.arg_scope(network.arg_scope(data_format=FLAGS.demo_data_format)):
             predictions, localisations, _, _ = network.net(image, training=False)

         # Restore SSD model.
         saver = tf.train.Saver()
         saver.restore(sess, FLAGS.demo_checkpoint_path)


      #*** 4) Inference, calculate output of network
      with tf.name_scope(None, "Inference") as scope:
         print('\n4) SSD Inference...')
         # Generate anchor boxes
         anchors = network.generate_anchors()
         # Read image from demo folder
         img = mpimg.imread(FLAGS.demo_path_of_demo_images + FLAGS.demo_image_name)
         # Run SSD network
         rimg, rpredictions, rlocalisations, rbbox_img = sess.run([image, predictions, localisations, bbox_img],
                                                                  feed_dict={input_image: img})


      # 5) Postprocessing
      with tf.name_scope(None, "Postprocessing") as scope:
         print('\n5) Postprocessing...')
         oPostprocess = Demo_Postprocessing()
         # Select bounding boxes
         rclasses, rscores, rbboxes = oPostprocess.select_top_bboxes(rpredictions, rlocalisations, anchors)
         # Sort bounding boxes
         rclasses, rscores, rbboxes = oPostprocess.sort_bboxes(rclasses, rscores, rbboxes)
         # Non Maximum Supression: fuse boxes if Jaccard score > threshold
         rclasses, rscores, rbboxes = oPostprocess.non_maximum_supression_bboxes(rclasses, rscores, rbboxes)
      

      # 6) Visualization & Evaluation
      with tf.name_scope(None, "Visualization") as scope:
         print('\n6) Visualization...')
         oVisualization = Visualization()
         oVisualization.plot_bboxes(img, rclasses, rscores, rbboxes)

       
      writer = tf.summary.FileWriter("./logs/demo", sess.graph)
      writer.close()




if __name__ == '__main__':
    main()












































