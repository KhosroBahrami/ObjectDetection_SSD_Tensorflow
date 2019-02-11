
# This module evaluates the SSD object detection
import tensorflow as tf
from datasets.dataset import *
from preprocessing.preprocessing import *
from networks import network_factory
from postprocessing.postprocessing import * 
from evaluation.evaluation import *
from configs.config_test import *
from configs.config_common import *
DATA_FORMAT = 'NHWC'

# main module for evaluation of SSD
def main():

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():

        #*** 1) Data preparation
        with tf.name_scope(None, "DataPreparation") as scope:
            print('\n1) Data preparation...')
            # Load dataset: e.g. pascalvoc_2007, ./tfrecords_test  
            print(FLAGS.test_dataset_name, FLAGS.test_dataset_path)
            # Create an object of dataset
            oDataset = Dataset()
            # Read dataset from tfrecords
            dataset_ = oDataset.read_dataset_from_tfrecords(FLAGS.test_dataset_name, 'test', FLAGS.test_dataset_path)
            # Read attributes of the dataset
            [image, gt_bboxes, gt_labels, gt_difficult_objects] = oDataset.get_groundtruth_from_dataset(dataset_, 'test')

        print(image.get_shape(),image.get_shape().ndims )
        print(gt_bboxes.get_shape())
        print(gt_labels.get_shape())
        print(gt_difficult_objects.get_shape())


                   
            
        #*** 2) Preprocessing step
        with tf.name_scope(None, "Preprocessing") as scope:
            print('\n2) Preprocessing...')
            oPreprocess = Preprocessing()
            image, gt_labels, gt_bboxes, gt_bbox_img = oPreprocess.preprocess(image, gt_labels, gt_bboxes, 'test')

        print(image.get_shape())
        print(gt_bboxes.get_shape())
        print(gt_labels.get_shape())
        print(gt_bbox_img.get_shape())



        #*** 3) Create SSD model
        with tf.name_scope(None, "SSD_model") as scope:
            print('\n3) SSD model...')
            # Create an object of SSD network class
            network = network_factory.load_network(FLAGS.model_name)
            # Generate anchors
            anchors = network.generate_anchors()
            # Generate groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = network.generate_groundtruth_bboxes(gt_labels, gt_bboxes, anchors)
            batch_shape = [1] * 5 + [len(anchors)] * 3
            # Generate test batches for evaluation 
            (b_image, b_glabels, b_gbboxes, b_gdifficults,
              b_gbbox_img, b_gclasses,b_glocalisations, b_gscores) = network.generate_test_batches(
                image, gt_labels, gt_bboxes, gt_difficult_objects,
                gt_bbox_img, gclasses, glocalisations, gscores, batch_shape,
                batch_size = FLAGS.test_batch_size, num_threads = FLAGS.test_num_preprocessing_threads,
                capacity = 5 * FLAGS.test_batch_size, dynamic_pad=True)
        


        #*** 4) Inference, calculate output of network
        with tf.name_scope(None, "Inference") as scope:
            print('\n4) Inference...')
            with slim.arg_scope(network.arg_scope(data_format=DATA_FORMAT)):
                 predictions, localisations, logits, end_points = network.net(b_image, training=False)


        

        #*** 5) Postprocessing
        with tf.name_scope(None, "Postprocessing") as scope:
            print('\n5) Postprocessing...')
            oPostprocess = Postprocessing()
            localisations = network.generate_relative_bboxes(localisations, anchors)
            # Select top bboxes
            rscores, rbboxes = oPostprocess.select_top_bboxes(predictions, localisations)
            # Sort bounding boxes
            rscores, rbboxes = oPostprocess.sort_bboxes(rscores, rbboxes)
            # Non Maximum Supression 
            rscores, rbboxes = oPostprocess.non_maximum_supression_bboxes(rscores, rbboxes)

            writer = tf.summary.FileWriter( './logs/test ')
            writer.add_graph(tf.Graph())


        #*** 6) Evaluation
        with tf.name_scope(None, "Evaluation") as scope:
            print('\n6) Evaluation...')
            oEvaluation = Evaluation()
            oEvaluation.evaluate(rscores, rbboxes, b_glabels, b_gbboxes, b_gdifficults) 



 

if __name__ == '__main__':
    main()



