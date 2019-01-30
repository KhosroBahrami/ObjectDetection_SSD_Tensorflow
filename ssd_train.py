
# This module trains the SSD object detection using a given dataset
import tensorflow as tf
from datasets.dataset import *
from preprocessing.preprocessing import *
from networks import network_factory
from training.training import *
from configs.config_train import *

# main module for training of SSD
def main():

    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        global_step = slim.create_global_step()
        
        #*** 1) Data preparation
        with tf.name_scope(None, "DataPreparation") as scope:
            print('\n1) Data preparation...')
            # Load dataset: e.g.pascalvoc_2007, ./tfrecords_train/ 
            print(FLAGS.train_dataset_name, FLAGS.train_dataset_path)
            # Create an object of dataset
            oDataset = Dataset()
            # Read dataset from tfrecords
            dataset_ = oDataset.read_dataset_from_tfrecords(FLAGS.train_dataset_name, 'train', FLAGS.train_dataset_path)
            # Read attributes of the dataset
            [image, gt_bboxes, gt_labels, gt_difficult_objects] = oDataset.get_groundtruth_from_dataset(dataset_, 'train')



        #*** 2) Preprocessing step
        with tf.name_scope(None, "Preprocessing") as scope:
            print('\n2) Preprocessing...')
            oPreprocess = Preprocessing()
            image, gt_labels, gt_bboxes, _ = oPreprocess.preprocess(image, gt_labels, gt_bboxes, 'train')

                                
                
        #*** 3) Create SSD model
        with tf.name_scope(None, "SSD_model") as scope:
            print('\n3) SSD model...')
            # Create an object of SSD network class
            network = network_factory.load_network(FLAGS.model_name)
            # Generate anchors
            anchors = network.generate_anchors()
            # Generate groundtruth labels and bboxes.
            gclasses, glocalisations, gscores = network.generate_groundtruth_bboxes(gt_labels, gt_bboxes, anchors)
            batch_shape = [1] + [len(anchors)] * 3
            # Generate train batches 
            (b_image, b_gclasses,b_glocalisations, b_gscores, batch_queue) = network.generate_train_batches(
                 image, gclasses, glocalisations, gscores, batch_shape, batch_size = FLAGS.train_batch_size,
                      num_threads = FLAGS.train_num_preprocessing_threads,
                      capacity = 5 * FLAGS.train_batch_size, dynamic_pad=True)


        
      

        #*** 4) Training
        with tf.name_scope(None, "Training") as scope:
            print('\n4) Training...')
            oTraining = Training() 
            oTraining.training(network, b_image, b_gclasses,b_glocalisations, b_gscores, batch_queue, batch_shape, dataset_, global_step)


 


if __name__ == '__main__':
    main()


