
# This module converts images to TFrecords

import tensorflow as tf
from datasets.dataset import *


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset_name', 'pascalvoc', 'The name of the dataset to convert.')


# Convert test images to TF records 
tf.app.flags.DEFINE_string('dataset_dir', './voc/voc_2007_test/', 'Input directory of images.')
tf.app.flags.DEFINE_string('output_name', 'voc_2007_test', 'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string('output_dir', './tfrecords_test', 'Output directory to store TFRecords.')

'''
# Convert train images to TF records 
tf.app.flags.DEFINE_string('dataset_dir', './voc/voc_2007_train/', 'Input directory of images.')
tf.app.flags.DEFINE_string('output_name', 'voc_2007_train', 'Basename used for TFRecords output files.')
tf.app.flags.DEFINE_string('output_dir', './tfrecords_train', 'Output directory to store TFRecords.')
'''

def main(_):
    print('Dataset directory:', FLAGS.dataset_dir)
    print('Output directory:', FLAGS.output_dir)

    if FLAGS.dataset_name == 'pascalvoc':
        oDataset = Dataset()
        oDataset.run_PascalVOC(FLAGS.dataset_dir, FLAGS.output_dir, FLAGS.output_name)
    else:
        raise ValueError('Dataset [%s] was not recognized.' % FLAGS.dataset_name)

if __name__ == '__main__':
    tf.app.run()




