
# This module is used to load pascalvoc datasets (2007 or 2012)
import os
import tensorflow as tf
from configs.config_common import *
from configs.config_train import *
from configs.config_test import *
import sys
import random
import numpy as np
import xml.etree.ElementTree as ET

# Original dataset organisation.
DIRECTORY_ANNOTATIONS = 'Annotations/'
DIRECTORY_IMAGES = 'JPEGImages/'

# TFRecords convertion parameters.
RANDOM_SEED = 4242
SAMPLES_PER_FILES = 200

slim = tf.contrib.slim



class Dataset(object):

    def __init__(self):
        # Descriptions of the image items
        self.items_descriptions = {
            'image': 'A color image of varying height and width.',
            'shape': 'Shape of the image',
            'object/bbox': 'A list of bounding boxes, one per each object.',
            'object/label': 'A list of labels, one per each object.',
        }
        # Features of Pascal VOC TFRecords.
        self.features = {
            'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
            'image/object/bbox/xmin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymin': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/xmax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/ymax': tf.VarLenFeature(dtype=tf.float32),
            'image/object/bbox/label': tf.VarLenFeature(dtype=tf.int64),
            'image/object/bbox/difficult': tf.VarLenFeature(dtype=tf.int64),
        }
        # Items in Pascal VOC TFRecords.
        self.items = {
            'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
            'gt_bboxes': slim.tfexample_decoder.BoundingBox(['ymin','xmin','ymax','xmax'], 'image/object/bbox/'),
            'gt_labels': slim.tfexample_decoder.Tensor('image/object/bbox/label'),
            'difficult_objects': slim.tfexample_decoder.Tensor('image/object/bbox/difficult'),
        }



    # This function reads dataset from tfrecords
    # Inputs:
    #     datase_name: pascalvoc_2007
    #     train_or_test: test
    #     dataset_path: './tfrecords_test/'
    # Outputs:
    #     loaded dataset
    def read_dataset_from_tfrecords(self, dataset_name, train_or_test, dataset_path):
        with tf.name_scope(None, "read_dataset_from_tfrecords") as scope:
            if dataset_name == 'pascalvoc_2007' or dataset_name == 'pascalvoc_2012':
               dataset = self.load_dataset(dataset_name, train_or_test, dataset_path)
            return dataset



    # This function is used to load pascalvoc2007 or psaclvoc2012 datasets
    # Inputs:
    #     dataset_name: pascalvoc_2007
    #     train_or_test: test
    #     dataset_path: './tfrecords_test/'
    # Output:
    #     loaded dataset   
    def load_dataset(self, dataset_name, train_or_test, dataset_path):
            dataset_file_name = dataset_name[6:] + '_%s_*.tfrecord'
            if dataset_name == 'pascalvoc_2007':
               train_test_sizes = {
                  'train': FLAGS.pascalvoc_2007_train_size,
                  'test': FLAGS.pascalvoc_2007_test_size,
               }
            elif dataset_name == 'pascalvoc_2012':
               train_test_sizes = {
                  'train': FLAGS.pascalvoc_2012_train_size, 
               }   
            dataset_file_name = os.path.join(dataset_path, dataset_file_name % train_or_test)
            reader = tf.TFRecordReader
            decoder = slim.tfexample_decoder.TFExampleDecoder(self.features, self.items)
            return slim.dataset.Dataset(
                    data_sources=dataset_file_name,
                    reader=reader,
                    decoder=decoder,
                    num_samples=train_test_sizes[train_or_test],
                    items_to_descriptions=self.items_descriptions,
                    num_classes=FLAGS.num_classes-1,
                    labels_to_names=None)



    # This function gets groundtruth bboxes & labels from dataset
    # Inputs:
    #     dataset
    #     train_or_test: train/test
    # Output:
    #     image, ground-truth bboxes, ground-truth labels, ground-truth difficult objects
    def get_groundtruth_from_dataset(self, dataset, train_or_test):
        # Dataset provider
        with tf.name_scope(None, "get_groundtruth_from_dataset") as scope:
            if train_or_test == 'test':
                provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        num_readers=FLAGS.test_num_readers,
                        common_queue_capacity=FLAGS.test_common_queue_capacity,
                        common_queue_min=FLAGS.test_batch_size,
                        shuffle=FLAGS.test_shuffle)
            elif train_or_test == 'train':
                provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        num_readers= FLAGS.train_num_readers,
                        common_queue_capacity= FLAGS.train_common_queue_capacity,
                        common_queue_min= 10 * FLAGS.train_batch_size,
                        shuffle=FLAGS.train_shuffle)
            # Get images, groundtruth bboxes & groundtruth labels from database
            [image, gt_bboxes, gt_labels] = provider.get(['image','gt_bboxes','gt_labels'])
            # Discard difficult objects
            gt_difficult_objects = tf.zeros(tf.shape(gt_labels), dtype=tf.int64)
            if FLAGS.test_discard_difficult_objects:
                [gt_difficult_objects] = provider.get(['difficult_objects'])
            return  [image, gt_bboxes, gt_labels, gt_difficult_objects]





    ##########################################
    # Convert PascalVOC to TF recorsd
    # Process a image and annotation file.
    # Inputs:
    #  filename: string, path to an image file e.g., '/path/to/example.JPG'.
    #  coder: instance of ImageCoder to provide TensorFlow image coding utils.
    # Outputs:
    #  image_buffer: string, JPEG encoding of RGB image.
    #  height: integer, image height in pixels.
    #  width: integer, image width in pixels.
    def _process_image_PascalVOC(self, directory, name):

        # Read the image file.
        filename = directory + DIRECTORY_IMAGES + name + '.jpg'
        image_data = tf.gfile.FastGFile(filename, 'r').read()

        # Read the XML annotation file.
        filename = os.path.join(directory, DIRECTORY_ANNOTATIONS, name + '.xml')
        tree = ET.parse(filename)
        root = tree.getroot()

        # Image shape.
        size = root.find('size')
        shape = [int(size.find('height').text), int(size.find('width').text), int(size.find('depth').text)]
        # Find annotations.
        bboxes = []
        labels = []
        labels_text = []
        difficult = []
        truncated = []
        for obj in root.findall('object'):
            label = obj.find('name').text
            labels.append(int(VOC_LABELS[label][0]))
            labels_text.append(label.encode('ascii'))

            if obj.find('difficult'):
                difficult.append(int(obj.find('difficult').text))
            else:
                difficult.append(0)
            if obj.find('truncated'):
                truncated.append(int(obj.find('truncated').text))
            else:
                truncated.append(0)

            bbox = obj.find('bndbox')
            bboxes.append((float(bbox.find('ymin').text) / shape[0],
                           float(bbox.find('xmin').text) / shape[1],
                           float(bbox.find('ymax').text) / shape[0],
                           float(bbox.find('xmax').text) / shape[1]
                           ))
        return image_data, shape, bboxes, labels, labels_text, difficult, truncated




    # Build an Example proto for an image example.
    #    Args:
    #      image_data: string, JPEG encoding of RGB image;
    #      labels: list of integers, identifier for the ground truth;
    #      labels_text: list of strings, human-readable labels;
    #      bboxes: list of bounding boxes; each box is a list of integers;
    #      shape: 3 integers, image shapes in pixels.
    #    Returns:
    #      Example proto
    def _convert_to_example_PascalVOC(self, image_data, labels, labels_text, bboxes, shape, difficult, truncated):

        xmin = []
        ymin = []
        xmax = []
        ymax = []
        for b in bboxes:
            assert len(b) == 4
            # pylint: disable=expression-not-assigned
            [l.append(point) for l, point in zip([ymin, xmin, ymax, xmax], b)]
            # pylint: enable=expression-not-assigned

        image_format = b'JPEG'
        example = tf.train.Example(features=tf.train.Features(feature={
                'image/height': self.int64_feature(shape[0]),
                'image/width': self.int64_feature(shape[1]),
                'image/channels': self.int64_feature(shape[2]),
                'image/shape': self.int64_feature(shape),
                'image/object/bbox/xmin': self.float_feature(xmin),
                'image/object/bbox/xmax': self.float_feature(xmax),
                'image/object/bbox/ymin': self.float_feature(ymin),
                'image/object/bbox/ymax': self.float_feature(ymax),
                'image/object/bbox/label': self.int64_feature(labels),
                'image/object/bbox/label_text': self.bytes_feature(labels_text),
                'image/object/bbox/difficult': self.int64_feature(difficult),
                'image/object/bbox/truncated': self.int64_feature(truncated),
                'image/format': self.bytes_feature(image_format),
                'image/encoded': self.bytes_feature(image_data)}))
        return example



    # Loads data from image and annotations files and add them to a TFRecord.
    # Inputs:
    #      dataset_dir: Dataset directory;
    #      name: Image name to add to the TFRecord;
    #      tfrecord_writer: The TFRecord writer to use for writing.
    def _add_to_tfrecord_PascalVOC(self, dataset_dir, name, tfrecord_writer):

        image_data, shape, bboxes, labels, labels_text, difficult, truncated = self._process_image_PascalVOC(dataset_dir, name)
        example = self._convert_to_example_PascalVOC(image_data, labels, labels_text, bboxes, shape, difficult, truncated)
        tfrecord_writer.write(example.SerializeToString())



    def _get_output_filename_PascalVOC(output_dir, name, idx):
        return '%s/%s_%03d.tfrecord' % (output_dir, name, idx)



    # Convert images to tfrecords
    #    Args:
    #      dataset_dir: The dataset directory where the dataset is stored.
    #      output_dir: Output directory.
    def run_PascalVOC(self, dataset_dir, output_dir, name='voc_train', shuffling=False):

        if not tf.gfile.Exists(dataset_dir):
            tf.gfile.MakeDirs(dataset_dir)
        # Dataset filenames, and shuffling.
        path = os.path.join(dataset_dir, DIRECTORY_ANNOTATIONS)
        filenames = sorted(os.listdir(path))
        if shuffling:
            random.seed(RANDOM_SEED)
            random.shuffle(filenames)
        # Process dataset files.
        i = 0
        fidx = 0
        while i < len(filenames):
            # Open new TFRecord file.
            tf_filename = self._get_output_filename(output_dir, name, fidx)
            with tf.python_io.TFRecordWriter(tf_filename) as tfrecord_writer:
                j = 0
                while i < len(filenames) and j < SAMPLES_PER_FILES:
                    sys.stdout.write('\r>> Converting image %d/%d' % (i+1, len(filenames)))
                    sys.stdout.flush()

                    filename = filenames[i]
                    img_name = filename[:-4]
                    self._add_to_tfrecord_PascalVOC(dataset_dir, img_name, tfrecord_writer)
                    i += 1
                    j += 1
                fidx += 1
        print('\n ImageDB to TF conversion finished. ')



    # Wrapper for inserting int64 features into Example proto.
    def int64_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


    # Wrapper for inserting float features into Example proto.
    def float_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))


    # Wrapper for inserting bytes features into Example proto.
    def bytes_feature(self, value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))






        


