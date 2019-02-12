# SSD: Single Shot MultiBox Object Detector based on Tensorflow

A TensorFlow implementation of the Single Shot Detector (SSD) for object detection. 

It has been originally published in this research [paper](https://arxiv.org/abs/1512.02325). This repository contains a TensorFlow re-implementation of SSD which is inspired by the previous caffe and tensorflow implementations. However, this code has clear pipelines for train, test and demo; it is modular that can be extended or be use for new applications; and it support 7 backbone networks. The backbone networks includes: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2.



## Introduction
This is a pipeline of Single Shot Detector (SSD) for object detection based on tensorflow. This implementation is designed with these goals:
###### Pipeline: has full pipeline of object detection for demo, test and train.
###### Backbone Networks: has 7 backbone network for SSD including: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2. 
###### Modularity: This code is modular and easy to expand with new ideas.


## Prerequisite
The main requirements can be installed by:

```bash
pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

# Install opencv for preprocessing training examples
pip install opencv-python
```

## Datasets
For training & testing, I used Pascal VOC datasets (2007 and 2012). 
To prapare tha datasets:
1. Download VOC2007 and VOC2012 datasets. I assume the data is stored in /datasets/
```
$ cd datasets
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
$ wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
```
2. Convert the data to Tensorflow records:
```
$ tar -xvf VOCtrainval_11-May-2012.tar
$ tar -xvf VOCtrainval_06-Nov-2007.tar
$ tar -xvf VOCtest_06-Nov-2007.tar
$ python3 ssd_image_to_tf.py
```
The resulted tf records will be stored into tfrecords_test and tfrecords_train folders.

## Configuration
Before running the code, you need to touch the configuration based on your needs. We have 5 config files in /configs:
- config_general.py: in this file, you can indicate the backbone model that you want to use for train, test and demo. You should uncomment only one of the models to use as backbone. Also, you can indicate the traning mode. 
- config_general.py: this file includes the common parameters that are used in training, testing and demo.   
- config_train.py: this file includes training parameters.  
- config_test.py: this file includes testing parameters.  
- config_demo.py: this file includes demo parameters.  


## Demo of SSD
Demo uses the pretrained model that has been stored in /checkpoints/ssd_... .

To run the demo, use the following command:
```python
# Run demo of SSD for one image
python3 ssd_demo.py
```
The demo module has the following 6 steps:
1) Define a placeholder for the input image 
2) Preprocessing step
3) Create SSD model
4) Inference, calculate output of network
5) Postprocessing
6) Visualization & Evaluation


## Evaluating (Testing) SSD 
Test module uses the pretrained model that has been stored in /checkpoints/ssd_... . 

To test the SSD, use the following command:
```python
# Run test of SSD
python3 ssd_test.py
```
Evaluation module has the following 6 steps:
1) Data preparation
2) Preprocessing step
3) Create SSD model
4) Inference, calculate output of network
5) Postprocessing        
6) Evaluation




## Training SSD
We have 3 modes for training:
- Training an existing SSD boject detection for a spacific dataset or new sets of parameters. 
- Training (first step fine-tuning) SSD based on an existing ImageNet classification model.
- Training (second step fine-tuning) SSD based on an existing ImageNet classification model.

The mode should be specified in configs/config_general.py.

The input of training should be in /checkpoints/[model_name]

the output of training will be store in checkpoints/ssd_[model_name]


To train the SSD, use the following command:
```python
# Run training of SSD
python3 ssd_train.py
```

The Training module has the following 4 steps:
1) Data preparation
2) Preprocessing step
3) Create SSD model
4) Training




# Understanding SSD
SSD is designed for object detection in real-time whish has one step. In contrast, Faster R-CNN uses a region proposal network and has two steps for object detection. SSD speeds up the process by eliminating the need of the region proposal network. To recover the drop in accuracy, SSD applies a few improvements including multi-scale features and default boxes. These improvements allow SSD to match the Faster R-CNN’s accuracy using lower resolution images, which further pushes the speed higher. According to the following comparison, it achieves the real-time processing speed and even beats the accuracy of the Faster R-CNN. (Accuracy is measured as the mean average precision mAP: the precision of the predictions.)





## Using backbone networks in SSD: 
In this section, I explain the details of how I used different backbone networks for SSD object detection.   


### VGG


### ResnetV1


### ResnetV2


### MobilenetV1


### MobilenetV2


### InceptionV4


### InceptionResnetV2





