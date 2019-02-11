# SSD: Single Shot MultiBox Object Detector based on Tensorflow

A TensorFlow implementation of the Single Shot Detector (SSD) for object detection. 

It has been originally published in this research [paper](https://arxiv.org/abs/1512.02325). This repository contains a TensorFlow re-implementation inspired by the caffe and tensorflow implementation. It implements SSD networks with 7 backbone architectures, including: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2.



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
The Pascal VOC datasets (2007 and 2012). 


## Demo of SSD
```python
# Run demo of SSD for one image
python3 ssd_demo.py

```


## Evaluating SSD 
```python
# Run test of SSD
python3 ssd_test.py

```


## Training SSD
```python
# Run training of SSD
python3 ssd_train.py

```



# Understanding SSD
SSD is designed for object detection in real-time whish has one step. In contrast, Faster R-CNN uses a region proposal network and has two steps for object detection. SSD speeds up the process by eliminating the need of the region proposal network. To recover the drop in accuracy, SSD applies a few improvements including multi-scale features and default boxes. These improvements allow SSD to match the Faster R-CNN’s accuracy using lower resolution images, which further pushes the speed higher. According to the following comparison, it achieves the real-time processing speed and even beats the accuracy of the Faster R-CNN. (Accuracy is measured as the mean average precision mAP: the precision of the predictions.)





## Backbone: 

### VGG

### ResnetV1

### ResnetV2

### MobilenetV1

### MobilenetV2

### InceptionV4

### InceptionResnetV2





