# SSD: Single Shot MultiBox Object Detector based on Tensorflow

A TensorFlow implementation of the Single Shot Detector (SSD) for object detection. 

It has been originally published in this research [article](https://arxiv.org/abs/1512.02325). This repository contains a TensorFlow re-implementation inspird by the prevoius caffe and tensorflow implementation. It only implements SSD networks with 300 input, and the 7 backbone architectures: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2.



## Introduction
This is a pipeline of Single Shot Detector (SSD) for object detection based on tensorflow. This implementation is designed with these goals:
###### Pipeline: has full pipeline of object detection for demo, test and train.
###### Backbone Networks: has 7 backbone network for SSD including: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2. 
###### Modularity: This code is modular and easy to expand with new ideas.


## Prerequisite
The main requirements can be installed by:

```bash
# pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

# Install scipy for loading mat files
pip install scipy

# Install matplotlib for visualizing tracking results
pip install matplotlib

# Install opencv for preprocessing training examples
pip install opencv-python
```


## Demo
```python
# Run demo of SSD for one image
python3 ssd_demo.py

```


## Test
```python
# Run test of SSD
python3 ssd_test.py

```


## Train
```python
# Run training of SSD
python3 ssd_train.py

```





