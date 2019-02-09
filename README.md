# SSD: Single Shot MultiBox Object Detector based on Tensorflow

A TensorFlow implementation of the Single Shot Detector (SSD) for object detection.


## Introduction
This is a TensorFlow implementation of Single Shot Detector (SSD) for object detection. 

This implementation is designed with these goals:
###### Readability. The code should be clear and consistent.
###### Modularization. The whole system should be modularized and easy to expand with new ideas.

 
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





