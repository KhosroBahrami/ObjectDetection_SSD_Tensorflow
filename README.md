# SSD: Single Shot MultiBox Object Detector based on Tensorflow

A TensorFlow implementation of the Single Shot Detector (SSD) for object detection. 

It has been originally published in this research [paper](https://arxiv.org/abs/1512.02325). This repository contains a TensorFlow re-implementation of SSD which is inspired by the previous caffe and tensorflow implementations. However, this code has clear pipelines for train, test and demo; it is modular that can be extended or be use for new applications; and it also it supports 7 backbone networks. The backbone networks, including VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2.

I will explain the details of using these backbones in SSD object detection, at the end of this document.  


## Introduction
This implementation of Single Shot Detector (SSD) for object detection based on tensorflow is designed with the following goals:
- Pipeline: it has full pipeline of object detection for demo, test and train with seperate modules.
- Backbone Networks: it has 7 backbone networks, including: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2. Any new backbone can be easily added to the code.
- Modularity: This code is modular and easy to expand for any specific application or new ideas.


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




# How SSD works?
SSD has been designed for object detection in real-time. In SSD, we only need to take one single shot to detect multiple objects within the image, while regional proposal network (RPN) based approaches such as Faster R-CNN needs two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches. The following table compare SSD, Faster RCNN and YOLO.

| Object Detection Method | VOC2007 test mAP |  Speed (FPS) | Number of Prior Boxes | Input Resolution |
| :---: |   :---:     | :---: | :---: | :---: |
| Faster R-CNN (VGG16) | 73.2% | 7 | 6000 | 1000*600
| YOLOv1 | 63.4% | 45 |  98 | 448*448 |
| SSD300 (VGG16) | 74.3% | 59 | 8732 | 300*300 |
| SSD300 (VGG16) | 76.9% | 22 | 24564 | 512*512 |

### Backbone network & Feature maps
The input of SSD is an image of fixed size, for example, 300x300 for SSD300. The image feeds into a CNN backbone network with several layers and generates multiple feature maps at different scales. 

Features maps (i.e. the results of the convolutional blocks) are a representation of the dominant features of the image at different scales, therefore running MultiBox on multiple feature maps increases the likelihood of any object (large and small) to be eventually detected, localized and appropriately classified. The image below shows how the network “sees” a given image across its feature maps:

![Alt text](figs/vgg_feature.png?raw=true "VGG Feature Map Visualisation (from Brown Uni)")

The CNN backbone network (VGG, Mobilenet, ...) gradually reduces the feature map size and increase the depth as it goes to the deeper layers. The deep layers cover larger receptive fields and construct more abstract representation, while the shallow layers cover smaller receptive fields. By using extracted features at different levels, we can use shallow layers to predict small objects and deeper layers to predict large objects.

The output of SSD is a prediction map. Each location in this map stores classes confidence and bounding box information as if there is indeed an object of interests at every location. Obviously, there will be a lot of false alarms, so a further process is used to select a list of predictions.

For example, for VGG backbone network, the first feature map is generated from layer 23 with a size of 38x38 of depth 512. Every point in the 38x38 feature map represents a part of the image, and the 512 channels are the features for every point. By using the features of 512 channels, we can predict the class label (using classification) and the bounding box (using regression) of the small objects on every point. The second feature map has a size of 19x19, which can be used for larger objects, as the points of the features cover larger receptive fields. Finally, in the last layer, there is only one point in the feature map which is used for big objects.

To train the network, one needs to compare the ground truth (a list of objects) against the prediction map. This is achieved with the help of prior boxes.

### Prior boxes (Anchor points) 
Intuitively, object detection is a local task: what is in the top left corner of an image is usually unrelated to predict an object in the bottom right corner of the image. So one needs to measure how relevance each ground truth is to each prediction. The criterion for matching a prior and a ground-truth box is IoU (Intersection Over Union), which is also called Jaccard index. The more overlap, the better match. Also, to have the same block size, the ground-truth boxes should be scaled to the same scale.

The procedure for matching prior boxes with ground-truth boxes is as follows:
- We put one priorbox at each location in the prediction map.
- We compute the intersect over union (IoU) between the priorbox and the ground truth.
- The ground truth object that has the highest IoU is used as the target for each prediction, given its IoU is higher than a threshold.
- For predictions who have no valid match, the target class is set to the background class and they will not be used for calculating the localization loss.
- If there is significant overlapping between a priorbox and a ground truth object, then the ground truth can be used at that location. The class of the ground truth is directly used to compute the classification loss; whereas the offset between the ground truth bounding box and the priorbox is used to compute the location loss.

Also, usually, different sizes for predictions at different scales are used. For example, SSD300 uses 21, 45, 99, 153, 207, 261 as the sizes of the priorbox at its 6 different prediction layers.

In practice, SSD uses a few different types of priorbox, each with a different scale or aspect ratio, in a single layer. Doing so creates different "experts" for detecting objects of different shapes. For example, SSD300 use ... types of different priorboxes for its seven prediction layers, whereas the aspect ratio of these priorboxes can be chosen from 1:3, 1:2, 1:1, 2:1 or 3:1. Notice, experts in the same layer take the same underlying input (the same receptive field). They behave differently because they use different parameters (convolutional filters) and use different ground truth fetch by different priorboxes.

### Scales and Aspect Ratios of Prior Boxes

![Alt text](figs/scale_bb.png?raw=true "Scale of Default Boxes")

Suppose we have m feature maps for prediction, we can calculate Sk for the k-th feature map. Smin is 0.2, Smax is 0.9. That means the scale at the lowest layer is 0.2 and the scale at the highest layer is 0.9. All layers in between is regularly spaced.

For each scale, sk, we have 5 non-square aspect ratios:

![Alt text](figs/ns_bb.png?raw=true "5 Non-Square Bounding Boxes")


For aspect ratio of 1:1, we got sk’:

![Alt text](figs/s_bb.png?raw=true "1 Square Bounding Box")

Therefore, we can have at most 6 bounding boxes in total with different aspect ratios. For layers with only 4 bounding boxes, ar = 1/3 and 3 are omitted.


### Number of Prior Boxses: 
The number of prior boxes is calculated as follow: 
For example for VGG as backbone, 6 feature maps from layers Conv4_3, Conv7, Conv8_2, Conv9_2, Conv10_2 and Conv11_2 are used. 
At Conv4_3, feature map is of size 38×38×512. 3×3 conv is applied. And there are 4 bounding boxes and each bounding box will have (classes + 4) outputs. Thus, at Conv4_3, the output is 38×38×4×(c+4). Suppose there are 20 object classes plus one background class, the output is 38×38×4×(21+4) = 144,400. In terms of number of bounding boxes, there are 38×38×4 = 5776 bounding boxes.
Similarly for other conv layers:
- Conv7: 19×19×6 = 2166 boxes (6 boxes for each location)
- Conv8_2: 10×10×6 = 600 boxes (6 boxes for each location)
- Conv9_2: 5×5×6 = 150 boxes (6 boxes for each location)
- Conv10_2: 3×3×4 = 36 boxes (4 boxes for each location)
- Conv11_2: 1×1×4 = 4 boxes (4 boxes for each location)

If we sum them up, we got 5776 + 2166 + 600 + 150 + 36 +4 = 8732 boxes in total for SSD. 
In YOLO, there are 7×7 locations at the end with 2 bounding boxes for each location. YOLO only got 7×7×2 = 98 boxes. Hence, SSD has 8732 bounding boxes which is more than that of YOLO.




### MultiBox Detection 

Multi-scale Detection: The resolution of the detection equals the size of its prediction map. Multi-scale detection is achieved by generating prediction maps of different resolutions. For example, SSD512 outputs seven prediction maps of resolutions 64x64, 32x32, 16x16, 8x8, 4x4, 2x2, and 1x1 respectively. You can think there are 5461 "local prediction" behind the scene. The input of each prediction is effectively the receptive field of the output feature.


### Hard Negative Mining
Priorbox uses a distance-based metric (IoU) to create ground truth predictions, including backgrounds (no matched objects) and objects. However, there can be an imbalance between foreground samples and background samples. There are a lot more unmatched priors (priors without any object). In consequence, the detector may produce many false negatives due to the lack of a training signal of foreground objects. In other words, the huge number of priors labelled as background make the dataset very unbalanced.

To address this problem, SSD uses hard negative mining: all background samples are sorted by their predicted background scores in the ascending order. Only the top K samples are kept for proceeding to the computation of the loss. K is computed on the fly for each batch to keep a 1:3 ratio between foreground samples and background samples.


![Alt text](figs/hnm.png?raw=true "Example of hard negative mining (from Jamie Kang blog)")




### Image Augmentation
The authors of SSD stated that data augmentation, like in many other deep learning applications, has been crucial to teach the network to become more robust to various object sizes in the input. To this end, they generated additional training examples with patches of the original image at different IoU ratios (e.g. 0.1, 0.3, 0.5, etc.) and random patches as well. Moreover, each image is also randomly horizontally flipped with a probability of 0.5, thereby making sure potential objects appear on left and right with similar likelihood.


### Loss function
The loss function is the combination of confidence loss (classification loss) and localization loss (regression loss). 
MultiBox’s loss function also combined two critical components that made their way into SSD:

Location loss: This measures how far away the network’s predicted bounding boxes are from the ground truth ones from the training set. It is the smooth L1 (L2) loss between the predicted box (l) and the ground-truth box (g) parameters. These parameters include the offsets for the center point (cx, cy), width (w) and height (h) of the bounding box. This loss is similar to the one in Faster R-CNN.

Confidence loss: is the confidence loss which is the softmax loss over multiple classes confidences.
this measures how confident the network is of the objectness of the computed bounding box. Categorical cross-entropy is used to compute this loss.

multibox_loss = confidence_loss + alpha * location_loss


### Non Maxmimum Supression (NMS)
Given the large number of boxes generated during a forward pass of SSD at inference time , it is essential to prune most of the bounding box by applying a technique known as non-maximum suppression: boxes with a confidence loss threshold less than ct (e.g. 0.01) and IoU less than lt (e.g. 0.45) are discarded, and only the top N predictions are kept. This ensures only the most likely predictions are retained by the network, while the more noisier ones are removed.


### Prediction
For object detection, we feed an image into the SSD model, the priors of the features maps will generate a set of bounding boxes and labels for an object. To remove duplicate bounding boxes, non-maximum suppression is used to have final bounding box for one object.


# Using backbone networks in SSD: 
In this section, I explain the details of how I used different backbone networks for SSD object detection.   


### VGG


| Layer | Feature Size |
| :---: |   :---:     |
|       |      38      |
|       |      19      |
|       |      10      |
|       |      5      |
|       |      3      |
|       |      1      |


### ResnetV1


### ResnetV2


### MobilenetV1


### MobilenetV2


### InceptionV4


### InceptionResnetV2





