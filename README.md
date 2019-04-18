# SSD: Single Shot MultiBox Object Detector based on Tensorflow

This is a TensorFlow implementation of the Single Shot Detector (SSD) for object detection. 

Single Shot Detector (SSD) has been originally published in this research [paper](https://arxiv.org/abs/1512.02325). This repository contains a TensorFlow re-implementation of SSD which is inspired by the previous caffe and tensorflow implementations. However, this code has clear pipelines for train, test and demo; it is modular that can be extended or can be used for new applications; and also supports 7 backbone networks. The backbone networks include VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2.

I will explain the details of using these backbones in SSD object detection, at the end of this document.  


## Introduction
This implementation of SSD based on tensorflow is designed with the following goals:
- Clear Pipeline: it has full pipeline of object detection for demo, test and train with seperate modules.
- More Backbone Networks: it has 7 backbone networks, including: VGG, ResnetV1, ResnetV2, MobilenetV1, MobilenetV2, InceptionV4, InceptionResnetV2. Any new backbone can be easily added to the code.
- Modularity: This code is modular and easy to expand for any specific application or new ideas.


## Prerequisite
The main requirements to run SSD can be installed by:

```bash
pip install tensorflow    # For CPU
pip install tensorflow-gpu  # For GPU

pip install opencv-python  # Opencv for processing images
```


## Datasets
For training & testing, I used Pascal VOC datasets (2007 and 2012). 
To prepare the datasets:
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
Before running the code, you need to touch the configuration based on your needs. There are 5 config files in /configs:
- config_general.py: in this file, you can indicate the backbone model that you want to use for train, test and demo. You should uncomment only one of the models to use as backbone. Also, you can indicate the training mode. 
- config_general.py: this file includes the common parameters that are used in training, testing and demo.   
- config_train.py: this file includes training parameters.  
- config_test.py: this file includes testing parameters.  
- config_demo.py: this file includes demo parameters.  


## Demo of SSD
For demo, you can run SSD for object detection in a single image.  
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
4) Inference, calculate output of the SSD network
5) Postprocessing
6) Visualization & Evaluation

The Output of demo is the image with bounding boxes. The following image shows an example of demo:



## Evaluating (Testing) SSD 
This module evaluates the accuracy of SSD with a pretrained model (stored in /checkpoints/ssd_...) for a testing dataset. 

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
We have 3 modes for training of SSD:
- Training an existing SSD model for a new object detection dataset or new sets of parameters. 
- Training (first step fine-tuning) SSD based on an existing ImageNet classification model.
- Training (second step fine-tuning) SSD based on an existing ImageNet classification model.

The mode should be specified in configs/config_general.py.

The input model of training should be in /checkpoints/[model_name]

the output model of training will be stored in checkpoints/ssd_[model_name]


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
SSD has been designed for object detection in real-time. In SSD, we only need to take one single shot to detect multiple objects within the image, while regional proposal network (RPN) based approaches such as Faster R-CNN needs two steps, first step for generating region proposals, and the second step for detecting the object of each proposal. Thus, SSD is much faster than two steps RPN-based approaches. The following table compares SSD, Faster RCNN and YOLO.

| Object Detection Method | VOC2007 test mAP |  Speed (FPS) | Number of Prior Boxes | Input Resolution |
| :---: |   :---:     | :---: | :---: | :---: |
| Faster R-CNN (VGG16) | 73.2% | 7 | 6000 | 1000*600 |
| YOLOv1 (VGG16) | 63.4% | 45 |  98 | 448*448 |
| SSD300 (VGG16) | 74.3% | 59 | 8732 | 300*300 |
| SSD512 (VGG16) | 76.9% | 22 | 24564 | 512*512 |

### Backbone network & Feature maps
The input of SSD is an image of fixed size, for example, 300x300 for SSD300. The image feeds into a CNN backbone network with several layers and generates multiple feature maps at different scales. Features maps (i.e. the results of the convolutional blocks) represent the features of the image at different scales, therefore using multiple feature maps increases the likelihood of any object (large and small) to be detected, localized and classified. The following figure shows feature maps of a network for a given image at different levels:

![Alt text](figs/vgg_feature.png?raw=true "VGG Feature Map Visualisation (from Brown Uni)")

The CNN backbone network (VGG, Mobilenet, ...) gradually reduces the feature map size and increase the depth as it goes to the deeper layers. The deep layers cover larger receptive fields and construct more abstract representation, while the shallow layers cover smaller receptive fields. By using extracted features at different levels, we can use shallow layers to predict small objects and deeper layers to predict large objects.

The output of SSD is a prediction map. Each location in this map stores classes confidence and bounding box information as if there is indeed an object of interests at every location. Obviously, there will be a lot of false alarms, so a further process is used to select a list of predictions.

For example, for VGG backbone network, the first feature map is generated from layer 23 with a size of 38x38 of depth 512. Every point in the 38x38 feature map represents a part of the image, and the 512 channels are the features for every point. By using the features of 512 channels, we can predict the class label (using classification) and the bounding box (using regression) of the small objects on every point. The second feature map has a size of 19x19, which can be used for larger objects, as the points of the features cover larger receptive fields. Finally, in the last layer, there is only one point in the feature map which is used for big objects.

To train the network, one needs to compare the ground truth (a list of objects) against the prediction map. This is achieved with the help of prior boxes.

### Prior boxes (Anchor points) 
Object detection is a local task, meaning that prediction of an object in top left corner of an image is usually unrelated to predict an object in the bottom right corner of the image. So one needs to measure how relevance each ground truth is to each prediction. The criterion for matching a prior and a ground-truth box is IoU (Intersection Over Union), which is also called Jaccard index. The more overlap, the better match. Also, to have the same block size, the ground-truth boxes should be scaled to the same scale.

The procedure for matching prior boxes with ground-truth boxes is as follows:
- We put one priorbox at each location in the prediction map.
- We compute the intersect over union (IoU) between the priorbox and the ground truth.
- The ground truth object that has the highest IoU is used as the target for each prediction, given its IoU is higher than a threshold.
- For predictions who have no valid match, the target class is set to the background class and they will not be used for calculating the localization loss.
- If there is significant overlapping between a priorbox and a ground truth object, then the ground truth can be used at that location. The class of the ground truth is directly used to compute the classification loss; whereas the offset between the ground truth bounding box and the priorbox is used to compute the location loss.

Also, usually, different sizes for predictions at different scales are used. For example, SSD300 uses 21, 45, 99, 153, 207, 261 as the sizes of the priorbox at its 6 different prediction layers.

In practice, SSD uses a few different types of priorbox, each with a different scale or aspect ratio, in a single layer. Doing so creates different "experts" for detecting objects of different shapes. For example, SSD300 uses 5 types of different priorboxes for its seven prediction layers, whereas the aspect ratio of these priorboxes can be chosen from 1:3, 1:2, 1:1, 2:1 or 3:1. Notice, experts in the same layer take the same underlying input (the same receptive field). They behave differently because they use different parameters (convolutional filters) and use different ground truth fetch by different priorboxes.

### Scales and Aspect Ratios of Prior Boxes
Size of default prior boxes are chosen manually. SSD defines a scale value for each feature map layer. Starting from first feature map, Conv4_3 detects objects at the smallest scale 0.2 (or 0.1 sometimes) and then increases linearly to the last layer (Conv11_2) at a scale of 0.9. Combining the scale value with the target aspect ratios, we compute the width and the height of the default boxes. For layers making 6 predictions, SSD starts with 5 target aspect ratios: 1, 2, 3, 1/2 and 1/3. 
Suppose we have m feature maps for prediction, we can calculate Sk for the k-th feature map. Smin is 0.2, Smax is 0.9. That means the scale at the lowest layer is 0.2 and the scale at the highest layer is 0.9. All layers in between is regularly spaced. Then the width and the height of the default boxes are calculated as:
![Alt text](figs/scale_bb.png?raw=true "Scale of Default Boxes")

For each scale, sk, we have 5 non-square aspect ratios:

![Alt text](figs/ns_bb.png?raw=true "5 Non-Square Bounding Boxes")

Then, SSD adds an extra prior box for aspect ratio of 1:1, as:
![Alt text](figs/s_bb.png?raw=true "1 Square Bounding Box")

Therefore, we can have at most 6 bounding boxes in total with different aspect ratios. For layers with only 4 bounding boxes, 1/3 and 3 are omitted.

Note: YOLO uses k-means clustering on the training dataset to determine those default boundary boxes.


### Number of Prior Boxses: 
The number of prior boxes is calculated as follow. For VGG16 as backbone, 6 feature maps from layers Conv4_3, Conv7, Conv8_2, Conv9_2, Conv10_2 and Conv11_2 are used. 
At Conv4_3, feature map is of size 38×38×512. There are 4 bounding boxes and each bounding box will have (classes + 4) outputs. Thus, at Conv4_3, the output is 38×38×4×(c+4). Suppose there are 20 object classes plus one background class, the output is 38×38×4×(21+4) = 144,400. Each prediction composes of a boundary box and 21 scores for each class (one extra class for no object), and we pick the highest score as the class for the bounded object. Conv4_3 makes a total of 38 × 38 × 4 predictions: four predictions per cell regardless of the depth of the feature maps. As expected, many predictions contain no object. SSD reserves a class “0” to indicate it has no objects. Each prediction includes a boundary box and 21 scores for 21 classes (one class for no object). Making multiple predictions containing boundary boxes and confidence scores is called multibox.

In terms of number of bounding boxes, there are 38×38×4 = 5776 bounding boxes.
Therefore:
- Conv4_3: 38×38×4 = 5776 boxes (4 boxes for each location)
- Conv7: 19×19×6 = 2166 boxes (6 boxes for each location)
- Conv8_2: 10×10×6 = 600 boxes (6 boxes for each location)
- Conv9_2: 5×5×6 = 150 boxes (6 boxes for each location)
- Conv10_2: 3×3×4 = 36 boxes (4 boxes for each location)
- Conv11_2: 1×1×4 = 4 boxes (4 boxes for each location)

If we sum them up, we got 5776 + 2166 + 600 + 150 + 36 +4 = 8732 boxes in total for SSD. 




### MultiBox Detection 

In this step, SSD does the Multi-scale Detection. The resolution of the detection equals the size of its prediction map. Multi-scale detection is achieved by generating prediction maps of different resolutions. For example, SSD300 outputs 6 prediction maps of resolutions 38x38, 19x19, 10x10, 5x5, 3x3, and 1x1 respectively. Use these 56 feature maps, SSD does 8732 local prediction. 


### Hard Negative Mining
Priorbox uses a distance-based metric (IoU) to create ground truth predictions, including backgrounds (no matched objects) and objects. However, there can be an imbalance between foreground samples and background samples. There are a lot more unmatched priors (priors without any object). In consequence, the detector may produce many false negatives due to the lack of a training signal of foreground objects. In other words, the huge number of priors labelled as background make the dataset very unbalanced.

To address this problem, SSD uses Hard Negative Mining (HNM). In HNM, all background samples are sorted by their predicted background scores in the ascending order. Only the top K samples are kept for proceeding to the computation of the loss. K is computed on the fly for each batch to keep a 1:3 ratio between foreground samples and background samples.

However, we make far more predictions than the number of objects presence. So there are much more negative matches than positive matches. This creates a class imbalance which hurts training. We are training the model to learn background space rather than detecting objects. However, SSD still requires negative sampling so it can learn what constitutes a bad prediction. So, instead of using all the negatives, we sort those negatives by their calculated confidence loss. SSD picks the negatives with the top loss and makes sure the ratio between the picked negatives and positives is at most 3:1. This leads to a faster and more stable training.

![Alt text](figs/hnm.png?raw=true "Example of hard negative mining (from Jamie Kang blog)")



### Matching Prior and Ground-truth bounding boxes
SSD predictions are classified as positive matches or negative matches. SSD only uses positive matches in calculating the localization cost (the mismatch of the boundary box). If the corresponding default boundary box (not the predicted boundary box) has an IoU greater than 0.5 with the ground truth, the match is positive. Otherwise, it is negative. 



### Image Augmentation
SSD like many other deep learning applications use data augmentation on training images. This step is crucial to teach the network to become more robust to various object sizes in the input. In image augmentation, SSD generates additional training examples with patches of the original image at different IoU ratios (e.g. 0.1, 0.3, 0.5, etc.) and random patches as well. Moreover, each image is also randomly horizontally flipped with a probability of 0.5, thereby making sure potential objects appear on left and right with similar likelihood.

Data augmentation is important in improving accuracy. Augment data with flipping, cropping and color distortion. To handle variants in various object sizes and shapes, each training image is randomly sampled by one of the following options:

- Use the original,
- Sample a patch with IoU of 0.1, 0.3, 0.5, 0.7 or 0.9,
- Randomly sample a patch. The sampled patch will have an aspect ratio between 1/2 and 2. Then it is resized to a fixed size and we flip one-half of the training data. 
- Color distortions.


### Loss function
In SSD, multibox loss function is the combination of localization loss (regression loss) and confidence loss (classification loss):

Localization loss: This measures how far away the network’s predicted bounding boxes are from the ground truth ones from the training set. It is the smooth L1 (L2) loss between the predicted box (l) and the ground-truth box (g) parameters. These parameters include the offsets for the center point (cx, cy), width (w) and height (h) of the bounding box. This loss is similar to the one in Faster R-CNN.
The localization loss is the mismatch between the ground truth box and the predicted boundary box. SSD only penalizes predictions from positive matches. We want the predictions from the positive matches to get closer to the ground truth. Negative matches can be ignored.

Confidence loss: is the confidence loss which is the softmax loss over multiple classes confidences.
this measures how confident the network is of the objectness of the computed bounding box. Categorical cross-entropy is used to compute this loss.
The confidence loss is the loss in making a class prediction. For every positive match prediction, we penalize the loss according to the confidence score of the corresponding class. For negative match predictions, we penalize the loss according to the confidence score of the class “0”: class “0” classifies no object is detected.

multibox_loss = 1/N *(confidence_loss + α * location_loss)

where N is the number of positive match and α is the weight for the localization loss.


### Non Maxmimum Supression (NMS)
Given the large number of boxes generated during a forward pass of SSD at inference time, it is essential to prune most of the bounding box by applying a technique known as non-maximum suppression (NMS). in NMS, the boxes with a confidence loss threshold less than ct (e.g. 0.01) and IoU less than lt (e.g. 0.45) are discarded, and only the top N predictions are kept. This ensures only the most likely predictions are retained by the network, while the more noisier ones are removed.



### Prediction
For object detection, we feed an image into the SSD model, the priors of the features maps will generate a set of bounding boxes and labels for an object. To remove duplicate bounding boxes, non-maximum suppression is used to have final bounding box for one object.


# Using backbone networks in SSD: 
In this section, I explain how I used different backbone networks for SSD object detection.   


### VGG16

To use VGG as backbone, I add 4 auxiliary convolution layers after the VGG16. For object detection, 2 features maps from original layers of VGG16 and 4 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Map Size | Layer Type  |  
| :---: |   :---:      |  :---: | 
|   layer 4   |    38*38     |  Original |
|   layer 7   |    19*19     |  Original | 
|   layer 8   |    10*10     |  auxiliary |  
|   layer 9   |     5*5      |  auxiliary | 
|   layer 10  |      3*3     |  auxiliary | 
|   layer 11  |      1*1     |  auxiliary | 


### ResnetV1

To use ResnetV1 as backbone, I add 3 auxiliary convolution layers after the ResnetV1. For object detection, 3 features maps from original layers of ResnetV1 and 3 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Map Size | Layer Type  | 
| :---: |   :---:      |  :---: |
|   resnet_v1/block2   |    19*19     |  Original |
|   resnet_v1/block3   |    10*10     |  Original |  
|   resnet_v1/block4   |    10*10     |  Original | 
|   layer 13   |     5*5      |  auxiliary | 
|   layer 14  |      3*3     |  auxiliary | 
|   layer 15  |      1*1     |  auxiliary | 


### ResnetV2

To use ResnetV2 as backbone, I add 3 auxiliary convolution layers after the ResnetV2. For object detection, 3 features maps from original layers of ResnetV2 and 3 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Map Size | Layer Type  |  
| :---: |   :---:      |  :---: | 
|   resnet_v2/block2   |    19*19     |  Original |
|   resnet_v2/block3   |    10*10     |  Original | 
|   resnet_v2/block4   |    10*10     |  Original | 
|   layer 13   |     5*5      |  auxiliary |   
|   layer 14  |      3*3     |  auxiliary | 
|   layer 15  |      1*1     |  auxiliary | 


### MobilenetV1

To use MobilenetV1 as backbone, I add 4 auxiliary convolution layers after the MobilenetV1. For object detection, 2 features maps from original layers of MobilenetV1 and 4 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Size | Layer Type  |  
| :---: |   :---:      |  :---: | 
|   Conv2d_5_pointwise   |    38*38     |  Original |
|   Conv2d_11_pointwise   |    19*19     |  Original | 
|   layer 12   |    10*10     |  auxiliary |  
|   layer 13   |     5*5      |   auxiliary |  
|   layer 14  |      3*3     |  auxiliary | 
|   layer 15  |      1*1     |  auxiliary | 


### MobilenetV2

To use MobilenetV2 as backbone, I add 4 auxiliary convolution layers after the MobilenetV2. For object detection, 2 features maps from original layers of MobilenetV2 and 4 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Size | Layer Type  |  
| :---: |   :---:      |  :---: | 
|   Conv2d_5_pointwise   |    38*38     |  Original |
|   Conv2d_11_pointwise   |    19*19     |  Original | 
|   layer 12   |    10*10     |  auxiliary |  
|   layer 13   |     5*5      |  auxiliary |   
|   layer 14  |      3*3     |  auxiliary | 
|   layer 15  |      1*1     |  auxiliary | 


### InceptionV4

To use InceptionV4 as backbone, I add 2 auxiliary convolution layers after the VGG16. For object detection, 4 features maps from original layers of InceptionV4 and 2 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Size | Layer Type  |  
| :---: |   :---:      |  :---: | 
|   Mixed_6d   |    17*17     |  Original |
|   Mixed_6h   |    17*17     |  Original | 
|   Mixed_7b   |    8*8     |  Original | 
|   Mixed_7d   |     8*8      |  Original |  
|   layer 13  |      4*4     |  auxiliary | 
|   layer 14  |      2*2     |  auxiliary | 

### InceptionResnetV2

To use InceptionResnetV2 as backbone, I add 2 auxiliary convolution layers after the InceptionResnetV2. For object detection, 4 features maps from original layers of InceptionResnetV2 and 2 feature maps from added auxiliary layers (totally 6 feature maps) are used in multibox detection. 

| Layer | Feature Size | Layer Type  |  
| :---: |   :---:      |  :---: | 
|   Conv2d_4a_3x3   |    71*71     |  Original |
|   MaxPool_5a_3x3   |    35*35     |  Original | 
|   Mixed_6a   |    17*17     |  Original | 
|   Mixed_7a   |     8*8      |  Original |  
|   layer 13  |      4*4     |  auxiliary | 
|   layer 14  |      2*2     |  auxiliary | 




