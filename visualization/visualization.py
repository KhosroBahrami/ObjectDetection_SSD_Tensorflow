
# Visualization of SSD for demo
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as mpcm


class Visualization(object):

    def __init__(self):      
        a=1

    # Visualize bounding boxes
    def plot_bboxes(self, img, classes, scores, bboxes):
        figsize=(10,10)
        linewidth=1.5
        fig = plt.figure(figsize=figsize)
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(classes.shape[0]):
            cls_id = int(classes[i])
            if cls_id >= 0:
                score = scores[i]
                if cls_id not in colors:
                    colors[cls_id] = (random.random(), random.random(), random.random())
                ymin = int(bboxes[i, 0] * height)
                xmin = int(bboxes[i, 1] * width)
                ymax = int(bboxes[i, 2] * height)
                xmax = int(bboxes[i, 3] * width)
                rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                     edgecolor=colors[cls_id], linewidth=linewidth)
                plt.gca().add_patch(rect)
                class_name = str(cls_id)
                plt.gca().text(xmin, ymin - 2, '{:s} | {:.3f}'.format(class_name, score),
                               bbox=dict(facecolor=colors[cls_id], alpha=0.5), fontsize=12, color='white')
        plt.show()


    
