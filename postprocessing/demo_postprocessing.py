
# Postprocessing for demo module 
import numpy as np

from configs.config_common import *



class Demo_Postprocessing(object):

    def __init__(self):      
        a=1

    # Compute the relative bounding boxes from the layer features and reference anchor bounding boxes.
    # Outputs:
    #      numpy array Nx4: ymin, xmin, ymax, xmax
    def decode_bboxes(self, feat_localizations, anchor_bboxes, prior_scaling=[0.1, 0.1, 0.2, 0.2]):
        # Reshape for easier broadcasting.
        l_shape = feat_localizations.shape
        feat_localizations = np.reshape(feat_localizations, (-1, l_shape[-2], l_shape[-1]))
        yref, xref, href, wref = anchor_bboxes
        xref = np.reshape(xref, [-1, 1])
        yref = np.reshape(yref, [-1, 1])
        # Compute center, height and width
        cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
        cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
        w = wref * np.exp(feat_localizations[:, :, 2] * prior_scaling[2])
        h = href * np.exp(feat_localizations[:, :, 3] * prior_scaling[3])
        # bboxes: ymin, xmin, xmax, ymax.
        bboxes = np.zeros_like(feat_localizations)
        bboxes[:, :, 0] = cy - h / 2.
        bboxes[:, :, 1] = cx - w / 2.
        bboxes[:, :, 2] = cy + h / 2.
        bboxes[:, :, 3] = cx + w / 2.
        # Back to original shape.
        bboxes = np.reshape(bboxes, l_shape)
        return bboxes



    # Extract classes, scores and bounding boxes from features in one layer.
    #    Args:
    #      predictions_layer: A SSD prediction layer;
    #      localizations_layer: A SSD localization layer;
    #    Return:
    #      d_scores, d_bboxes
    def select_top_bboxes_layer(self, predictions_layer, localizations_layer, anchors_layer):
        
        select_threshold=FLAGS.demo_selection_threshold
        img_shape=(FLAGS.image_size, FLAGS.image_size)
        num_classes=FLAGS.num_classes
        # First decode localizations features if necessary.
        localizations_layer = self.decode_bboxes(localizations_layer, anchors_layer)
        # Reshape features to: Batches x N x N_labels | 4.
        p_shape = predictions_layer.shape
        batch_size = p_shape[0] if len(p_shape) == 5 else 1
        predictions_layer = np.reshape(predictions_layer, (batch_size, -1, p_shape[-1]))
        l_shape = localizations_layer.shape
        localizations_layer = np.reshape(localizations_layer, (batch_size, -1, l_shape[-1]))

        # Boxes selection: use threshold or score > no-label criteria.
        if select_threshold is None or select_threshold == 0:
            # Class prediction and scores: assign 0. to 0-class
            classes = np.argmax(predictions_layer, axis=2)
            scores = np.amax(predictions_layer, axis=2)
            mask = (classes > 0)
            classes = classes[mask]
            scores = scores[mask]
            bboxes = localizations_layer[mask]
        else:
            sub_predictions = predictions_layer[:, :, 1:]
            idxes = np.where(sub_predictions > select_threshold)
            classes = idxes[-1]+1
            scores = sub_predictions[idxes]
            bboxes = localizations_layer[idxes[:-1]]

        return classes, scores, bboxes




    # Extract classes, scores and bounding boxes from network output layers.
    # Outputs:
    #      classes, scores, bboxes: Numpy arrays...
    def select_top_bboxes(self, predictions_net, localizations_net, anchors_net):

        l_classes = []
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            classes, scores, bboxes = self.select_top_bboxes_layer(
                predictions_net[i], localizations_net[i], anchors_net[i])
            l_classes.append(classes)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        classes = np.concatenate(l_classes, 0)
        scores = np.concatenate(l_scores, 0)
        bboxes = np.concatenate(l_bboxes, 0)
        return classes, scores, bboxes



    # Sort bounding boxes by decreasing order and keep only the top_k
    def sort_bboxes(self, classes, scores, bboxes):
        top_k=400
        idxes = np.argsort(-scores)
        classes = classes[idxes][:top_k]
        scores = scores[idxes][:top_k]
        bboxes = bboxes[idxes][:top_k]
        return classes, scores, bboxes



    # Computing jaccard index between bboxes1 and bboxes2.
    def bboxes_jaccard(self, bboxes1, bboxes2):
        bboxes1 = np.transpose(bboxes1)
        bboxes2 = np.transpose(bboxes2)
        # Intersection bbox and volume.
        int_ymin = np.maximum(bboxes1[0], bboxes2[0])
        int_xmin = np.maximum(bboxes1[1], bboxes2[1])
        int_ymax = np.minimum(bboxes1[2], bboxes2[2])
        int_xmax = np.minimum(bboxes1[3], bboxes2[3])

        int_h = np.maximum(int_ymax - int_ymin, 0.)
        int_w = np.maximum(int_xmax - int_xmin, 0.)
        int_vol = int_h * int_w
        # Union volume.
        vol1 = (bboxes1[2] - bboxes1[0]) * (bboxes1[3] - bboxes1[1])
        vol2 = (bboxes2[2] - bboxes2[0]) * (bboxes2[3] - bboxes2[1])
        jaccard = int_vol / (vol1 + vol2 - int_vol)
        return jaccard


    # Apply non-maximum selection to bounding boxes.
    def non_maximum_supression_bboxes(self, classes, scores, bboxes):
        nms_threshold=FLAGS.demo_nms_threshold
        keep_bboxes = np.ones(scores.shape, dtype=np.bool)
        for i in range(scores.size-1):
            if keep_bboxes[i]:
                # Computer overlap with bboxes which are following.
                overlap = self.bboxes_jaccard(bboxes[i], bboxes[(i+1):])
                # Overlap threshold for keeping + checking part of the same class
                keep_overlap = np.logical_or(overlap < nms_threshold, classes[(i+1):] != classes[i])
                keep_bboxes[(i+1):] = np.logical_and(keep_bboxes[(i+1):], keep_overlap)
        idxes = np.where(keep_bboxes)
        return classes[idxes], scores[idxes], bboxes[idxes]






