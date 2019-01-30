
# Evaluation of SSD object detection 
import math
import sys
import six
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from networks import network_factory
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.contrib.framework.python.ops import variables as contrib_variables
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python import debug as tf_debug
from configs.config_test import *




class Evaluation(object):

    def __init__(self):
        a=1
        

    def evaluate(self, rscores, rbboxes, b_glabels, b_gbboxes, b_gdifficults):
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.test_gpu_memory_fraction)
        config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)       
        tf_global_step = slim.get_or_create_global_step()

        # Compute TP and FP statistics.
        num_gbboxes, tp, fp, rscores = self.bboxes_matching_batch(rscores.keys(), rscores, rbboxes,
                                          b_glabels, b_gbboxes, b_gdifficults)
        # Variables to restore: moving avg. or normal weights.
        variables_to_restore = slim.get_variables_to_restore()
        
        # Evaluation metrics.
        dict_metrics = {}
        # First add all losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
        # Extra losses as well.
        for loss in tf.get_collection('EXTRA_LOSSES'):
            dict_metrics[loss.op.name] = slim.metrics.streaming_mean(loss)
        '''
        # Add metrics to summaries and Print on screen.
        for name, metric in dict_metrics.items():
            summary_name = name
            op = tf.summary.scalar(summary_name, metric[0], collections=[])
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        '''
        tp_fp_metric = self.streaming_tp_fp_arrays(num_gbboxes, tp, fp, rscores)
        for c in tp_fp_metric[0].keys():
            dict_metrics['tp_fp_%s' % c] = (tp_fp_metric[0][c], tp_fp_metric[1][c])
        
        # Add to summaries precision/recall values.
        aps_voc07 = {}
        aps_voc12 = {}
        for c in tp_fp_metric[0].keys():
            # Precison and recall values.
            prec, rec = self.precision_recall(*tp_fp_metric[0][c])

            # Average precision VOC07.
            v = self.average_precision_voc07(prec, rec)
            summary_name = 'AP_VOC07/%s' % c
            op = tf.summary.scalar(summary_name, v, collections=[])
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc07[c] = v
            '''
            # Average precision VOC12.
            v = self.average_precision_voc12(prec, rec)
            summary_name = 'AP_VOC12/%s' % c
            op = tf.summary.scalar(summary_name, v, collections=[])
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            aps_voc12[c] = v
            '''
        # Mean average precision VOC07.
        summary_name = 'AP_VOC07/mAP....'
        mAP = tf.add_n(list(aps_voc07.values())) / len(aps_voc07)
        op = tf.summary.scalar(summary_name, mAP, collections=[])
        op = tf.Print(op, [mAP], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        '''
        # Mean average precision VOC12.
        summary_name = 'AP_VOC12/mAP....'
        mAP = tf.add_n(list(aps_voc12.values())) / len(aps_voc12)
        op = tf.summary.scalar(summary_name, mAP, collections=[])
        op = tf.Print(op, [mAP], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        '''
        # Split into values and updates ops.
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(dict_metrics)

        # Evaluation loop.
        if FLAGS.test_max_num_batches:
            num_batches = FLAGS.test_max_num_batches
        else:
            num_batches = math.ceil(dataset.num_samples / float(FLAGS.test_batch_size))
      
        if tf.gfile.IsDirectory(FLAGS.test_checkpoint_path):
                checkpoint_path = tf.train.latest_checkpoint(FLAGS.test_checkpoint_path)
        else:
                checkpoint_path = FLAGS.test_checkpoint_path
        tf.logging.info('Evaluating %s' % checkpoint_path)


        # Evaluation
        start = time.time()
        slim.evaluation.evaluate_once(
                master='',
                checkpoint_path=checkpoint_path,
                logdir=FLAGS.test_eval_dir,
                num_evals=num_batches,
                eval_op=self.flatten(names_to_updates.values()),
                variables_to_restore=variables_to_restore,
                session_config=config)
        # Time calculation
        elapsed = time.time() - start
        print('Time spent : %.3f seconds.' % elapsed)
        print('Time spent per BATCH: %.3f seconds.' % (elapsed / num_batches))







    # Matching a collection of detected boxes with groundtruth values.
    #    The algorithm goes as follows: for every detected box, check
    #    if one grountruth box is matching. If none, then considered as False Positive.
    #    If the grountruth box is already matched with another one, it also counts
    #    as a False Positive. 
    #    Args:
    #      rclasses, rscores, rbboxes: N(x4) Tensors. Detected objects, sorted by score;
    #      glabels, gbboxes: Groundtruth bounding boxes. 
    #    Return: Tuple of:
    #       n_gbboxes: Scalar Tensor with number of groundtruth boxes 
    #       tp_match: (N,)-shaped boolean Tensor containing with True Positives.
    #       fp_match: (N,)-shaped boolean Tensor containing with False Positives.
    def bboxes_matching(self, label, scores, bboxes, glabels, gbboxes, gdifficults, scope=None):

        matching_threshold=FLAGS.test_matching_threshold
        with tf.name_scope(scope, 'bboxes_matching_single', [scores, bboxes, glabels, gbboxes]):
            rsize = tf.size(scores)
            rshape = tf.shape(scores)
            rlabel = tf.cast(label, glabels.dtype)
            # Number of groundtruth boxes.
            gdifficults = tf.cast(gdifficults, tf.bool)
            n_gbboxes = tf.count_nonzero(tf.logical_and(tf.equal(glabels, label), tf.logical_not(gdifficults)))
            # Grountruth matching arrays.
            gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
            grange = tf.range(tf.size(glabels), dtype=tf.int32)
            # True/False positive matching TensorArrays.
            sdtype = tf.bool
            ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
            ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

            # Loop over returned objects.
            def m_condition(i, ta_tp, ta_fp, gmatch):
                r = tf.less(i, rsize)
                return r

            def m_body(i, ta_tp, ta_fp, gmatch):
                # Jaccard score with groundtruth bboxes.
                rbbox = bboxes[i]
                jaccard = self.bboxes_jaccard(rbbox, gbboxes)
                jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)

                # Best fit, checking it's above threshold.
                idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
                jcdmax = jaccard[idxmax]
                match = jcdmax > matching_threshold
                existing_match = gmatch[idxmax]
                not_difficult = tf.logical_not(gdifficults[idxmax])

                # TP: match & no previous match and FP: previous match | no match.
                tp = tf.logical_and(not_difficult, tf.logical_and(match, tf.logical_not(existing_match)))
                ta_tp = ta_tp.write(i, tp)
                fp = tf.logical_and(not_difficult, tf.logical_or(existing_match, tf.logical_not(match)))
                ta_fp = ta_fp.write(i, fp)
                # Update grountruth match.
                mask = tf.logical_and(tf.equal(grange, idxmax), tf.logical_and(not_difficult, match))
                gmatch = tf.logical_or(gmatch, mask)

                return [i+1, ta_tp, ta_fp, gmatch]
            # Main loop definition.
            i = 0
            [i, ta_tp_bool, ta_fp_bool, gmatch] = \
                tf.while_loop(m_condition, m_body, [i, ta_tp_bool, ta_fp_bool, gmatch], parallel_iterations=1,
                              back_prop=False)
            # TensorArrays to Tensors and reshape.
            tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
            fp_match = tf.reshape(ta_fp_bool.stack(), rshape)
            return n_gbboxes, tp_match, fp_match





    # Matching a collection of detected boxes with groundtruth values.
    #    Args:
    #      rclasses, rscores, rbboxes: BxN(x4) Tensors. Detected objects, sorted by score;
    #      glabels, gbboxes: Groundtruth bounding boxes. 
    #    Return: 
    #       n_gbboxes: Scalar Tensor with number of groundtruth boxes 
    #       tp: (B, N)-shaped boolean Tensor containing with True Positives.
    #       fp: (B, N)-shaped boolean Tensor containing with False Positives.
    def bboxes_matching_batch(self, labels, scores, bboxes, glabels, gbboxes, gdifficults, scope=None):

        # Dictionaries as inputs.
        if isinstance(scores, dict) or isinstance(bboxes, dict):
            with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
                d_n_gbboxes = {}
                d_tp = {}
                d_fp = {}
                for c in labels:
                    n, tp, fp, _ = self.bboxes_matching_batch(c, scores[c], bboxes[c],
                                                         glabels, gbboxes, gdifficults)
                    d_n_gbboxes[c] = n
                    d_tp[c] = tp
                    d_fp[c] = fp
                return d_n_gbboxes, d_tp, d_fp, scores

        with tf.name_scope(scope, 'bboxes_matching_batch', [scores, bboxes, glabels, gbboxes]):
            r = tf.map_fn(lambda x: self.bboxes_matching(labels, x[0], x[1], x[2], x[3], x[4]),
                          (scores, bboxes, glabels, gbboxes, gdifficults),
                          dtype=(tf.int64, tf.bool, tf.bool),
                          parallel_iterations=10,
                          back_prop=False,
                          swap_memory=True,
                          infer_shape=True)
            return r[0], r[1], r[2], scores




    # Compute jaccard score between a reference box and a collection of bounding boxes.
    #    Args:
    #      bbox_ref: (N, 4) or (4,) Tensor with reference bounding box(es).
    #      bboxes: (N, 4) Tensor, collection of bounding boxes.
    #    Return:
    #      (N,) Tensor with Jaccard scores.
    def bboxes_jaccard(self, bbox_ref, bboxes, name=None):

        with tf.name_scope(name, 'bboxes_jaccard'):
            # Should be more efficient to first transpose.
            bboxes = tf.transpose(bboxes)
            bbox_ref = tf.transpose(bbox_ref)
            # Intersection bbox and volume.
            int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
            int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
            int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
            int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
            h = tf.maximum(int_ymax - int_ymin, 0.)
            w = tf.maximum(int_xmax - int_xmin, 0.)
            # Volumes.
            inter_vol = h * w
            union_vol = -inter_vol \
                + (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1]) \
                + (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
            jaccard = self.safe_divide(inter_vol, union_vol, 'jaccard')
            return jaccard




    # Divides two values, returning 0 if the denominator is <= 0.
    #    Args:
    #      numerator: A real `Tensor`.
    #      denominator: A real `Tensor`, with dtype matching `numerator`.
    #      name: Name for the returned op.
    #    Returns:
    #      0 if `denominator` <= 0, else `numerator` / `denominator`
    def safe_divide(self, numerator, denominator, name):
        return tf.where( math_ops.greater(denominator, 0), math_ops.divide(numerator, denominator),
            tf.zeros_like(numerator), name=name)




    # Streaming computation of True and False Positive arrays. This metrics
    #    also keeps track of scores and number of grountruth objects.
    def streaming_tp_fp_arrays(self, num_gbboxes, tp, fp, scores,
                               remove_zero_scores=True,
                               metrics_collections=None,
                               updates_collections=None,
                               name=None):

        # Input dictionaries: dict outputs as streaming metrics.
        if isinstance(scores, dict) or isinstance(fp, dict):
            d_values = {}
            d_update_ops = {}
            for c in num_gbboxes.keys():
                scope = 'streaming_tp_fp_%s' % c
                v, up = self.streaming_tp_fp_arrays(num_gbboxes[c], tp[c], fp[c], scores[c],
                                               remove_zero_scores, metrics_collections,
                                               updates_collections, name=scope)
                d_values[c] = v
                d_update_ops[c] = up
            return d_values, d_update_ops

        # Input Tensors...
        with variable_scope.variable_scope(name, 'streaming_tp_fp', [num_gbboxes, tp, fp, scores]):
            num_gbboxes = math_ops.to_int64(num_gbboxes)
            scores = math_ops.to_float(scores)
            stype = tf.bool
            tp = tf.cast(tp, stype)
            fp = tf.cast(fp, stype)
            # Reshape TP and FP tensors and clean away 0 class values.
            scores = tf.reshape(scores, [-1])
            tp = tf.reshape(tp, [-1])
            fp = tf.reshape(fp, [-1])
            # Remove TP and FP both false.
            mask = tf.logical_or(tp, fp)
            if remove_zero_scores:
                rm_threshold = 1e-4
                mask = tf.logical_and(mask, tf.greater(scores, rm_threshold))
                scores = tf.boolean_mask(scores, mask)
                tp = tf.boolean_mask(tp, mask)
                fp = tf.boolean_mask(fp, mask)

            # Local variables accumlating information over batches.
            v_nobjects = self._create_local('v_num_gbboxes', shape=[], dtype=tf.int64)
            v_ndetections = self._create_local('v_num_detections', shape=[], dtype=tf.int32)
            v_scores = self._create_local('v_scores', shape=[0, ])
            v_tp = self._create_local('v_tp', shape=[0, ], dtype=stype)
            v_fp = self._create_local('v_fp', shape=[0, ], dtype=stype)

            # Update operations.
            nobjects_op = state_ops.assign_add(v_nobjects, tf.reduce_sum(num_gbboxes))
            ndetections_op = state_ops.assign_add(v_ndetections, tf.size(scores, out_type=tf.int32))
            scores_op = state_ops.assign(v_scores, tf.concat([v_scores, scores], axis=0), validate_shape=False)
            tp_op = state_ops.assign(v_tp, tf.concat([v_tp, tp], axis=0), validate_shape=False)
            fp_op = state_ops.assign(v_fp, tf.concat([v_fp, fp], axis=0), validate_shape=False)

            # Value and update ops.
            val = (v_nobjects, v_ndetections, v_tp, v_fp, v_scores)
            with ops.control_dependencies([nobjects_op, ndetections_op, scores_op, tp_op, fp_op]):
                update_op = (nobjects_op, ndetections_op, tp_op, fp_op, scores_op)

            if metrics_collections:
                ops.add_to_collections(metrics_collections, val)
            if updates_collections:
                ops.add_to_collections(updates_collections, update_op)
            return val, update_op



    # Creates a new local variable.
    #    Args:
    #        name: The name of the new or existing variable.
    #        shape: Shape of the new or existing variable.
    #        collections: A list of collection names to which the Variable will be added.
    #        validate_shape: Whether to validate the shape of the variable.
    #        dtype: Data type of the variables.
    #    Returns:
    #        The created variable.
    def _create_local(self, name, shape, collections=None, validate_shape=True, dtype=dtypes.float32):
        collections = list(collections or [])
        collections += [ops.GraphKeys.LOCAL_VARIABLES]
        return variables.Variable(initial_value=array_ops.zeros(shape, dtype=dtype),
                name=name, trainable=False, collections=collections, validate_shape=validate_shape)



    # Compute precision and recall from scores, true positives and false positives booleans arrays
    def precision_recall(self, num_gbboxes, num_detections, tp, fp, scores, dtype=tf.float64, scope=None):
        # Input dictionaries: dict outputs as streaming metrics.
        if isinstance(scores, dict):
            d_precision = {}
            d_recall = {}
            for c in num_gbboxes.keys():
                scope = 'precision_recall_%s' % c
                p, r = self.precision_recall(num_gbboxes[c], num_detections[c],
                                        tp[c], fp[c], scores[c], dtype, scope)
                d_precision[c] = p
                d_recall[c] = r
            return d_precision, d_recall
        # Sort by score.
        with tf.name_scope(scope, 'precision_recall', [num_gbboxes, num_detections, tp, fp, scores]):
            # Sort detections by score.
            scores, idxes = tf.nn.top_k(scores, k=num_detections, sorted=True)
            tp = tf.gather(tp, idxes)
            fp = tf.gather(fp, idxes)
            # Computer recall and precision.
            tp = tf.cumsum(tf.cast(tp, dtype), axis=0)
            fp = tf.cumsum(tf.cast(fp, dtype), axis=0)
            recall = self._safe_div(tp, tf.cast(num_gbboxes, dtype), 'recall')
            precision = self._safe_div(tp, tp + fp, 'precision')
            return tf.tuple([precision, recall])




    # Compute (interpolated) average precision from precision and recall Tensors.
    #    The implementation follows Pascal 2012 and ILSVRC guidelines.
    def average_precision_voc12(self, precision, recall, name=None):

        with tf.name_scope(name, 'average_precision_voc12', [precision, recall]):
            # Convert to float64 to decrease error on Riemann sums.
            precision = tf.cast(precision, dtype=tf.float64)
            recall = tf.cast(recall, dtype=tf.float64)

            # Add bounds values to precision and recall.
            precision = tf.concat([[0.], precision, [0.]], axis=0)
            recall = tf.concat([[0.], recall, [1.]], axis=0)
            # Ensures precision is increasing in reverse order.
            precision = self.cummax(precision, reverse=True)

            # Riemann sums for estimating the integral.
            mean_pre = precision[1:]
            diff_rec = recall[1:] - recall[:-1]
            ap = tf.reduce_sum(mean_pre * diff_rec)
            return ap



    # Compute (interpolated) average precision from precision and recall Tensors.
    #    The implementation follows Pascal 2007 guidelines.
    #        See also: https://sanchom.wordpress.com/tag/average-precision/
    def average_precision_voc07(self, precision, recall, name=None):
        with tf.name_scope(name, 'average_precision_voc07', [precision, recall]):
            # Convert to float64 to decrease error on cumulated sums.
            precision = tf.cast(precision, dtype=tf.float64)
            recall = tf.cast(recall, dtype=tf.float64)
            # Add zero-limit value to avoid any boundary problem...
            precision = tf.concat([precision, [0.]], axis=0)
            recall = tf.concat([recall, [np.inf]], axis=0)
            # Split the integral into 10 bins.
            l_aps = []
            for t in np.arange(0., 1.1, 0.1):
                mask = tf.greater_equal(recall, t)
                v = tf.reduce_max(tf.boolean_mask(precision, mask))
                l_aps.append(v / 11.)
            ap = tf.add_n(l_aps)
            return ap





    # Divides two values, returning 0 if the denominator is <= 0.
    #    Args:
    #      numerator: A real `Tensor`.
    #      denominator: A real `Tensor`, with dtype matching `numerator`.
    #      name: Name for the returned op.
    #    Returns:
    #      0 if `denominator` <= 0, else `numerator` / `denominator`
    def _safe_div(self, numerator, denominator, name):
        return tf.where(math_ops.greater(denominator, 0), math_ops.divide(numerator, denominator),
            tf.zeros_like(numerator), name=name)


    

    # Compute the cumulative maximum of the tensor `x` along `axis`. This
    #    operation is similar to the more classic `cumsum`. Only support 1D Tensor for now.
    #    Args:
    #    x: A `Tensor`. Must be one of the following types: `float32`, `float64`,
    #       `int64`, `int32`, `uint8`, `uint16`, `int16`, `int8`, `complex64`,
    #       `complex128`, `qint8`, `quint8`, `qint32`, `half`.
    #       axis: A `Tensor` of type `int32` (default: 0).
    #       reverse: A `bool` (default: False).
    #       name: A name for the operation (optional).
    #    Returns:
    #    A `Tensor`. Has the same type as `x`.
    def cummax(self, x, reverse=False, name=None):
        with ops.name_scope(name, "Cummax", [x]) as name:
            x = ops.convert_to_tensor(x, name="x")
            # Not very optimal: should directly integrate reverse into tf.scan.
            if reverse:
                x = tf.reverse(x, axis=[0])
            # 'Accumlating' maximum: ensure it is always increasing.
            cmax = tf.scan(lambda a, y: tf.maximum(a, y), x, initializer=None, parallel_iterations=1,
                           back_prop=False, swap_memory=False)
            if reverse:
                cmax = tf.reverse(cmax, axis=[0])
            return cmax





    def flatten(self, x): 
             result = [] 
             for el in x: 
                  if isinstance(el, tuple): 
                        result.extend(self.flatten(el))
                  else: 
                        result.append(el) 
             return result







    



