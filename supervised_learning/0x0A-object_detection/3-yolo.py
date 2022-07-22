#!/usr/bin/env python3
"""Initialize Yolo"""


import tensorflow.keras as K
import numpy as np


class Yolo:
    """class Yolo that uses the Yolo v3 algorithm to perform object detectio"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """class constructor.

        model_path: is the path to where a Darknet Keras model is stored
        classes_path: is the path to where the list of class names used for the
        Darknet model, listed in order of index, can be found
        class_t: is a float representing the box score threshold for the
        initial filtering step
        nms_t: is a float representing the IOU threshold for non-max
        suppression
        anchors: is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
        containing all of the anchor boxes:
            outputs: is the number of outputs (predictions) made by the Darknet
            model
            anchor_boxes: is the number of anchor boxes used for each
            prediction 2 => [anchor_box_width, anchor_box_height]"""
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            txt = f.read().split('\n')
            if len(txt[-1]) == 0:
                txt = txt[:-1]
        self.class_names = txt
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """Function that process outputs

        outputs is a list of numpy.ndarrays containing the predictions from
        the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
            anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
        [image_height, image_width]
        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively:
            4 => (x1, y1, x2, y2)
                (x1, y1, x2, y2) should represent the boundary box relative to
                original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the box confidences for
            each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the box’s class
            probabilities for each output, respectively"""
        boxes = []
        box_confidences = []
        box_class_probs = []
        ih, iw = image_size
        for i, op in enumerate(outputs):
            grid_h, grid_w, anchor_boxes, cls = op.shape
            box = np.zeros(op[..., :4].shape)

            t_x = op[..., 0]
            t_y = op[..., 1]
            t_w = op[..., 2]
            t_h = op[..., 3]

            # Calculate anchor boxes

            anchors_w = self.anchors[..., 0]
            # repeating each anchor belong all grids_w
            anchor_w = np.tile(anchors_w[i], grid_w)
            anchor_w = anchor_w.reshape(grid_w, 1, len(anchors_w[i]))

            anchors_h = self.anchors[..., 1]
            # repeating each anchor belong all grids_h
            anchor_h = np.tile(anchors_h[i], grid_h)
            anchor_h = anchor_h.reshape(grid_h, 1, len(anchors_h[i]))

            # Calculate corners
            cx = np.tile(np.arange(grid_w), grid_h)
            cx = cx.reshape(grid_w, grid_w, 1)
            cy = np.tile(np.arange(grid_h), grid_h)
            cy = cy.reshape(grid_h, grid_h).T
            cy = cy.reshape(grid_h, grid_h, 1)

            # prediction of each coordinate
            prediction_x = (1 / (1 + np.exp(-t_x))) + cx
            prediction_y = (1 / (1 + np.exp(-t_y))) + cy
            prediction_w = np.exp(t_w) * anchor_w
            prediction_h = np.exp(t_h) * anchor_h

            # Normalize values
            prediction_x /= grid_w
            prediction_y /= grid_h
            prediction_w /= self.model.input.shape[1]
            prediction_h /= self.model.input.shape[2]

            x1 = (prediction_x - (prediction_w / 2)) * iw
            y1 = (prediction_y - (prediction_h / 2)) * ih
            x2 = (prediction_x + (prediction_w / 2)) * iw
            y2 = (prediction_y + (prediction_h / 2)) * ih

            # Setting coordinates
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            # Predict and set confidence
            confidence = (1 / (1 + np.exp(-op[..., 4])))
            confidence = confidence.reshape(grid_h, grid_w, anchor_boxes, 1)
            box_confidences.append(confidence)

            # Predict class probability
            prob = (1 / (1 + np.exp(-op[..., 5:])))
            box_class_probs.append(prob)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Function that filter boxes, applying IoU

        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for each
            output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the processed box
            confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed
            box class probabilities for each output, respectively
        Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
            number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively"""
        # shape = (grid_h * grid_w * anchor_boxes, 4)
        box = np.concatenate([i.reshape(-1, 4) for i in boxes])
        box_scores = []

        # Calculate scores for each box
        for conf, prob in zip(box_confidences, box_class_probs):
            box_scores.append(conf * prob)

        cls_index = []
        cls_scores = []
        for score in box_scores:
            cls_index.append(np.argmax(score, -1).reshape(-1))
            cls_scores.append(np.max(score, -1).reshape(-1))
        cls_index = np.concatenate(cls_index)
        cls_scores = np.concatenate(cls_scores)

        conditional = np.where(cls_scores >= self.class_t)

        filtered_boxes = box[conditional]
        filtered_classes = cls_index[conditional]
        filtered_scores = cls_scores[conditional]

        return (filtered_boxes, filtered_classes, filtered_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Function that applyies non max supression

        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of the
        filtered bounding boxes
        box_classes: a numpy.ndarray of shape (?,) containing the class number
        for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores for
        each box in filtered_boxes, respectively
        Returns a tuple of (box_predictions, predicted_box_classes,
        predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing the
            class number for box_predictions ordered by class and box score,
            respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions ordered by class and box score,
            respectively"""
        selected = []
        nms_boxes = []
        nms_classes = []
        nms_scores = []
        cls = np.unique(box_classes)

        for obj in cls:
            i = np.where(obj == box_classes)
            boxes = filtered_boxes[i]
            classes = box_classes[i]
            scores = box_scores[i]

            mask = self.calculate_nms(boxes, scores)

            boxes = boxes[mask]
            classes = classes[mask]
            scores = scores[mask]

            nms_boxes.append(boxes)
            nms_classes.append(classes)
            nms_scores.append(scores)
        nms_boxes = np.concatenate(nms_boxes)
        nms_classes = np.concatenate(nms_classes)
        nms_scores = np.concatenate(nms_scores)

        return (nms_boxes, nms_classes, nms_scores)

    def calculate_nms(self, boxes, scores):
        """Function that calculate non max supression

        boxes: a numpy.ndarray of shape(?, 4) containing all of
            the predicted bounding boxes ordered by class and box score
        scores: a numpy.ndarray of shape (?) containing the
            box scores for box_predictions ordered by class and box score,
            respectively

        Return: indexes to keep value"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        selected = []

        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idx = np.argsort(scores)[::-1]

        while idx.size > 0:
            last = idx[0]
            selected.append(last)

            # found te corners of the boxes
            xx1 = np.maximum(x1[last], x1[idx[1:]])
            yy1 = np.maximum(y1[last], y1[idx[1:]])
            xx2 = np.minimum(x2[last], x2[idx[1:]])
            yy2 = np.minimum(y2[last], y2[idx[1:]])

            # Weight and Height of bounding boxes
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)

            overlap = (w * h) / (area[last] + area[idx[1:]] - (w * h))

            conditional = np.where(overlap <= self.nms_t)[0]
            idx = idx[conditional + 1]
        return selected