# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import os

#_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)] #BIG YOLO
_ANCHORS = [(10, 14),  (23, 27),  (37, 58), (81, 82),  (135, 169),  (344, 319)] # SMAL YOLO
class yolo:
    def __init__(self, norm_epsilon, norm_decay, anchors_path, classes_path):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path


    def batch_normalization_layer(self, input_layer, name = None, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        batch = tf.layers.batch_normalization(inputs = input_layer, momentum = norm_decay, epsilon = norm_epsilon, center = True, scale = True, training = training, name = name)
        return tf.nn.leaky_relu(batch, alpha = 0.1)


    def conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias = False, strides = 1):
        conv = tf.layers.conv2d(
            inputs = inputs, filters = filters_num,
            kernel_size = kernel_size, strides = [strides, strides], kernel_initializer = tf.glorot_uniform_initializer(),
            padding = ('SAME' if strides == 1 else 'VALID'), kernel_regularizer = tf.contrib.layers.l2_regularizer(scale = 5e-4), use_bias = use_bias, name = name)
        return conv


    def residual_layer(self, inputs, filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        layer = self.conv2d_layer(inputs, filters_num, kernel_size = 3, strides = 2, name = "conv2d_" + str(conv_index))
        layer = self.batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self.conv2d_layer(layer, filters_num // 2, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            #layer = self.myconvol(layer, conv_index, filters_num // 2, 1, 1)
            layer = self.batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer = self.conv2d_layer(layer, filters_num, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            layer = self.batch_normalization_layer(layer, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def get_size(self, shape, data_format):
            if len(shape) == 4:
                shape = shape[1:]
            return shape[1:3] if data_format == 'NCHW' else shape[0:2]                                                                                                                  

    def detection_layer(self, predictions, num_classes, anchors, img_size, data_format):
            num_anchors = len(anchors)
            shape = predictions.get_shape().as_list()
            grid_size = self.get_size(shape, data_format)
            dim = grid_size[0] * grid_size[1]
            bbox_attrs = 5 + num_classes
            #print "DETECT LAYER", shape, data_format
            if data_format == 'NCHW':
                predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
                predictions = tf.transpose(predictions, [0, 2, 1])

            predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

            stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

            anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

            box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)

            box_centers = tf.nn.sigmoid(box_centers)
            confidence = tf.nn.sigmoid(confidence)

            grid_x = tf.range(grid_size[0], dtype=tf.float32)
            grid_y = tf.range(grid_size[1], dtype=tf.float32)
            a, b = tf.meshgrid(grid_x, grid_y)

            x_offset = tf.reshape(a, (-1, 1))
            y_offset = tf.reshape(b, (-1, 1))

            x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
            x_y_offset = tf.reshape(tf.tile(x_y_offset, [1, num_anchors]), [1, -1, 2])

            box_centers = box_centers + x_y_offset
            box_centers = box_centers * stride

            anchors = tf.tile(anchors, [dim, 1])
            #box_sizes = tf.exp(box_sizes) * anchors
            box_sizes = tf.exp(box_sizes) * tf.to_float(anchors)
            box_sizes = box_sizes * stride
            #print predictions
            detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

            classes = tf.nn.sigmoid(classes)
            predictions = tf.concat([detections, classes], axis=-1)
            #print classes, predictions
            return predictions, classes*confidence

     
    def yolo_inference(self, inputs, num_anchors, num_classes, norm_decay = 0.99, norm_epsilon = 1e-3, training = True):
        conv_index = 1
        with tf.variable_scope('yolo_tiny'):
            for i in range(6):
                inputs = self.conv2d_layer(inputs, filters_num = 16 * pow(2, i), kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
                inputs = self.batch_normalization_layer(inputs, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
                conv_index += 1
                #_conv2d_fixed_padding(inputs, 16 * pow(2, i), 3)
                if i == 4:
                   route_1 = inputs
                if i == 5:
                   #inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=1, padding="SAME", name='pool2')                   
                   inputs = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding="SAME")
                else:
                   #inputs = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2, padding="SAME", name='pool2')   
                   inputs = tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
            print inputs #shape=(1, 13, 13, 512)
            inputs = self.conv2d_layer(inputs, filters_num = 1024, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            inputs = self.batch_normalization_layer(inputs, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            inputs = self.conv2d_layer(inputs, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            inputs = self.batch_normalization_layer(inputs, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1

            route_2 = inputs  

            inputs = self.conv2d_layer(inputs, filters_num = 512, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            inputs = self.batch_normalization_layer(inputs, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1

            inputs = self.conv2d_layer(inputs, filters_num = 255, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
            conv_index += 1
            print inputs
            detect_1, class_1 = self.detection_layer(inputs, num_classes, _ANCHORS[3:6], [416,416], 'NHWC')
            detect_1 = tf.identity(detect_1, name='detect_1')

            print "DETCET 1", detect_1 #DETEC1 (1, 507, 85)

#---------------------------------------------->
            inputs = self.conv2d_layer(route_2, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            inputs = self.batch_normalization_layer(inputs, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1

            unSample_0 = tf.image.resize_nearest_neighbor(inputs, [2 * tf.shape(inputs)[1], 2 * tf.shape(inputs)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, route_1], axis = -1, name = 'route_0')


            inputs = self.conv2d_layer(route0, filters_num = 256, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            inputs = self.batch_normalization_layer(inputs, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            inputs = self.conv2d_layer(inputs, filters_num = 255, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
            conv_index += 1
            print inputs
            detect_2, class_2 = self.detection_layer(inputs, num_classes, _ANCHORS[0:3], [416,416], 'NHWC')
            detect_2 = tf.identity(detect_2, name='detect_2')
            print "DETCET 2", detect_2 #DETEC2 (1, 2028, 85)


#---------------------------------------------->
            detections = tf.concat([detect_1, detect_2], axis=1)
            detections = tf.identity(detections, name='detections')

            class_detect = tf.concat([class_1, class_2], axis=1)
            print "DETCET 3", detections #(1, 2535, 85)
        return detections, class_detect

