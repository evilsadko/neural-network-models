# -*- coding:utf-8 -*-

import numpy as np
import tensorflow as tf
import os

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

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

    def conv2d(self, inputs, conv_index, filters_num, kernel_size, strides):
            old_filter = int(inputs.get_shape()[-1])
            alpha = 0.3 
            with tf.variable_scope("conv2d_" + str(conv_index)): 
                    kernel = tf.Variable(tf.truncated_normal([kernel_size, kernel_size, old_filter, filters_num], stddev=0.1), name="kernal") #0.1
                    #kernel = tf.Variable(tf.glorot_uniform_initializer(),[kernel_size, kernel_size, old_filter, filters_num], name="kernal")
                    #kernel = tf.get_variable("kernal", shape=[kernel_size, kernel_size, old_filter, filters_num], initializer=tf.glorot_uniform_initializer(), regularizer= tf.contrib.layers.l2_regularizer(scale=5e-4))
		    #pad_size = kernel_size//2
		    #pad_mat = np.array([[0,0],[1,1],[1,1],[0,0]])
                    #inputs_pad = tf.pad(inputs,pad_mat)

                    conv = tf.nn.conv2d(inputs, kernel, [1, strides, strides, 1], padding = ('SAME' if strides == 1 else 'VALID'))+tf.constant(0.1, shape=[filters_num]) #0.2 #0.3
                    #conv = tf.add(tf.nn.conv2d(inputs , kernel, [1, strides, strides, 1], padding = ('SAME' if strides == 1 else 'VALID')), tf.constant(0.1, shape=[filters_num]))
                    #conv = tf.nn.relu(conv)
                    #conv = tf.nn.leaky_relu(conv, alpha = 0.3)
                    conv = tf.maximum(alpha*conv,conv,name=str(conv_index)+'_leaky_relu')
                    
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
            print "DETECT LAYER", shape, data_format
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
            print classes, predictions
            return predictions, classes*confidence


    def darknet53(self, inputs, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        with tf.variable_scope('darknet53'):
            #filter: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
            #filters: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution).
            #conv = self.conv2d(inputs, conv_index, 32, 3, 1)
            conv = self.conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1
            conv, conv_index = self.residual_layer(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self.residual_layer(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv, conv_index = self.residual_layer(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route1 = conv
            conv, conv_index = self.residual_layer(conv, conv_index = conv_index, filters_num = 512, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            route2 = conv
            conv, conv_index = self.residual_layer(conv, conv_index = conv_index,  filters_num = 1024, blocks_num = 4, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        return  route1, route2, conv, conv_index


    def yolo_block(self, inputs, filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        conv = self.conv2d_layer(inputs, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = filters_num, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        route = conv
        conv = self.conv2d_layer(conv, filters_num = filters_num * 2, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
        conv = self.batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        conv = self.conv2d_layer(conv, filters_num = out_filters, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index), use_bias = True)
        conv_index += 1
        #print "THIS YOLO BLOCK", conv
        return route, conv, conv_index


    def yolo_inference(self, inputs, num_anchors, num_classes, training = True):
        conv_index = 1
        conv2d_26, conv2d_43, conv, conv_index = self.darknet53(inputs, conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self.yolo_block(conv, 512, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            conv2d_60 = self.conv2d_layer(conv2d_57, filters_num = 256, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_60 = self.batch_normalization_layer(conv2d_60, name = "batch_normalization_" + str(conv_index),training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60, [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[1]], name='upSample_0')
            route0 = tf.concat([unSample_0, conv2d_43], axis = -1, name = 'route_0')
            conv2d_65, conv2d_67, conv_index = self.yolo_block(route0, 256, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

            conv2d_68 = self.conv2d_layer(conv2d_65, filters_num = 128, kernel_size = 1, strides = 1, name = "conv2d_" + str(conv_index))
            conv2d_68 = self.batch_normalization_layer(conv2d_68, name = "batch_normalization_" + str(conv_index), training=training, norm_decay=self.norm_decay, norm_epsilon = self.norm_epsilon)
            conv_index += 1
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68, [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]], name='upSample_1')
            route1 = tf.concat([unSample_1, conv2d_26], axis = -1, name = 'route_1')
            _, conv2d_75, _ = self.yolo_block(route1, 128, num_anchors * (num_classes + 5), conv_index = conv_index, training = training, norm_decay = self.norm_decay, norm_epsilon = self.norm_epsilon)

        with tf.variable_scope('YOLO_DETECT'):    
                detect_1, class_1 = self.detection_layer(conv2d_59, num_classes, _ANCHORS[6:9], [416,416], 'NHWC')
                detect_1 = tf.identity(detect_1, name='detect_1')
                #print "DETCET 1", detect_1
                detect_2, class_2 = self.detection_layer(conv2d_67, num_classes, _ANCHORS[3:6], [416,416], 'NHWC')
                detect_2 = tf.identity(detect_2, name='detect_2') 
                #print "DETCET 2", detect_2
                detect_3, class_3 = self.detection_layer(conv2d_75, num_classes, _ANCHORS[0:3], [416,416], 'NHWC')
                detect_3 = tf.identity(detect_3, name='detect_3')
                #print "DETCET 3", detect_3
                detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
                class_detect = tf.concat([class_1, class_2, class_3], axis=1)
                #print "FINAL", detections, cl_de
        return detections, class_detect
