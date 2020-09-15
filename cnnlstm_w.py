#-*- coding: utf-8 -*-
__author__ = "skynet"
import time
import tensorflow as tf
from tensorflow.python.training import moving_averages
from tensorflow.keras import layers
import os
import datetime
import numpy as np
import cv2
#import common
def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
    
LEARNING_RATE_DECAY_FACTOR = 0.98 # The learning rate decay factor
INITIAL_LEARNING_RATE = 1e-3
DECAY_STEPS = 10000
train=True

B_SIZE = 1#12

DIGITS = '0123456789'
LETTERSB = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' #-
CHARS = DIGITS + LETTERSB# + ' '
num_epochs = 100000
num_hidden = 128

print("num_hidden:", num_hidden)
val_dir = 'NEWVK_val'
train_dir = 'NEWVK_tr'
num_classes = len(CHARS) + 1 + 1# + 1
encode_maps = {}
decode_maps = {}
for i, char in enumerate(CHARS, 1):
    encode_maps[char] = i
    decode_maps[i] = char

SPACE_INDEX = 0
SPACE_TOKEN = ''
encode_maps[SPACE_TOKEN] = SPACE_INDEX
decode_maps[SPACE_INDEX] = SPACE_TOKEN
#SPACE_INDEX = 0
#SPACE_TOKEN = ''
#encode_maps[SPACE_TOKEN] = SPACE_INDEX
#decode_maps[SPACE_INDEX] = SPACE_TOKEN

# Utility functions
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride=(1, 1), padding='SAME'):
    return tf.nn.conv2d(x, W, strides=[1, stride[0], stride[1], 1], padding=padding)


def max_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.max_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')


def avg_pool(x, ksize=(2, 2), stride=(2, 2)):
    return tf.nn.avg_pool(x, ksize=[1, ksize[0], ksize[1], 1],
                          strides=[1, stride[0], stride[1], 1], padding='SAME')

#def conv2d(x, name, filter_size, in_channels, out_channels, strides):
#        with tf.variable_scope(name):
#            kernel = tf.get_variable(name='W',
#                                     shape=[filter_size, filter_size, in_channels, out_channels],
#                                     dtype=tf.float32,
#                                     initializer=tf.glorot_uniform_initializer())   tf.glorot_normal_initializer

#            b = tf.get_variable(name='b',
#                                shape=[out_channels],
#                                dtype=tf.float32,
#                                initializer=tf.constant_initializer())

#            con2d_op = tf.nn.conv2d(x, kernel, [1, strides, strides, 1], padding='SAME')

#        return tf.nn.bias_add(con2d_op, b)

#def batch_norm(name, x):
#        """Batch normalization."""
#        with tf.variable_scope(name):
#            x_bn = \
#                tf.contrib.layers.batch_norm(
#                    inputs=x,
#                    decay=0.9,
#                    center=True,
#                    scale=True,
#                    epsilon=1e-5,
#                    updates_collections=None,
#                    is_training= 'train',
#                    fused=True,
#                    data_format='NHWC',
#                    zero_debias_moving_mean=True,
#                    scope='BatchNorm'
#                )

#        return x_bn

#def leaky_relu(x, leakiness=0.0):
#        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

#def max_pool(x, ksize, strides):
#        return tf.nn.max_pool(x,
#                              ksize=[1, ksize, ksize, 1],
#                              strides=[1, strides, strides, 1],
#                              padding='SAME',
#                              name='max_pool')

#def res_net_block(input_data):
#  x = layers.Conv2D(64, (1, 1), activation='relu', padding='same')(input_data)
#  x = layers.BatchNormalization()(x)
#  x = layers.Conv2D(64, (3, 3), activation=None, padding='same')(x)
#  x = layers.BatchNormalization()(x)
#  x = tf.nn.relu(x)
#  x = layers.Conv2D(64, (1, 1), activation=None, padding='same')(x)
#  x = layers.BatchNormalization()(x)
#  x = layers.Add()([x, input_data])
#  x = tf.nn.relu(x)#x = layers.Activation('relu')(x)
#  return x


class CNNLSTM(object):
	def __init__(self, mode):
		self.mode = mode
		# image
		self.inputs = tf.placeholder(tf.float32, [None, 64, 128, 1])#[None, 60, 180, 1])
                #tf.placeholder(tf.float32,[None,None,None])# [None,64,128,1]) # 3 | 1?
		# SparseTensor required by ctc_loss op
		self.labels = tf.sparse_placeholder(tf.int32)
		# 1d array of size [batch_size]
		#self.seq_len = tf.placeholder(tf.int32, [None])
		# l2
		self._extra_train_ops = []
        def build_graph(self):
                self._build_model()
                self.merged_summay = tf.summary.merge_all()
        
        def _build_model(self):
		filters = [1, 64, 128, 128, 64] 
		strides = [1, 2]
                
		with tf.variable_scope('cnn'):
#                    #x_expanded = tf.expand_dims(self.inputs, 1)
#		    # 1
#		    W_conv1 = weight_variable([7, 7, 1, 32]) #[3, 3, 1, 48] v2 -> [7, 7, 1, 12] #60x180
#		    b_conv1 = bias_variable([32]) # 12
#		    h_conv1 = tf.nn.relu(conv2d(self.inputs, W_conv1) + b_conv1)
#		    h_pool1 = max_pool(h_conv1, ksize=(2, 2), stride=(2, 2))      #60x180>30x90

#		    # 2
#		    W_conv2 = weight_variable([5, 5, 32, 64]) # [5, 5, 48, 64] | [5, 5, 12, 24]
#		    b_conv2 = bias_variable([64]) # 24
#		    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#		    h_pool2 = max_pool(h_conv2, ksize=(2, 2), stride=(2, 2))  #30x90>15x45

#		    # 3
#		    W_conv3 = weight_variable([3, 3, 64, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
#		    b_conv3 = bias_variable([128]) #[128] | 48
#		    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
#		    h_pool3 = max_pool(h_conv3, ksize=(2, 2), stride=(2, 2)) #15x45>7,5x22,5
#		    # 4
#		    W_conv4 = weight_variable([3, 3, 128, 128]) #[5, 5, 64, 128] | [5, 5, 24, 48]
#		    b_conv4 = bias_variable([128]) #[128] | 48
#		    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
#		    h_pool4 = max_pool(h_conv4, ksize=(2, 2), stride=(2, 2)) #7,5x22,5>3,75x11,25
#                    # 5
#                    
#		    W_conv5 = weight_variable([3, 3, 128, out_channels]) #[5, 5, 64, 128] | [5, 5, 24, 48]
#		    b_conv5 = bias_variable([out_channels]) #[128] | 48
#		    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
#		    h_pool5 = max_pool(h_conv5, ksize=(1, 1), stride=(1, 1)) # [64,524], In[1]: [560,2048]
#		    #layers.Conv2D(64, (1, 1), activation='relu', padding='same')(input_data)
#		    print (h_pool4.get_shape())
#----------------------------------------__>
                   x = layers.Conv2D(32, (1, 1), activation='relu', padding='same')(self.inputs)
                   x = layers.BatchNormalization()(x)
                   x = max_pool(x, ksize=(2, 2), stride=(2, 2)) 
                   
                   x = layers.Conv2D(64, (3, 3), activation=None, padding='same')(x)
                   x = layers.BatchNormalization()(x)
                   x = max_pool(x, ksize=(2, 2), stride=(2, 2)) 
                   
                   x = layers.Conv2D(128, (1, 1), activation='relu', padding='same')(x)
                   x = layers.BatchNormalization()(x)
                   x = max_pool(x, ksize=(2, 2), stride=(2, 2)) 
                   #---------------------__>
#                   x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#                   x = layers.BatchNormalization()(x)
#                   x = max_pool(x, ksize=(2, 2), stride=(2, 2)) 
#                   x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#                   x = layers.BatchNormalization()(x)
#                   x = max_pool(x, ksize=(2, 2), stride=(2, 2))
#                   x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#                   x = layers.BatchNormalization()(x)
#                   x = max_pool(x, ksize=(2, 2), stride=(2, 2))                   
                   print (x.get_shape())                    
                   #---------------------__>
                   out_channels = 64
                   
#                   x = layers.Conv2D(out_channels, (1, 1), activation='relu', padding='same')(x)
#                   x = layers.BatchNormalization()(x)

                   #x = layers.Conv2D(128, (2, 2), activation='relu', padding="valid")(x)
                   #x = layers.Conv2D(128, (2, 2), activation='relu', padding="valid")(x)
                   #x = layers.Conv2D(128, (1, 2), activation='relu', padding="valid")(x)
                   #x = layers.Conv2D(128, (1, 2), activation='relu', padding="valid")(x)
                   #x = layers.Conv2D(128, (1, 2), activation='relu', padding="valid")(x)
                   x = layers.Conv2D(out_channels, (1, 2), activation='relu', padding="valid")(x)
                   x = layers.BatchNormalization()(x)
                   h_pool5 = max_pool(x, ksize=(2, 2), stride=(2, 1))
                   _, feature_h, feature_w, _ = h_pool5.get_shape().as_list()
#----------------------------------------__>
#                   out_channels = 64
#                   x = self.inputs
#                   for i in range(4):
#                        with tf.variable_scope('unit-%d' % (i + 1)):
#                            x = conv2d(x, 'cnn-%d' % (i + 1), 3, filters[i], filters[i + 1], strides[0])
#                            x = batch_norm('bn%d' % (i + 1), x)
#                            x = leaky_relu(x, 0.01)
#                            x = max_pool(x, 2, strides[1])

#                   _, feature_h, feature_w, _ = x.get_shape().as_list()
#                   h_pool5 = x
#                   print('\nfeature_h: {}, feature_w: {}'.format(feature_h, feature_w))
#------------------------------------------->
                   print (h_pool5.get_shape())                  
		with tf.variable_scope('lstm'): 
                    # [batch_size, max_stepsize, num_features]
                    #In[0]: [64,696], In[1]: [560,2048]
                    #In[0]: [128,152], In[1]: [176,512]
                    #In[0]:[64,524], In[1]: [560,2048]
                    # (?, 4, 15, 64)
                    # (?, 3840)
                    # (?, 64)
                    # (?, 64, 256)
                    #x = tf.squeeze(h_pool5, axis=1, name='features')
#                    x = tf.reshape(h_pool5, [-1, 1 * 1 * out_channels]) 
#                    print (x.get_shape())
#                    x = tf.reshape(x, [B_SIZE, 1, out_channels])
#                    #x = tf.transpose(x, [1, 0, 2])
#                    print (x.get_shape())
                    #x = tf.reshape(h_pool5, [B_SIZE, -1, out_channels])
                    #self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
                    # x.set_shape([FLAGS.batch_size, filters[3], shp[1]])
                    #x.set_shape([B_SIZE, 64, 2])
#---------------------------------------->    
                    #x = tf.squeeze(h_pool5, axis=1, name='features') 
                    #print (x.get_shape()) 
#                    tf.squeeze(input_tensor, axis=1)               
#                    W_fc1 = weight_variable([4 * 15 * 64, 64])
#                    b_fc1 = bias_variable([64])
#                    conv_layer_flat = tf.reshape(h_pool5, [-1, 4 * 15 * 64])
#                    print (conv_layer_flat.get_shape())
#                    features = tf.nn.relu(tf.matmul(conv_layer_flat, W_fc1) + b_fc1)
#                    print (features.get_shape())
#                    shape = tf.shape(features)
#                    x = tf.reshape(features, [shape[0], 64, 60])   #batchsize * outputshape * 1
#                    print (x.get_shape())
#---------------------------------------->                   
                    print (_, feature_h, feature_w, _)
                    x = tf.transpose(h_pool5, [0, 2, 1, 3])
                    print (x.get_shape())
                    x = tf.reshape(x, [B_SIZE, feature_w, feature_h * out_channels])
                    print (x.get_shape())
                    self.seq_len = tf.fill([x.get_shape().as_list()[0]], feature_w)
                    print (self.seq_len.get_shape())
#--------------------------------------->
		    # tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.GRUCell
		    cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True) # работает с 180x60 512 rnn сетей/256
		    if self.mode == 'train':
		        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.8)

		    cell1 = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple=True)
		    if self.mode == 'train':
		        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell=cell1, output_keep_prob=0.8)

		    # Stacking rnn cells
		    stack = tf.nn.rnn_cell.MultiRNNCell([cell, cell1], state_is_tuple=True)
                    initial_state = stack.zero_state(B_SIZE, dtype=tf.float32)
		    # The second output is the last state and we will not use that
		    outputs, _ = tf.nn.dynamic_rnn(cell=stack, 
                                                   inputs=x, 
                                                   sequence_length=self.seq_len, 
                                                   initial_state=initial_state, 
                                                   dtype=tf.float32,
                                                   time_major=False)
                    print (">>>>>>>>>>", outputs.get_shape())
		    # Reshaping to apply the same weights over the timesteps
		    outputs = tf.reshape(outputs, [-1, num_hidden])

		    W = tf.get_variable(name='W',
		                        shape=[num_hidden, num_classes],
		                        dtype=tf.float32,
		                        initializer=tf.glorot_uniform_initializer())
		    b = tf.get_variable(name='b',
		                        shape=[num_classes],
		                        dtype=tf.float32,
		                        initializer=tf.constant_initializer())

		    self.logits = tf.matmul(outputs, W) + b
		    # Reshaping back to the original shape
		    shape = tf.shape(x)
		    self.logits = tf.reshape(self.logits, [shape[0], -1, num_classes])
		    # Time major
		    self.logits = tf.transpose(self.logits, (1, 0, 2))
      
   

# THE MAIN CODE!

#test_inputs, test_targets, test_seq_len = utils.get_data_set('test')
#test_inputs, test_targets, test_seq_len = utils.get_data_set('test')
#print("Data loaded....")
def decode_a_seq(indexes, spars_tensor):
    decoded = []
    #print (indexes, spars_tensor[1])
    for m in indexes:
        #print (spars_tensor[1][m])
        str = decode_maps[spars_tensor[1][m]]#CHARS[spars_tensor[1][m]]
        decoded.append(str)
    # Replacing blank label to none
    #str_decoded = str_decoded.replace(chr(ord('9') + 1), '')
    # Replacing space label to space
    #str_decoded = str_decoded.replace(chr(ord('0') - 1), ' ')
    # print("ffffffff", str_decoded)
    #print (decoded)
    return decoded
def decode_sparse_tensor(sparse_tensor):
    #print("sparse_tensor = ", sparse_tensor[0])
    decoded_indexes = list()
    current_i = 0
    current_seq = []
    for offset, i_and_index in enumerate(sparse_tensor[0]):
        i = i_and_index[0]
        if i != current_i:
            decoded_indexes.append(current_seq)
            current_i = i
            current_seq = list()
        current_seq.append(offset)
    decoded_indexes.append(current_seq)
    #print("decoded_indexes = ", decoded_indexes)
    result = []
    for index in decoded_indexes:
        #print("index = ", index)
        result.append(decode_a_seq(index, sparse_tensor))
    #print(result)
    return result

# graph = tf.Graph()
def report_accuracy(decoded_list, test_targets):
    original_list = decode_sparse_tensor(test_targets)
    detected_list = decode_sparse_tensor(decoded_list)
    true_numer = 0
    #print (detected_list)
    if len(original_list) != len(detected_list):
        print("len(original_list)", len(original_list), "len(detected_list)", len(detected_list),
              " test and detect length desn't match")
        return
    print("T/F: original(length) <-------> detectcted(length)")
    for idx, number in enumerate(original_list):
	detect_number = detected_list[idx]
	if(len(number) == len(detect_number)):
		hit = True
		for idy, value in  enumerate(number):
			detect_value = detect_number[idy]
			if(value != detect_value):
				hit = False
				break
		print(hit, number, "(", len(number), ") <-------> ", detect_number, "(", len(detect_number), ")")
		if hit:
			true_numer = true_numer + 1
    accuraccy = true_numer * 1.0 / len(original_list)
    print("Test Accuracy:", accuraccy)
    return accuraccy


class DataIterator:
    def __init__(self, data_dir):
        self.image = [] # <-LIST
        self.labels = []
        for root, sub_folder, file_list in os.walk(data_dir):
            for file_path in file_list:
                image_name = os.path.join(root, file_path)
                im = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE).astype(np.float32)/255. #0 - cv2.IMREAD_GRAYSCALE
                # resize to same height, different width will consume time on padding
                im = cv2.resize(im, (64, 128))
                im = np.reshape(im, [64, 128, 1])
                self.image.append(im) # IMAGE ADD LIST

                # image is named as /.../<folder>/00000_abcd.png
                code = image_name.split('/')[-1].split('_')[1].split('.')[0]
                #print (code)
                #code = '-'.join(code)
                #print ("START", code)
                code = [SPACE_INDEX if code == SPACE_TOKEN else encode_maps[c] for c in list(code)]
                #print ("CODING", code)
                #END = [SPACE_TOKEN if code == SPACE_INDEX else decode_maps[c] for c in code]
                #print ("END", END)
                #code = [encode_maps[c] for c in list(code)]
                #print (code)
                self.labels.append(code)

    @property
    def size(self):
        return len(self.labels)

    def the_label(self, indexs):
        #print indexs
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels
    
    def input_index_generate_batch(self, index=None):
        #print index
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels


        def get_input_lens(sequences): 
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([B_SIZE for _ in sequences], dtype=np.int64) # + ADD SEQ

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch)) # + ADD SEQ
        batch_labels = sparse_tuple_from_label(label_batch)
        #print batch_labels
        return batch_inputs, batch_seq_len, batch_labels

def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape
def train():
    #test_inputs, test_targets, test_seq_len = utils.get_data_set('LSTM2', 0, 128)#118100, 118200)#120100, 120200) #IMGN1 # GO2 120100, 120200
    S = 'train'
    m = CNNLSTM(S)
    m.build_graph()
    global_step = tf.train.get_or_create_global_step()#tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(INITIAL_LEARNING_RATE,            
                                               global_step,
                                               DECAY_STEPS,
                                               LEARNING_RATE_DECAY_FACTOR, 
                                               staircase=True)
    loss = tf.nn.ctc_loss(labels=m.labels, inputs=m.logits, sequence_length=m.seq_len)
    cost = tf.reduce_mean(loss)
    #cost = model.ctc_loss_layer(logits,targets,seq_len)

    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=common.MOMENTUM).minimize(cost, global_step=global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.9,
                                       beta2=0.999).minimize(loss,
                                                             global_step=global_step)
    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_beam_search_decoder(m.logits, m.seq_len, merge_repeated=False)

    # Accuracy: label error rate
    acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), m.labels))
    
    print('loading train data, please wait---------------------')
    train_feeder = DataIterator(data_dir=train_dir)
    print('get image: ', train_feeder.size)
    
    print('loading validation data, please wait---------------------')
    val_feeder = DataIterator(data_dir=val_dir)
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  # 100000
    num_batches_per_epoch = int(num_train_samples / B_SIZE) # номер партии
    shuffle_idx = np.random.permutation(num_train_samples)

    #-----------------------2й - словарь--------------------------#
    num_val_samples = val_feeder.size #val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / B_SIZE)  # example: 10000/100
    shuffle_idx_val = np.random.permutation(num_val_samples)
   
    
    def do_report():#Информация/сохранение модели
        indexs_val = [shuffle_idx_val[i % num_val_samples] for i in range(batch * B_SIZE, (batch + B_SIZE) * B_SIZE)]
        val_inputs, val_seq_len, val_labels = val_feeder.input_index_generate_batch(indexs_val)
        #test_feed = {m.inputs: val_inputs, m.labels: val_labels, m.seq_len: val_seq_len}
        test_feed = {m.inputs: val_inputs, m.labels: val_labels}
        dd, log_probs, accuracy = session.run([decoded[0], log_prob, acc], test_feed)
        accuracy = report_accuracy(dd, val_labels)
        
        save_path = saver.save(session, "models/ocr.model-" + str(accuracy), global_step=steps)
        # decoded_list = decode_sparse_tensor(dd)

    def do_batch(): #Партия
        #feed = {m.inputs: train_inputs, m.labels: train_labels, m.seq_len: train_seq_len}
        feed = {m.inputs: train_inputs, m.labels: train_labels}
        b_cost, steps, _ = session.run([cost, global_step, optimizer], feed)
        #b_cost, steps, _ = session.run([m.cost, m.global_step, m.train_op], feed)
        #print "ПАРТИЯ"
        if steps > 0 and steps % 10000 == 0:
                  do_report()
        return b_cost, steps
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.30)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)) as session:
        ckpt = tf.train.get_checkpoint_state("models")
        #writer = tf.summary.FileWriter('log/', graph=session.graph)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver()
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            print("no checkpoint found")
            # Initializate the weights and biases
            #init = tf.initialize_all_variables()
            #session.run(init)
            #saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)
            session.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        for curr_epoch in xrange(num_epochs):
            #variables = tf.all_variables()
            #for i in variables:
                 #print(i.name)

            print("Epoch.......", curr_epoch)
            
            train_cost = train_ler = 0
            for batch in range(num_batches_per_epoch):
                #print (batch)
                start = time.time()
                #train_inputs, train_targets, train_seq_len = utils.get_data_set('GO1', batch * common.BATCH_SIZE,
                #                                                                (batch + 1) * common.BATCH_SIZE)
                indexs = [shuffle_idx[i % num_train_samples] for i in range(batch * B_SIZE, (batch + B_SIZE) * B_SIZE)]
                #print indexs
		train_inputs, train_seq_len, train_labels = train_feeder.input_index_generate_batch(indexs)#utils.get_data_set('LSTM2', batch * common.BATCH_SIZE, (batch + 1) * common.BATCH_SIZE)

                

                #print("get data time", time.time() - start)
                start = time.time()
                c, steps = do_batch()
                train_cost += c * B_SIZE
                seconds = time.time() - start
                #print("Step:", steps, ", batch seconds:", seconds)

                #feed = {m.inputs: train_inputs, m.labels: train_targets, m.seq_len: train_seq_len} #готовлю данные
                #summary_str, batch_cost, step, _ = session.run([m.merged_summay, m.cost, m.global_step, m.train_op], feed)
            train_cost /= B_SIZE
            indexs_val = [shuffle_idx_val[i % num_val_samples] for i in range(batch * B_SIZE, (batch + B_SIZE) * B_SIZE)]    
            val_inputs, val_seq_len, val_labels = val_feeder.input_index_generate_batch(indexs_val)
            #val_feed = {m.inputs: val_inputs, m.labels: val_labels, m.seq_len: val_seq_len}
            val_feed = {m.inputs: val_inputs, m.labels: val_labels}
            val_cost, val_ler, lr, steps = session.run([cost, acc, learning_rate, global_step], feed_dict=val_feed)
            log = "Epoch {}/{}, steps = {}, train_cost = {:.3f}, train_ler = {:.3f}, val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}s, learning_rate = {}"
            print(log.format(curr_epoch + 1, num_epochs, steps, train_cost, train_ler, val_cost, val_ler, time.time() - start, lr))

            
if __name__ == '__main__':
    train()
    


