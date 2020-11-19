

import os
import copy
import random
import numpy as np
import matplotlib.pyplot as plt


# tensorflow1.14~1.15
import tensorflow as tf




# ===================================================================================================
class ReviewClassifier:
    def __init__(self, graph, model_name, n_features=256, n_classes=8, d_layers=3,
                 learn_rate=1e-5, filter=32, drop_out=0.7, cost_wgt=[]):
        #
        self.Graph = graph
        self.N_FEATS = n_features
        self.ModelName = model_name
        print(model_name)
        #
        NUM_CLASSES = n_classes
        DROP_OUT = drop_out
        #
        self.DENSE_LAYERS = d_layers
        self.DENSE_NODE = filter
        #
        tf.set_random_seed(777)
        self.learning_rate = learn_rate
        #
        # ----------------------------------------------------------------------
        # Make Network
        self.X = tf.placeholder(tf.float32, (None, self.N_FEATS))
        self.Y = tf.placeholder(tf.int64, [None, NUM_CLASSES])
        self.IS_TRAIN = tf.placeholder(tf.bool)
        #
        self.net = self.X
        for i in range(self.DENSE_LAYERS):
            self.net = tf.layers.dense(self.net, self.DENSE_NODE, activation=tf.nn.relu)
            self.net = tf.layers.dropout(self.net, DROP_OUT)
            self.DENSE_NODE = self.DENSE_NODE * 2
        #
        self.logits = tf.layers.dense(self.net, NUM_CLASSES,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.logits = tf.nn.softmax(self.logits)
        self.logits_origin = tf.nn.softmax(self.logits)
        if cost_wgt != [] and len(cost_wgt) == self.logits.shape[1]._value:
            print('apply weight:', cost_wgt)
            self.logits = tf.multiply(self.logits, cost_wgt)
            self.logits = tf.nn.softmax(self.logits)
        #
        print('net:', self.net)
        #
        # ----------------------------------------------------------------------
        # for Batch Normalization
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.Y))
        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        #
        # ----------------------------------------------------------------------
        # Prediction and Accuracy
        self.predict = tf.argmax(self.logits, 1)
        self.correct_prediction = tf.equal(self.predict, tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        #
        # ----------------------------------------------------------------------
        # Make Session
        self.sess = tf.Session(graph=self.Graph)
        self.sess.run(tf.global_variables_initializer())
        #
        if model_name != '':
            try:
                self.saver = tf.train.Saver()
                self.saver.restore(self.sess, model_name)
            except:
                print("Exception when loading:", model_name)
    #
    # ===================================================================================================
    # Functions
    def train(self, x, y, is_train):
        if is_train:
            _ = self.sess.run(self.optimizer, feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        l, p, c, a = self.sess.run([self.logits, self.predict, self.cost, self.accuracy],
                                   feed_dict={self.X: x, self.Y: y, self.IS_TRAIN: is_train})
        return c, l, p, a
    #
    def test(self, x):
        l, p, l_origin = self.sess.run([self.logits, self.predict, self.logits_origin],
                                       feed_dict={self.X: x, self.IS_TRAIN: False})
        return l, p, l_origin
    #
    def save(self):
        self.saver.save(self.sess, self.ModelName)
        return
    #
    def save(self, model_name):
        self.saver.save(self.sess, model_name)
        return

