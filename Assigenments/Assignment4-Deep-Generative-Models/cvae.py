#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import time
import gzip

import tensorflow as tf
import six
from six.moves import range
from six.moves import cPickle 
import numpy as np
from skimage import io, img_as_ubyte
import zhusuan as zs
import matplotlib.pyplot as plt


def Loaddata(path):
    f = gzip.open(path, 'rb')
    if six.PY2:
        train_set, valid_set, test_set = cPickle.load(f)
    else:
        train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()
    x_train, t_train = train_set[0], train_set[1]
    x_valid, t_valid = valid_set[0], valid_set[1]
    x_test, t_test = test_set[0], test_set[1]
    n_y = t_train.max() + 1
    return x_train, t_train, x_valid, t_valid, x_test, t_test


# From /zhusuan/utils.py and make changes
def save_image_collections(x, filename, shape=(10, 10)):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    n = x.shape[0]
    n_channels = x.shape[3]
    x = img_as_ubyte(x)
    r, c = shape
    if r * c < n:
        print('Shape too small to contain all images')
    h, w = x.shape[1:3]
    ret = np.zeros((h * r, w * c, n_channels), dtype='uint8')
    for i in range(r):
        for j in range(c):
            if i * c + j < n:
                ret[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = x[i * c + j]
    ret = ret.squeeze()
    io.imsave(filename, ret)


@zs.meta_bayesian_net(scope="gen", reuse_variables=True)
def build_gen(x_dim, y, z_dim, n):
    bn = zs.BayesianNet()
    z_mean = tf.zeros([n, z_dim])
    z = bn.normal("z", z_mean, std=1., group_ndims=1)
    y = tf.layers.dense(y, z_dim, activation=tf.nn.relu)
    h = tf.multiply(z, y)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    x_logits = tf.layers.dense(h, x_dim)
    bn.deterministic("x_mean", tf.sigmoid(x_logits)) # x_mean
    bn.bernoulli("x", x_logits, group_ndims=1, dtype=tf.float32) # x
    return bn


@zs.reuse_variables(scope="q_net")
def build_q_net(x, z_dim, y):
    bn = zs.BayesianNet()
    h = tf.layers.dense(x, 500, activation=tf.nn.relu)
    h = tf.layers.dense(h, 500, activation=tf.nn.relu)
    h = tf.multiply(h, tf.layers.dense(y, 500, activation=tf.nn.relu))
    z_mean = tf.layers.dense(h, z_dim)
    z_logstd = tf.layers.dense(h, z_dim)
    bn.normal("z", z_mean, logstd=z_logstd, group_ndims=1)
    return bn


def build_train(meta_model, variational, x, y):
    lower_bound = zs.variational.elbo(meta_model, {"x": x}, variational=variational)
    cost = tf.reduce_mean(lower_bound.sgvb())
    lower_bound = tf.reduce_mean(lower_bound)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    infer_op = optimizer.minimize(cost)
    return infer_op, lower_bound


def random_generation(meta_model):
    x_gen = tf.reshape(meta_model.observe()["x_mean"], [-1, 28, 28, 1])
    return x_gen


def main():
    # Load MNIST
    x_train, t_train, x_valid, t_valid, x_test, t_test = \
        Loaddata("mnist.pkl.gz")
    x_train = np.random.binomial(1, x_train, size=x_train.shape).astype(np.float32)
    x_test = np.random.binomial(1, x_test, size=x_test.shape).astype(np.float32)
    y_train = np.float32(np.eye(10)[t_train])
    y_test = np.float32(np.eye(10)[t_test]) 
    x_dim = x_train.shape[1]
    y_dim = y_train.shape[1]

    y_gen = [0] * 100
    for i in range(10):
        y_gen[10 * i : 10 * (i+1)] = [i] * 10
    y_gen = np.float32(np.eye(10)[y_gen])

    # Define model parameters
    z_dim = 40

    # Build the computation graph
    x = tf.placeholder(tf.float32, shape=[None, x_dim], name="x")
    y = tf.placeholder(tf.float32, shape=[None, y_dim], name="y")
    n = tf.placeholder(tf.int32, shape=[], name="n")

    meta_model = build_gen(x_dim, y, z_dim, n)
    variational = build_q_net(x,z_dim,y)

    infer_op, lower_bound = build_train(meta_model, variational, x, y)

    x_gen = random_generation(meta_model)

    # Define training/evaluation parameters
    epochs = 10
    batch_size = 128
    iters = x_train.shape[0] // batch_size
    result_path = "Results_cvae"

    # Run the inference
    lower_b = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs + 1):
            lbs = []
            for t in range(iters):
                x_batch = x_train[t * batch_size:(t + 1) * batch_size]
                y_batch = y_train[t * batch_size:(t + 1) * batch_size]
                # print(x_batch.shape, x_batch.dtype, y_batch.shape, y_batch.dtype)
                _, lb = sess.run([infer_op, lower_bound],
                                 feed_dict={x: x_batch,
                                            n: batch_size,
                                            y: y_batch})
                lbs.append(lb)
            lower_b.append(np.mean(lbs))
            print("Epoch {}: Lower bound = {}".format(epoch, np.mean(lbs)))

            images = sess.run(x_gen, feed_dict={n: 100, y: y_gen})
            name = os.path.join(result_path, "vae.epoch.{}.png".format(epoch))
            save_image_collections(images, name)

    #Plot the lower_bound change
    plt.figure()
    plt.plot(lower_b)
    plt.title('Lower bound and epochs')
    plt.xlabel('epoches')
    plt.ylabel('Lower bound')
    plt.savefig('lowerbound.png')
    plt.show()


if __name__ == "__main__":
    main()
