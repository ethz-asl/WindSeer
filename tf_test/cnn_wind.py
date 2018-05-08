# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import read_wind_data as wind_data
from layers import (weight_variable, weight_variable_devonc, bias_variable, conv2d, deconv2d,
                    max_pool, crop_and_concat, pixel_wise_softmax_2, cross_entropy)

tf.logging.set_verbosity(tf.logging.INFO)

def dataset_input_fn(dataset, batch_size=32, num_epochs=1000, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle()             # buffer_size=10000
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels

def unet_model_fn(features, labels, mode):

    input_layer = features

    depth = 5
    input = input_layer

    conv = []
    pool = []

    # Down, down down
    for i in range(depth):
        conv.append(tf.layers.conv2d(inputs=input,
                                     filters=8*(i+1),
                                     kernel_size=[3, 3],
                                     padding="same",
                                     activation=tf.nn.leaky_relu(alpha=0.2)))

        pool.append(tf.layers.max_pooling2d(inputs=conv[-1], pool_size=[2, 2], strides=2))
        input = conv[-1]


    # Dense Layer
    pool2_flat = tf.reshape(pool[-1], [-1, 4*2*128])
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)

def main(unused_argv):
    # Load training and evaluation data
    train = wind_data.build_tf_dataset('data/train')
    test = wind_data.build_tf_dataset('data/train')

    wind_unet = tf.estimator.Estimator(
            model_fn=unet_model_fn, model_dir="tmp/wind_convnet_model")
    # Logging
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #         tensors=tensors_to_log, every_n_iter=50)

    # Train the model
    train_input_fn = dataset_input_fn(train, batch_size=32, num_epochs=1000, shuffle=True)

    wind_unet.train(
            input_fn=train_input_fn,
            steps=20000)
            # hooks=[logging_hook])

    # Test the model
    test_input_fn = dataset_input_fn(test, batch_size=1, num_epochs=1, shuffle=False)
    test_results = wind_unet.evaluate(input_fn=test_input_fn)
    print(test_results)

if __name__ == "__main__":
  tf.app.run()

