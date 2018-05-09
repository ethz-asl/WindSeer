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
        dataset = dataset.shuffle(buffer_size=10000)             #
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels

def unet_model_fn(features, labels, mode):      #, params

    # input_layer = tf.feature_column.input_layer(features, params['feature_columns'])
    input_layer = tf.stack([features['Ux_in'], features['Uz_in'], features['isWind']], axis=3)

    depth = 5
    feed_layer = input_layer

    conv = []
    pool = []

    # Down, down down
    for i in range(depth):
        conv.append(tf.layers.conv2d(inputs=feed_layer,
                                     filters=2**(i+3),
                                     kernel_size=3,
                                     padding="same",
                                     activation=tf.nn.leaky_relu))

        pool.append(tf.layers.max_pooling2d(inputs=conv[-1], pool_size=[2, 2], strides=2))
        feed_layer = conv[-1]


    # Dense Layer
    pool2_flat = tf.reshape(pool[-1], [-1, 4*2*128])
    dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.leaky_relu)
    dense2 = tf.layers.dense(inputs=dense, units=1024, activation=tf.nn.leaky_relu)
    feed_layer = tf.reshape(dense2, [-1, 2, 4, 128])

    # Up, up, up
    for i in range(depth-1, -1, -1):
        conv.append(tf.layers.conv2d_transpose(inputs=feed_layer,
                                               filters=2**(i+3),
                                               strides=2,
                                               kernel_size=3,
                                               padding="same",
                                               activation=tf.nn.leaky_relu))
        feed_layer = conv[-1]

    # Output layer (to get to 2 output channels, maybe?)
    conv.append(tf.layers.conv2d(inputs=feed_layer,
                                 filters=2,
                                 kernel_size=1,
                                 padding='same'))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "Ux_out": tf.slice(conv[-1], (0, 0, 0, 0), (-1, -1, -1, 1)),
        "Uz_out": tf.slice(conv[-1], (0, 0, 0, 0), (-1, -1, -1, 2))
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    label_tensor = tf.stack([labels['Ux_out'], labels['Uz_out']], axis=3)
    label_weights = tf.stack([features['isWind'], features['isWind']], axis=3)
    loss = tf.losses.mean_squared_error(labels=label_tensor, predictions=conv[-1])  #, weights=label_weights)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and evaluation data
    train = wind_data.build_tf_dataset('data/mini_train')
    # test = wind_data.build_tf_dataset('data/mini_train')


    wind_unet = tf.estimator.Estimator(
        model_fn=unet_model_fn,
        model_dir="tmp/wind_convnet_model"
    )
    # Logging
    # tensors_to_log = {"probabilities": "softmax_tensor"}
    # logging_hook = tf.train.LoggingTensorHook(
    #         tensors=tensors_to_log, every_n_iter=50)

    # Train the model

    train_input_fn = lambda: dataset_input_fn(train, batch_size=1, num_epochs=1000, shuffle=True)

    wind_unet.train(
            input_fn=train_input_fn,
            steps=20000)
            # hooks=[logging_hook])

    # Test the model
    # test_input_fn = lambda: dataset_input_fn(test, batch_size=1, num_epochs=1, shuffle=False)
    # test_results = wind_unet.evaluate(input_fn=test_input_fn)
    # print(test_results)

if __name__ == "__main__":
  tf.app.run()




""" JUNK
    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in ['Ux_in', 'Uz_in', 'isWind']:
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    ,
        params={'feature_columns': my_feature_columns}        
        


"""