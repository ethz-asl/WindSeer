# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import read_wind_data as wind_data
import matplotlib.pyplot as plt
import glob

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
        feed_layer = pool[-1]


    # Dense Layer
    pool2_flat = tf.reshape(pool[-1], [-1, 2*4*128])
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
        "U_out": conv[-1],
        # "U_truth": tf.stack([labels['Ux_out'], labels['Uz_out']], axis=3)
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    label_tensor = tf.stack([labels['Ux_out'], labels['Uz_out']], axis=3)
    label_weights = tf.stack([features['isWind'], features['isWind']], axis=3)
    loss = tf.losses.mean_squared_error(labels=label_tensor, predictions=conv[-1], weights=label_weights)  #

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        # optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.0001)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MSE": tf.metrics.mean_squared_error(
            labels=label_tensor, predictions=conv[-1])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    # Load training and evaluation data
    train = wind_data.build_tf_dataset('data/train')
    test = wind_data.build_tf_dataset('data/test')


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

    wind_unet.train(input_fn=train_input_fn, steps=20000) # hooks=[logging_hook])

    # Make some predictions
    test_input_fn = lambda: dataset_input_fn(test, batch_size=1, num_epochs=1, shuffle=False)
    test_results = wind_unet.predict(test_input_fn)
    all_files = glob.glob('data/test/Y*.csv')

    for i in range(5):
        output = test_results.next()
        wind_out = wind_data.read_wind_csv(all_files[i])
        Ux = wind_out.get('U:0').values.reshape([wind_data.WINDNZ, wind_data.WINDNX])
        Uz = wind_out.get('U:2').values.reshape([wind_data.WINDNZ, wind_data.WINDNX])

        fh, ah = plt.subplots(2, 2)
        fh.set_size_inches(10,5)
        # ah[0].quiver(truth_out['Ux_out'], truth_out['Uz_out'], np.sqrt(truth_out['Ux_out'] ** 2 + truth_out['Uz_out'] ** 2))
        ht0 = ah[0][0].imshow(Ux, origin='lower'); fh.colorbar(ht0, ax=ah[0][0]); ah[0][0].set_title('True Ux')
        ht1 = ah[0][1].imshow(Uz, origin='lower'); fh.colorbar(ht1, ax=ah[0][1]); ah[0][1].set_title('True Uz')
        ht2 = ah[1][0].imshow(output['U_out'][:, :, 0], origin='lower'); fh.colorbar(ht2, ax=ah[1][0]); ah[1][0].set_title('Est Ux')
        ht3 = ah[1][1].imshow(output['U_out'][:, :, 1], origin='lower'); fh.colorbar(ht3, ax=ah[1][1]); ah[1][1].set_title('Est Uz')
        # ah[1].quiver(U[:,:,0], U[:,:,0], np.sqrt(U[:,:,0] ** 2 + U[:,:,1] ** 2))
        # ah[1].set_aspect('equal')
    plt.show()

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