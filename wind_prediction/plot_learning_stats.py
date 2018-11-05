import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


model_name = 'model'

# get the log file name
log_name = os.listdir('trained_models/' + model_name + '/learningcurve')[-1]

grad_min_1 = []
grad_max_1 = []
grad_mean_1 = []
grad_min_2 = []
grad_max_2 = []
grad_mean_2 = []
train_loss = []
val_loss = []

for event in tf.train.summary_iterator('trained_models/' + model_name + '/learningcurve/' + log_name):
    for val in event.summary.value:
        if val.tag == '_ModelEDNN3D_Twin__model_mean/_ModelEDNN3D__conv/0/weight/grad':
            grad_min_1.append(val.histo.min)
            grad_max_1.append(val.histo.max)
            grad_mean_1.append(val.histo.sum/val.histo.num)

        elif val.tag == '_ModelEDNN3D_Twin__model_uncertainty/_ModelEDNN3D__conv/0/weight/grad':
            grad_min_2.append(val.histo.min)
            grad_max_2.append(val.histo.max)
            grad_mean_2.append(val.histo.sum/val.histo.num)

        elif val.tag == 'Train/Loss':
            train_loss.append(val.simple_value)

        elif val.tag == 'Val/Loss':
            val_loss.append(val.simple_value)

epochs = np.arange(len(train_loss))


fig = plt.figure()
fig.patch.set_facecolor('white')
plt.plot(epochs, train_loss, color = 'b', linewidth = 2)
plt.plot(epochs, val_loss, color = 'g', linewidth = 2)
plt.ylim([0.0, 0.1])
plt.xlim([0, 1000])
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend(['Train Loss', 'Validation Loss'])

if grad_max_1:
    print(max(grad_max_1))
    print(min(grad_min_1))
    fig, ax= plt.subplots()
    fig.patch.set_facecolor('white')
    plt.plot(epochs[::2], grad_max_1, color = 'r', linewidth = 1)
    plt.plot(epochs[::2], grad_min_1, color = 'r', linewidth = 1)
    plt.plot(epochs[::2], grad_mean_1, color = 'r', linewidth = 2, label='Gradient')
    ax.fill_between(epochs[::2], grad_min_1, grad_max_1, color = 'r', alpha=0.1)
    # plt.plot(epochs[::2], grad_max_2, color = 'g', linewidth = 1)
    # plt.plot(epochs[::2], grad_min_2, color = 'g', linewidth = 1)
    # fh_gradient = plt.plot(epochs[::2], grad_mean_2, color = 'g', linewidth = 2)
    # ax.fill_between(epochs[::2], grad_min_2, grad_max_2, color = 'g', alpha=0.1)

    fh_train_loss = ax.plot(epochs, train_loss, color = 'b', linewidth = 2, label='train_loss')

    plt.legend()
    plt.ylim([-40, 40])
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('train_loss | gradient')
    plt.tight_layout()
plt.show()
