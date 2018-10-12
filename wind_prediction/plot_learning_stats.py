import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


model_name = 'pretrained3_naKd4sF8mK'

# get the log file name
log_name = os.listdir('trained_models/' + model_name + '/learningcurve')[-1]

grad_min_1 = []
grad_max_1 = []
grad_mean_1 = []
grad_min_2 = []
grad_max_2 = []
grad_mean_2 = []
loss = []

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
            loss.append(val.simple_value)

epochs = np.arange(len(loss))
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

fh_loss = ax.plot(epochs, loss, color = 'b', linewidth = 2, label='Loss')

plt.legend()
plt.ylim([-40, 40])
plt.grid(True)
plt.xlabel('epoch')
plt.ylabel('loss | gradient')
plt.tight_layout()
plt.show()
