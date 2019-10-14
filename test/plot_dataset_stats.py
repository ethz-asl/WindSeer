import matplotlib.pyplot as plt
import numpy as np

validation_name = 'dataset_stats_validation2.npy'
test_name = 'dataset_stats_test.npy'
train_name = 'dataset_stats_train.npy'

def main():
    train_data = np.load(train_name).item()
    validation_data = np.load(validation_name).item()
    test_data = np.load(test_name).item()

    labels = ['train', 'validation', 'test']

    ux = [train_data['ux'], validation_data['ux'], test_data['ux']]
    uy = [train_data['uy'], validation_data['uy'], test_data['uy']]
    uz = [train_data['uz'], validation_data['uz'], test_data['uz']]
    turb = [train_data['turb'], validation_data['turb'], test_data['turb']]
    reflow = [train_data['reflow_ratio'], validation_data['reflow_ratio'], test_data['reflow_ratio']]
    div_max = [train_data['max_div'], validation_data['max_div'], test_data['max_div']]
    div_mean = [train_data['mean_div'], validation_data['mean_div'], test_data['mean_div']]
    terrain = [train_data['terrain'], validation_data['terrain'], test_data['terrain']]

    violin_plot(labels, ux, 'datasets', 'ux [m/s]', [0,30])
    violin_plot(labels, uy, 'datasets', 'uy [m/s]', [0,30])
    violin_plot(labels, uz, 'datasets', 'uz [m/s]', [0,10])
    violin_plot(labels, turb, 'datasets', 'k [J/kg]', [0,10])

    violin_plot(labels, reflow, 'datasets', 'reflow ratio [-]', [0,1])

    violin_plot(labels, div_max, 'datasets', 'maximum divergence [-]', [0,1])
    violin_plot(labels, div_mean, 'datasets', 'mean divergence [-]', [0,0.006])
    violin_plot(labels, terrain, 'datasets', 'terrain [m]', [-20,800])

    plt.show()

def violin_plot(labels, data, xlabel, ylabel, ylim):
    index = np.arange(len(labels))

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')

    # need to manually set the factor and make sure that it is not too small, otherwise a numerical underflow will happen
    factor = np.power(len(data[0]), -1.0 / (len(data) + 4))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False, points=300, bw_method=np.max([factor, 0.6]))

    for pc in parts['bodies']:
        pc.set_facecolor('#D43F3A')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    quartile1 = []
    medians = []
    quartile3 = []
    for channel in data:
        quartile1_channel, medians_channel, quartile3_channel = np.percentile(channel, [25, 50, 75])
        quartile1.append(quartile1_channel)
        medians.append(medians_channel)
        quartile3.append(quartile3_channel)

    whiskers = np.array([adjacent_values(sorted(sorted_array), q1, q3) for sorted_array, q1, q3 in zip(data, quartile1, quartile3)])
    whiskersMin, whiskersMax = whiskers[:, 0], whiskers[:, 1]

    inds = np.arange(1, len(medians) + 1)
    ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
    ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
    ax.vlines(inds, whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(inds)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    fig.tight_layout()

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

if __name__== "__main__":
  main()