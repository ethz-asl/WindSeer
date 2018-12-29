'''
Script to visualize the planning benchmark results stored in a hdf5 database.
'''

#----------------------------------------------------------
# default values
database_name_default = 'build/planning_results.hdf5'
#----------------------------------------------------------

import argparse
import h5py
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Script to visualize the planning results stored in a HDF5 database')
    parser.add_argument('-d', dest='database_name', default=database_name_default, help='Database file name')
    args = parser.parse_args()

    # open the database
    database = h5py.File(args.database_name, 'r')

    # extract the database
    sample_dict = {}
    model_dict = {}
    model_dict['reference'] = {}
    model_dict['reference']['execution_valid'] = []
    model_dict['reference']['planned_valid'] = []
    model_dict['reference']['planned_cost'] = []
    model_dict['reference']['execution_cost'] = []
    model_dict['reference']['normalized_planned_cost'] = []
    model_dict['reference']['normalized_execution_cost'] = []
    model_dict['reference']['execution_planned_ratio'] = []

    for sample_key in list(database.keys()):
        sample = database[sample_key]
        sample_dict[sample_key] = {}

        for configuration_key in list(sample.keys()):
            configuration = sample[configuration_key]
            sample_dict[sample_key][configuration_key] = {}

            reference_cost = configuration['reference_cost'][0]
            reference_valid = configuration['path_possible'][0]

            sample_dict[sample_key][configuration_key]['reference_cost'] = reference_cost
            sample_dict[sample_key][configuration_key]['predictions'] = {}

            model_dict['reference']['execution_valid'].append(reference_valid)
            model_dict['reference']['planned_valid'].append(reference_valid)
            model_dict['reference']['planned_cost'].append(reference_cost)
            model_dict['reference']['execution_cost'].append(reference_cost)
            model_dict['reference']['normalized_planned_cost'].append(1.0)
            model_dict['reference']['normalized_execution_cost'].append(1.0)
            model_dict['reference']['execution_planned_ratio'].append(1.0)

            for prediction_key in list(configuration['predictions'].keys()):
                prediction = configuration['predictions'][prediction_key]
                sample_dict[sample_key][configuration_key]['predictions'][prediction_key] = {}

                planned_cost = prediction['planned_cost'][0]
                execution_cost = prediction['execution_cost'][0]
                execution_valid = np.isfinite(planned_cost) and np.isfinite(execution_cost)
                planned_valid = np.isfinite(planned_cost)



                sample_dict[sample_key][configuration_key]['predictions'][prediction_key]['planned_cost'] = planned_cost
                sample_dict[sample_key][configuration_key]['predictions'][prediction_key]['execution_cost'] = execution_cost

                sample_dict[sample_key][configuration_key]['predictions'][prediction_key]['execution_valid'] = execution_valid
                sample_dict[sample_key][configuration_key]['predictions'][prediction_key]['planned_valid'] = planned_valid

                if (not (prediction_key in model_dict)):
                    model_dict[prediction_key] = {}
                    model_dict[prediction_key]['execution_valid'] = []
                    model_dict[prediction_key]['planned_valid'] = []
                    model_dict[prediction_key]['planned_cost'] = []
                    model_dict[prediction_key]['execution_cost'] = []
                    model_dict[prediction_key]['normalized_planned_cost'] = []
                    model_dict[prediction_key]['normalized_execution_cost'] = []
                    model_dict[prediction_key]['execution_planned_ratio'] = []

                model_dict[prediction_key]['execution_valid'].append(execution_valid)
                model_dict[prediction_key]['planned_valid'].append(planned_valid)
                model_dict[prediction_key]['planned_cost'].append(planned_cost)
                model_dict[prediction_key]['execution_cost'].append(execution_cost)
                if execution_valid and reference_valid:
                    normalized_planned_cost = planned_cost / reference_cost
                    normalized_execution_cost = execution_cost / reference_cost
                    model_dict[prediction_key]['normalized_planned_cost'].append(normalized_planned_cost)
                    model_dict[prediction_key]['normalized_execution_cost'].append(normalized_execution_cost)
                    model_dict[prediction_key]['execution_planned_ratio'].append(execution_cost/planned_cost)
                    sample_dict[sample_key][configuration_key]['predictions'][prediction_key]['normalized_planned_cost'] = normalized_planned_cost
                    sample_dict[sample_key][configuration_key]['predictions'][prediction_key]['normalized_execution_cost'] = normalized_execution_cost

    # general data for plotting
    idx_path_possible = [i for i, e in enumerate(model_dict['reference']['planned_valid']) if e == True]
    idx_path_not_possible = [i for i, e in enumerate(model_dict['reference']['planned_valid']) if e == False]

    labels = []
    fraction_valid_path_found = []
    fraction_planned_valid_exec_valid = []
    fraction_wrong_valid_plan = []
    fraction_path_found_reference_not = []

    planned_valid = []
    normalized_execution_cost = []
    execution_planned_ratio = []

    for key in list(model_dict.keys()):
        if key == 'reference':
            continue

        labels.append(key)

        tmp = [model_dict[key]['execution_valid'][i] for i in idx_path_possible]
        fraction_valid_path_found.append(np.sum(tmp) / len(idx_path_possible))

        idx_planned_valid = [i for i, e in enumerate(model_dict[key]['planned_valid']) if e == True]
        tmp = [model_dict[key]['execution_valid'][i] for i in idx_planned_valid]
        fraction_planned_valid_exec_valid.append((np.sum(tmp)) / len(idx_planned_valid))

        tmp = [model_dict[key]['planned_valid'][i] and not model_dict[key]['execution_valid'][i] for i in idx_path_not_possible]
        fraction_wrong_valid_plan.append((np.sum(tmp)) / len(idx_path_not_possible))

        tmp = [model_dict[key]['planned_valid'][i] and model_dict[key]['execution_valid'][i] for i in idx_path_not_possible]
        fraction_path_found_reference_not.append((np.sum(tmp)) / len(idx_path_not_possible))

        normalized_execution_cost.append(model_dict[key]['normalized_execution_cost'])
        execution_planned_ratio.append(model_dict[key]['execution_planned_ratio'])

    # visualize the valid path results
    bar_plot(labels, fraction_valid_path_found, 'Prediction Models', 'Fraction Valid Execution / Valid Reference Path [-]', [0.9, 1.0])

    bar_plot(labels, fraction_planned_valid_exec_valid, 'Prediction Models', 'Fraction Valid Execution / Valid Planned Paths [-]', [0.5, 1.0])

    bar_plot(labels, fraction_wrong_valid_plan, 'Prediction Models', 'Fraction Invalid Execution and Valid Planned Paths / Invalid Preference Paths  [-]', [0.0, 1.0])

    bar_plot(labels, fraction_path_found_reference_not, 'Prediction Models', 'Fraction Valid Execution and Valid Planned Paths / Invalid Preference Paths  [-]', [0.0, 0.1])

    violin_plot(labels, normalized_execution_cost, 'Prediction Models', 'Normalized Execution Cost [-]', [0.8, 1.2])

    violin_plot(labels, execution_planned_ratio, 'Prediction Models', 'Ratio Executed Cost / Planned Cost [-]', [0.5, 1.5])

    plt.show()


def bar_plot(labels, data, xlabel, ylabel, ylim):
    index = np.arange(len(labels))
    bar_width = 0.8

    opacity = 1.0

    fig, ax = plt.subplots()
    fig.patch.set_facecolor('white')

    ax.bar(index, data, bar_width, alpha=opacity, color='b')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(ylim)
    fig.tight_layout()

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
