import numpy as np
import commons as c


def get_min_values():
    # Initialize the result dictionary
    min_values = c.getModelTolerance('brisbane')

    # Find the minimum value for each metric
    for hw in c.hardware:
        for metric, value in c.getModelTolerance(hw).items():
            min_values[metric] = min(min_values[metric], value)

    return min_values


def get_max_values():
    # Initialize the result dictionary
    max_values = c.getModelTolerance('brisbane')

    # Find the minimum value for each metric
    for hw in c.hardware:
        for metric, value in c.getModelTolerance(hw).items():
            max_values[metric] = max(max_values[metric], value)

    return max_values


def get_middle_threshold():
    min_values = get_min_values()
    max_values = get_max_values()
    middle_threshold = {}

    for metric in c.metrics.values():
        if metric == 'fidelity':
            middle_threshold[metric] = np.median([max_values[metric], c.tolerance_values_ideal[metric]])
        else:
            middle_threshold[metric] = np.median([min_values[metric], c.tolerance_values_ideal[metric]])

    return middle_threshold


def get_above_threshold(middle_threshold):
    min_values = get_min_values()
    max_values = get_max_values()
    above_threshold = {}

    for metric in c.metrics.values():
        if metric == 'fidelity':
            diff = middle_threshold[metric] - max_values[metric]
            above_threshold[metric] = min_values[metric] - diff
        else:
            diff = min_values[metric] - middle_threshold[metric]
            above_threshold[metric] = max_values[metric] + diff

    return above_threshold


def main():
    middle_threshold = get_middle_threshold()
    above_threshold = get_above_threshold(middle_threshold)

    print('----------------------------------------')
    print('Middle Threshold: ')
    print(f"Hellinger: {middle_threshold['hellinger']}")
    print(f"Jensenshannon: {middle_threshold['jensenshannon']}")
    print(f"Trace: {middle_threshold['trace']}")
    print(f"Fidelity: {middle_threshold['fidelity']}")
    print(f"Expectation: {middle_threshold['expectation']}")
    print('----------------------------------------')
    print('Above Threshold: ')
    print(f"Hellinger: {above_threshold['hellinger']}")
    print(f"Jensenshannon: {above_threshold['jensenshannon']}")
    print(f"Trace: {above_threshold['trace']}")
    print(f"Fidelity: {above_threshold['fidelity']}")
    print(f"Expectation: {above_threshold['expectation']}")
    print('----------------------------------------')


if __name__ == "__main__":
    main()
