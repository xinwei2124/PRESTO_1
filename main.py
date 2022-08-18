from __future__ import division
import numpy as np
import pycarl
import pycarl.core
import stormpy
import stormpy.core
import stormpy.info

import pycarl
import pycarl.core

import stormpy.examples
import stormpy.examples.files
import matplotlib.pyplot as plt
import stormpy._config as config
from numpy import genfromtxt
from scipy import optimize
import numpy as np


def parametric_model_checking(path, formula_str):
    # Check support for parameters
    if not config.storm_with_pars:
        print("Support parameters is missing. Try building storm-pars.")
        return

    import stormpy.pars
    from pycarl.formula import FormulaType, Relation
    if stormpy.info.storm_ratfunc_use_cln():
        import pycarl.cln.formula
    else:
        import pycarl.gmp.formula

    prism_program = stormpy.parse_prism_program(path)
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    model = stormpy.build_parametric_model(prism_program, properties)

    initial_state = model.initial_states[0]
    result = stormpy.model_checking(model, properties[0])
    parameters = model.collect_probability_parameters()
    return result.at(initial_state), parameters


def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def evaluateExpression(exp, Varlist):
    EvaResult = pycarl.cln.FactorizedRationalFunction.evaluate(exp, Varlist)
    return convert(EvaResult)


def obtaining_data_from_csv(path, colunm_value, skip_header_value):
    my_data = genfromtxt(path, delimiter=',', skip_header=skip_header_value)
    return my_data[:, colunm_value]


def linear_fit(x, a, b):
    return a * x + b


def diff_value(num1, num2):
    if num1 > num2:
        diff = num1 - num2
    else:
        diff = num2 - num1
    return diff


# returning x value for when a linear fitting is needed
def identifying_updating_index(linear_slope, linear_intercept, x, y, residual_threshold, N, end_of_window):
    i = end_of_window + 1
    counter_positive = 0
    counter_negative = 0
    while i < y.size:
        linear_result = linear_slope * i + linear_intercept
        if y[i] > linear_result and diff_value(y[i], linear_result) >= residual_threshold:
            counter_positive = counter_positive + 1
            counter_negative = 0
        if y[i] < linear_result and diff_value(y[i], linear_result) >= residual_threshold:
            counter_negative = counter_negative + 1
            counter_positive = 0

        if counter_negative >= N:
            return i - counter_negative
        if counter_positive >= N:
            return i - counter_positive

        i = i + 1
    return end_of_window


def piecewise_linear_analysis(x, y, window_size, N, residual_threashold):
    fitting_index = 0
    fit_a, fit_b = optimize.curve_fit(linear_fit, x[fitting_index:fitting_index + window_size],
                                      y[fitting_index:fitting_index + window_size])[0]
    linear_function = np.array([fit_a, fit_b, fitting_index])
    while fitting_index <= y.size - window_size:
        fitting_index = identifying_updating_index(fit_a, fit_b, x, y, residual_threashold, N,
                                                   fitting_index + window_size)
        fit_a, fit_b = optimize.curve_fit(linear_fit, x[fitting_index:fitting_index + window_size],
                                          y[fitting_index:fitting_index + window_size])[0]
        new_function = np.array([fit_a, fit_b, fitting_index])
        linear_function = np.vstack((linear_function, new_function))
    return linear_function


def linear_prediction(model, horizon):
    predicted_slope = model[-1, 0]
    predicted_intercept = model[-1, 1]
    predicted_index = model[-1, 2]
    predicted_y = np.array(predicted_slope * predicted_index + predicted_intercept)
    for i in range(1, horizon):
        predicted_x = i + predicted_index
        new_predicted_y = np.array(predicted_slope * predicted_x + predicted_intercept)
        predicted_y = np.vstack((predicted_y, new_predicted_y))
    predicted_x = np.arange(predicted_index, predicted_index + horizon)[:, np.newaxis]
    return np.hstack((predicted_x, predicted_y))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    ModelCheckerResult = parametric_model_checking("/Users/xinweifang/Documents/prism-4.6-osx64/SEAM2022/Example.pm",
                                                   "P=?[F s=4]")
    # ModelCheckerResult[0] : Expression  ModelCheckerResult[1] : List for parameters
    algebraic_formulae = ModelCheckerResult[0]
    model_parameters = ModelCheckerResult[1]

    # Hyper-parameters
    window_size_setting = 20
    N_setting = 5
    residual_threshold_setting = 0.02
    prediction_horizon = 50
    predicted_variable_set = {}
    # loop for analysing and predicting each of parameter
    for i in range(1, len(model_parameters) + 1):
        # Path to csv files, column number for which y data is stored, the number of header skip
        y_reading = eval(f"obtaining_data_from_csv('/Users/xinweifang/Desktop/data{i}.csv', 2, 1)")
        x_generating = np.arange(0, y_reading.size, 1, dtype=int)
        fitted_linear_model = piecewise_linear_analysis(x_generating, y_reading, window_size_setting, N_setting,
                                                        residual_threshold_setting)
        predicted_variable_set[i] = linear_prediction(fitted_linear_model, prediction_horizon)
        # plt.figure()
        # plt.plot(x_generating, y_reading, "o")
        # for t in range(0, len(fitted_linear_model) - 1):
        #     start = fitted_linear_model[t, 2]
        #     end = fitted_linear_model[t + 1, 2]
        #     xd = np.linspace(start, end)
        #     plt.plot(xd, fitted_linear_model[t, 0] * xd + fitted_linear_model[t, 1])

    # plt.show()
    disruption_prediction = np.zeros(shape=(1, 2))
    for i in range(0, len(predicted_variable_set.get(1))):
        point = dict()
        index = 1
        for x in model_parameters:
            point[x] = stormpy.RationalRF(predicted_variable_set.get(index)[i, 1])
            index = index + 1

        disruption_prediction = np.vstack((disruption_prediction, np.array([predicted_variable_set.get(1)[i, 0], evaluateExpression(algebraic_formulae, point)])))
    disruption_prediction = np.delete(disruption_prediction, 0, 0)
    plt.figure()
    plt.plot(disruption_prediction[:, 0], disruption_prediction[:, 1])
    plt.show()
