from __future__ import division

import time

import stormpy.info

import pycarl
import csv
import stormpy.examples.files
import matplotlib.pyplot as plt
import stormpy._config as config
from numpy import genfromtxt
from scipy import optimize
import numpy as np
import random
from sklearn.metrics import confusion_matrix

import pandas as pd


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
    parameters = model.collect_all_parameters()
    return result.at(
        initial_state), parameters, model.collect_probability_parameters(), model.collect_reward_parameters()


def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)


def evaluateExpression(exp, Varlist):
    EvaResult = pycarl.cln.FactorizedRationalFunction.evaluate(exp, Varlist)
    return convert(EvaResult)


# def obtaining_data_from_csv(path, colunm_value, skip_header_value):
#     my_data = genfromtxt(path, delimiter=',', skip_header=skip_header_value)
#     return my_data[:, colunm_value]


def linear_fit(x, a, b):
    return a * x + b


def diff_value(num1, num2):
    if num1 > num2:
        diff = num1 - num2
    else:
        diff = num2 - num1
    return diff


def linear_analysis(x, y):
    popt, pcov = optimize.curve_fit(linear_fit, x, y)
    fit_a, fit_b = popt[0], popt[1]
    line_fitting_error = np.mean(np.diag(pcov))
    return np.array([fit_a, fit_b, x[0], line_fitting_error])


def normalise_in_range(x, a, b):
    return (b - a) * (x - np.min(x)) / (np.max(x) - np.min(x)) + a


def linear_data_generator(xrange, sign, high, low):
    rate = linear_slope_generator(sign)
    y = xrange * rate
    a = normalise_in_range(y, low, high)
    return a


def moving_avarage_function(arr, window_size):
    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array to consider
    # every window of size 3
    while i < len(arr) - window_size + 1:
        # Store elements from i to i+window_size
        # in list to get the current window
        window = arr[i: i + window_size]

        # Calculate the average of current window
        window_average = round(sum(window) / window_size, 2)

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1
    return moving_averages


def exp_data_generator(xrange, sign, high, low):
    tau = random.randint(round(len(xrange) * 0.5), round(len(xrange)) * 2)
    if sign == "decrease":
        if round(random.random()) == 0:
            return normalise_in_range(np.exp(-xrange / tau), low, high)
        else:
            return normalise_in_range(1 - np.exp(xrange / tau) / np.sum(np.exp(xrange / tau)), low, high)
    elif sign == "increase":
        if round(random.random()) == 0:
            return normalise_in_range(-np.exp(-xrange / tau), low, high)
        else:
            return normalise_in_range(np.exp(xrange / tau) / np.sum(np.exp(xrange / tau)), low, high)


def curve1(in_array_):
    return (in_array_ ** 3) + ((in_array_ * .9 - 4) ** 2)


def curve2(in_array_):
    return (20 * np.sin((in_array_) * 3 + 4) + 20) + curve1(in_array_)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def linear_slope_generator(trend):
    if trend == "increase":
        return np.random.exponential(scale=1.0, size=None)
    elif trend == "decrease":
        return -np.random.exponential(scale=1.0, size=None)


def new_data(type, noise, default_trend, data_size, max_value, min_value):
    trend = default_trend

    if type == "linear":
        y_reading_ref = np.array(linear_data_generator(np.linspace(1, data_size,
                                                                   data_size), trend, max_value, min_value))
    elif type == "non-linear":
        y_reading_ref = np.array(exp_data_generator(np.linspace(1, data_size,
                                                                data_size), trend, max_value, min_value))
    elif type == "non-mono":
        y_reading_ref = np.array(exp_data_generator(np.linspace(1, data_size,
                                                                data_size), trend, max_value, min_value))
        y_reading_ref = y_reading_ref + np.random.normal(0, diff_value(min_value, max_value), y_reading_ref.shape)
        x = np.arange(0, y_reading_ref.size, 1, dtype=int)
        pfit = np.poly1d(np.polyfit(x, y_reading_ref, 5))
        y_reading_ref = pfit(x)
        y_reading_ref = normalise_in_range(y_reading_ref, min_value, max_value)
    y_reading = y_reading_ref
    a = diff_value(min_value, max_value)
    if noise == 1:
        y_reading = y_reading + np.random.normal(0, a * 0.01, y_reading.shape)
    elif noise == 2:
        y_reading = y_reading + np.random.normal(0, a * 0.05, y_reading.shape)
    elif noise == 3:
        y_reading = y_reading + np.random.normal(0, 0.01, y_reading.shape)

    # y_reading = y_reading_ref[0:-prediction_horizon]
    x_generating = np.arange(0, y_reading.size, 1, dtype=int)

    # plt.figure()
    # plt.plot(x_generating, y_reading)
    # np.savetxt("/Users/xinweifang/Desktop/y1.csv", y_reading, delimiter=",", fmt='%0.6f')
    return x_generating, y_reading, y_reading_ref


def plotgeneratedData(x_generating, y_reading, y_reading_ref, fitted_linear_model):
    plt.figure()
    plt.plot(x_generating, y_reading, "o")
    if fitted_linear_model.ndim > 1:
        for t in range(0, len(fitted_linear_model) - 1):
            start = fitted_linear_model[t, 2]
            end = fitted_linear_model[t + 1, 2]
            xd = np.linspace(start, end)
            plt.plot(xd, fitted_linear_model[t, 0] * xd + fitted_linear_model[t, 1])
        final_xd = np.linspace(fitted_linear_model[t + 1, 2], len(y_reading_ref))
        plt.plot(final_xd, fitted_linear_model[t + 1, 0] * final_xd + fitted_linear_model[t + 1, 1])
    else:
        final_xd = np.linspace(fitted_linear_model[2], len(y_reading_ref))
        plt.plot(final_xd, fitted_linear_model[0] * final_xd + fitted_linear_model[1])


def PRESTO_evaluation(fitted_model, H, ref_data, idx):
    predicted_result = dict()
    reference_result = dict()
    point = dict()
    ref_point = dict()
    disruption_prediction = np.zeros(shape=(1, 2))
    disruption_reference = np.zeros(shape=(1, 2))

    for i in model_parameters:
        predicted_result[i] = 0
        reference_result[i] = 0
        if H > len(ref_data.get(i)[2]) - idx:
            H = len(ref_data.get(i)[2]) - idx - 1

    if H > 0:
        for h in range(0, H):
            for i in model_parameters:
                if fitted_linear_model[i].ndim < 2:
                    linear_cal = fitted_model.get(i)[0] * (h + idx) + fitted_model.get(i)[1]
                    point[i] = stormpy.RationalRF(linear_cal)
                    val = (h + idx - 1)
                    ref_point[i] = stormpy.RationalRF(ref_data.get(i)[2][val])
                    predicted_result[i] = np.vstack((predicted_result.get(i), linear_cal))
                else:
                    linear_cal = fitted_model.get(i)[-1][0] * (h + idx) + fitted_model.get(i)[-1][1]
                    point[i] = stormpy.RationalRF(linear_cal)
                    val = (h + idx - 1)
                    ref_point[i] = stormpy.RationalRF(ref_data.get(i)[2][val])
                    predicted_result[i] = np.vstack((predicted_result.get(i), linear_cal))
                reference_result[i] = np.vstack((reference_result.get(i), ref_data.get(i)[2][(h + idx - 1)]))
            disruption_prediction = np.vstack((disruption_prediction, np.array(
                [(h + idx), evaluateExpression(algebraic_formulae, point)])))
            disruption_reference = np.vstack((disruption_reference, np.array(
                [(h + idx), evaluateExpression(algebraic_formulae, ref_point)])))
        for i in model_parameters:
            predicted_result[i] = np.delete(predicted_result[i], 0, 0)
            reference_result[i] = np.delete(reference_result[i], 0, 0)
        disruption_prediction = np.delete(disruption_prediction, 0, 0)
        disruption_reference = np.delete(disruption_reference, 0, 0)

        # plt.figure()
        # plt.plot(disruption_prediction[:, 0], disruption_prediction[:, 1])
        # plt.plot(disruption_reference[:, 0], disruption_reference[:, 1])
        # plt.legend(["Predicted", "reference"])
        fitting_index[i] = -1

        np.savetxt("/Users/xinweifang/Desktop/disruptionR1.csv",
                   np.hstack((disruption_prediction, disruption_reference)),
                   delimiter=",", fmt='%f')

    return disruption_prediction, disruption_reference, idx, predicted_result, reference_result


def paper_evaluation_result(requirement, system_level_prediction, system_level_ref):
    ref = -1
    predict = -1
    ref_idx = -1
    predict_idx = -1

    if max(system_level_ref[:, 1]) > requirement > min(system_level_ref[:, 1]):
        ref = 1
        ref_idx = find_nearest(system_level_ref[:, 1], requirement)
        # plt.figure()
        # plt.plot(system_level_prediction[:, 0], system_level_prediction[:, 1])
        # plt.plot(system_level_ref[:, 0], system_level_ref[:, 1])
        # plt.legend(["Predicted", "reference"])
    else:
        ref = 0

    if max(system_level_prediction[:, 1]) > requirement > min(system_level_prediction[:, 1]):
        predict = 1
        predict_idx = find_nearest(system_level_prediction[:, 1], requirement)
    else:
        predict = 0

    return np.hstack((ref, predict)), np.hstack((ref_idx, predict_idx))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # # Fruit picking model
    # ModelCheckerResult = parametric_model_checking("/Users/xinweifang/Documents/PRESTO/SEAM_example/Example.pm",
    #                                                "R{\"totalTime\"}=?[F s=5]")
    # requirement = 5
    #  successfully complete the task : P=?[F s=4] r = 0.8
    #   R{\"totalTime\"}=?[F s=5] r = 5
    #   R{\"totalCost\"}=?[F s=5]

    # # # RAD model
    ModelCheckerResult = parametric_model_checking(
        "/Users/xinweifang/Documents/PRESTO/simplifiedRAD/RAD_model_modified.pm",
        "R{\"totalTime\"}=? [ F \"end\" ]")
    requirement = 13

    # P =? [F \"complete\" ] R = 0.5
    # P=? [ !\"correction\" U \"complete\" ] R=0.5
    # R{\"totalTime\"}=? [ F \"end\" ] R=13

    # ModelCheckerResult[0] : Expression  ModelCheckerResult[1] : List for parameters
    algebraic_formulae = ModelCheckerResult[0]
    model_parameters = ModelCheckerResult[1]
    prob_parameters = ModelCheckerResult[2]
    rwd_parameters = ModelCheckerResult[3]

    loopMatrix = np.hstack((-1, -1, -1, -1, -1))

    # prediction_horizon = 360
    # N_setting = 700  # Number of consecutive points above or below the residual_threshold_setting
    counter = 0
    # for window_size_setting in [10, 50, 250, 750, 1500]:
    #     for prediction_horizon in [np.round(window_size_setting*0.5).astype(int), window_size_setting, window_size_setting*2]:
    #         for N_setting in [np.round(window_size_setting*0.5).astype(int), window_size_setting, window_size_setting*2]:
    for window_size_setting in [500]:
        for prediction_horizon in [window_size_setting*2]:
            for N_setting in [np.round(window_size_setting*0.8).astype(int)]:

                total_evaluation = 0
                matrix = np.hstack((-1, -1))
                accuracy = np.hstack((-1, -1))
                system_level_prediction_idx = np.array(0)
                evaluation_ending_flag = 0
                counter += 1
                print(counter, window_size_setting, prediction_horizon, N_setting)
                while evaluation_ending_flag < 43200:
                    # Hyper-parameters

                    # window_size_setting = 50

                    residual_threshold_setting = 0.05

                    predicted_variable_set = {}
                    reference_variable_set = {}
                    size_of_linear_analysis_return = 4

                    # generating synthetic data for parameters
                    data = dict()
                    datasize = -1
                    data_size_temp = np.array(0)
                    for i in prob_parameters:
                        # fruit picking
                        # data[i] = new_data("non-mono", noise=2, default_trend="decrease",
                        #                    data_size=random.randint(7200, 10080),
                        #                    max_value=1,
                        #                    min_value=0.6)
                        data[i] = new_data("non-mono", noise=0, default_trend="decrease", data_size=random.randint(7200, 10080),
                                           max_value=0.6,
                                           min_value=0.3)
                        # plt.figure()
                        # plt.plot(data[i][0], data[i][1], "o")
                        # print(len(data.get(i)[0]))
                        data_size_temp = np.vstack((data_size_temp, len(data.get(i)[0])))
                    if len(rwd_parameters) > 0:
                        for i in rwd_parameters:
                            # fruit picking
                            data[i] = new_data("non-mono", noise=0, default_trend="increase",
                                               data_size=random.randint(7200, 10080),
                                               max_value=10,
                                               min_value=1)
                            # data[i] = new_data("non-linear", noise=0, default_trend="increase",
                            #                    data_size=random.randint(7200, 10080),
                            #                    max_value=5,
                            #                    min_value=1)
                            data_size_temp = np.vstack((data_size_temp, len(data.get(i)[0])))
                    data_size_temp = np.delete(data_size_temp, 0, 0)
                    datasize = np.min(data_size_temp)

                    # simulate the run-time updating of sensor reading
                    fitting_index = dict()
                    counter_positive = dict()
                    counter_negative = dict()
                    fitted_linear_model = dict()
                    evaluation_index = dict()
                    # first run
                    t_time = time.time()
                    for i in model_parameters:
                        counter_positive[i] = 0
                        counter_negative[i] = 0
                        fitting_index[i] = -1
                        evaluation_index[i] = -1
                        if datasize >= window_size_setting:
                            fitted_linear_model[i] = linear_analysis(data.get(i)[0][0:window_size_setting],
                                                                     data.get(i)[1][0: window_size_setting])
                    system_level_prediction, system_level_ref, origin = PRESTO_evaluation(fitted_linear_model,
                                                                                          prediction_horizon, data,
                                                                                          window_size_setting)[0:3]
                    total_evaluation = +1
                    temp1, temp2 = paper_evaluation_result(requirement, system_level_prediction, system_level_ref)

                    if temp1[0] == 0 and temp1[1] == 0:
                        flag = 0
                        while flag == 0:
                            # simulate new data coming in
                            last_value = 0
                            if origin + prediction_horizon >= datasize - 1:
                                last_value = datasize - 1
                                flag = 1
                            else:
                                last_value = origin + prediction_horizon

                            for data_sample in range(origin, last_value):
                                for i in model_parameters:
                                    x = data.get(i)[0][data_sample]
                                    y = data.get(i)[1][data_sample]

                                    if fitted_linear_model[i].ndim < 2:
                                        linear_result = fitted_linear_model[i][0] * x + fitted_linear_model[i][1]
                                    else:
                                        linear_result = fitted_linear_model[i][-1][0] * x + fitted_linear_model[i][-1][1]

                                    if y > linear_result:
                                        counter_positive[i] = counter_positive[i] + 1
                                        counter_negative[i] = 0
                                    if y < linear_result:
                                        counter_negative[i] = counter_negative[i] + 1
                                        counter_positive[i] = 0
                                    if y == linear_result:
                                        counter_negative[i] = 0
                                        counter_positive[i] = 0

                                    if counter_negative[i] > N_setting:
                                        fitting_index[i] = data_sample - counter_negative[i]
                                    if counter_positive[i] > N_setting:
                                        fitting_index[i] = data_sample - counter_positive[i]

                                    #  this threshold value is not used in this version
                                    # if diff_value(y, linear_result) >= residual_threshold_setting:
                                    #     fitting_index[i] = data_sample

                                    if datasize - window_size_setting >= fitting_index.get(i) >= 0:
                                        updated_linear_model = linear_analysis(
                                            data.get(i)[0][fitting_index.get(i):fitting_index.get(i) + window_size_setting],
                                            data.get(i)[1][fitting_index.get(i):fitting_index.get(i) + window_size_setting])
                                        fitted_linear_model[i] = np.vstack((fitted_linear_model[i], updated_linear_model))
                                        evaluation_index[i] = fitting_index.get(i) + window_size_setting
                                #  This is for prediction system-level proerpty
                                idx_previous = -1
                                for i in model_parameters:
                                    if evaluation_index.get(i) > 0:
                                        if idx_previous != evaluation_index.get(i):
                                            system_level_prediction, system_level_ref, origin = PRESTO_evaluation(
                                                fitted_linear_model,
                                                prediction_horizon, data,
                                                evaluation_index.get(i))[0:3]
                                            total_evaluation = +1
                                            temp1, temp2 = paper_evaluation_result(requirement, system_level_prediction,
                                                                                   system_level_ref)
                                            matrix = np.vstack((matrix, temp1))
                                            accuracy = np.vstack((accuracy, temp2))
                                            system_level_prediction_idx = np.vstack((system_level_prediction_idx, origin))
                                            if temp1[0] == 1 or temp1[1] == 1:
                                                flag = 1

                                            idx_previous = evaluation_index.get(i)
                                            fitting_index[i] = -1
                                            evaluation_index[i] = -1
                                            counter_positive[i] = 0
                                            counter_negative[i] = 0
                                # for i in model_parameters:
                                #     plotgeneratedData(data.get(i)[0], data.get(i)[1], data.get(i)[1], fitted_linear_model.get(i))
                            # predict system-level again using when reaching the end of the previous prediction horizon
                            # print(time.time() - t_time)
                            if flag == 0:
                                system_level_prediction, system_level_ref, origin = PRESTO_evaluation(fitted_linear_model,
                                                                                                      prediction_horizon,
                                                                                                      data,
                                                                                                      last_value)[0:3]
                                total_evaluation = +1
                                temp1, temp2 = paper_evaluation_result(requirement, system_level_prediction,
                                                                       system_level_ref)
                                matrix = np.vstack((matrix, temp1))
                                accuracy = np.vstack((accuracy, temp2))
                                system_level_prediction_idx = np.vstack((system_level_prediction_idx, origin))
                                if temp1[0] == 1 or temp1[1] == 1:
                                    flag = 1
                    else:
                        # print("hi")
                        matrix = np.vstack((matrix, temp1))
                        accuracy = np.vstack((accuracy, temp2))
                        system_level_prediction_idx = np.vstack((system_level_prediction_idx, origin))
                    evaluation_ending_flag = evaluation_ending_flag + system_level_prediction_idx[-1]
                    # print(evaluation_ending_flag)
                matrix = np.delete(matrix, 0, 0)
                accuracy = np.delete(accuracy, 0, 0)
                system_level_prediction_idx = np.delete(system_level_prediction_idx, 0, 0)
                # print(accuracy)
                # print(confusion_matrix(matrix[:, 0], matrix[:, 1]))
                # tn, fp, fn, tp

                if not np.any(matrix[:, 0]) and not np.any(matrix[:, 1]):
                    fp = 0
                    tp = 0
                else:
                    tn, fp, fn, tp = confusion_matrix(matrix[:, 0], matrix[:, 1]).ravel()
                print(confusion_matrix(matrix[:, 0], matrix[:, 1]).ravel())
                # print(confusion_matrix(matrix[:, 0], matrix[:, 1]))
                # print(accuracy[:, 0], accuracy[:, 1])
                # print(matrix)
                # np.savetxt("/Users/xinweifang/Documents/PRESTO/SEAM_example/Reachability/with_noise/matrix.csv", matrix, delimiter=",", fmt='%d')
                # np.savetxt("/Users/xinweifang/Documents/PRESTO/SEAM_example/Reachability/with_noise/accuracy.csv", accuracy, delimiter=",", fmt='%d')
                loopMatrix = np.vstack((loopMatrix, np.hstack((window_size_setting, prediction_horizon, N_setting, fp, tp))))
                # np.savetxt("/Users/xinweifang/Documents/PRESTO/plot/sensitivity/FruitPickingR1/R1_noise.csv", loopMatrix,
                #            delimiter=",", fmt='%d')
                print("done")
                # plt.show()
    loopMatrix = np.delete(loopMatrix, 0, 0)
    # np.savetxt("/Users/xinweifang/Documents/PRESTO/plot/sensitivity/FruitPickingR1/R1_noise.csv", loopMatrix,
    #            delimiter=",", fmt='%d')
    np.savetxt("/Users/xinweifang/Documents/PRESTO/plot/sensitivity/FruitPickingR1/R1_noise0_accuracy.csv", accuracy,
               delimiter=",", fmt='%d')
