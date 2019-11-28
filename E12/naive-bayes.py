# Author: Junkai-Yan
# Finished in 2019/11/28
# This file predicts the classification of testing set by using naive bayes.

import numpy as np
import math
from math import e

# get data from file
def get_data(filename):
    """
    :param filename: name of file
    :return: data set (list)
    """
    f = open(filename, 'r')
    data_list = list()
    for line in f:
        data = line.strip().split()
        data = [int(num) for num in data]
        data_list.append(data)
    return data_list

# calculate mean and variance of data with label == 1 and label == 0 respectively
def get_mean_var(data_set, continue_list):
    """
    :param data_set: data set
    :param continue_list: index of continuous variable
    :return: mean and variance of data with label == 1 and label == 0 respectively
    """
    mean_list_true = list()
    variance_list_true = list()
    mean_list_false = list()
    variance_list_false = list()
    for i in range(len(data_set[0])):
        true_list = list()
        false_list = list()
        for j in range(len(data_set)):
            # note every value for data with label == 1 and label == 0 respectively
            if data_set[j][-1] == 1 and i in continue_list:
                true_list.append(data_set[j][i])
            if data_set[j][-1] == 0 and i in continue_list:
                false_list.append(data_set[j][i])
        # if is continuous variable, calculate mean and variance
        if i in continue_list:
            true_list = np.array(true_list)
            false_list = np.array(false_list)
            mean_list_true.append(true_list.mean())
            variance_list_true.append(true_list.var())
            mean_list_false.append(false_list.mean())
            variance_list_false.append(false_list.var())
        # else, padding 0
        else:
            mean_list_true.append(0)
            variance_list_true.append(0)
            mean_list_false.append(0)
            variance_list_false.append(0)
    return mean_list_true, variance_list_true, mean_list_false, variance_list_false

# get prior probability for data with label == 1 and label == 0 respectively
def get_probability(data_set, continue_list, attribute_dict):
    """
    :param data_set: data set
    :param continue_list: index of continuous variable
    :param attribute_dict: dict, key is every attribute's number, value is a list representing their domain
    :return: prior probability for data with label == 1 and label == 0 respectively
    """
    total_train = len(data_set)
    false_datadict = list()
    true_datadict = list()
    # initialize and data smoothing by adding one
    for attribute in attribute_dict:
        single_dict1 = {i:1 for i in attribute_dict[attribute]}
        single_dict2 = {i:1 for i in attribute_dict[attribute]}
        false_datadict.append(single_dict1)
        true_datadict.append(single_dict2)

    # counting the frequency for evey value of every attribute
    for data in data_set:
        for i in range(len(data)):
            if (i not in continue_list) and data[-1] == 0:
                false_datadict[i][data[i]] += 1
            if (i not in continue_list) and data[-1] == 1:
                true_datadict[i][data[i]] += 1

    # calculating the prior probability
    for data_dict in false_datadict:
        i = false_datadict.index(data_dict)
        new_len = len(attribute_dict[i])
        for key in data_dict:
            data_dict[key] /= (total_train+2*new_len)
    for data_dict in true_datadict:
        i = true_datadict.index(data_dict)
        new_len = len(attribute_dict[i])
        for key in data_dict:
            data_dict[key] /= (total_train+2*new_len)
    return true_datadict, false_datadict

# calculating the probability with answer == flag using Bayes formula
def calculate_probability(continue_list, probability_list, data, mean_list, variance_list, flag):
    probability = probability_list[-1][flag]
    for i in range(len(data)-1):
        # for discrete variable, find its probability from dict
        if i not in continue_list:
            probability *= probability_list[i][data[i]]
        # for continuous variable, calculate its probability using Gaussian distribution
        else:
            value = pow(e, -((data[i]-mean_list[i])**2)/(2*variance_list[i]))/(math.sqrt(2*math.pi*variance_list[i]))
            probability *= value
    return probability

if __name__=="__main__":
    # initialize
    continue_list = [0, 2, 4, 10, 11, 12]
    attribute_dict = {0:[0, 1], 1:[i for i in range(8)],
                      2:[0, 1], 3:[i for i in range(16)],
                      4:[0, 1], 5:[i for i in range(7)],
                      6:[i for i in range(14)], 7:[i for i in range(6)],
                      8:[i for i in range(5)], 9:[i for i in range(2)],
                      10:[0, 1], 11:[0, 1],
                      12:[0, 1], 13:[i for i in range(41)], 14:[0, 1]}
    training_set = get_data('num_pre_adult.data')
    testing_set = get_data('num_pre_adult.test')
    total = len(testing_set)

    mean_list_true, variance_list_true, mean_list_false, variance_list_false = get_mean_var(training_set, continue_list)
    true_probability_list, false_probability_list = get_probability(training_set, continue_list, attribute_dict)
    count = 0

    # testing data
    for data in testing_set:
        label = data[-1]
        true_pro = calculate_probability(continue_list, true_probability_list, data, mean_list_true, variance_list_true, 1)
        false_pro = calculate_probability(continue_list, false_probability_list, data, mean_list_false, variance_list_false, 0)
        # compare which is better
        if true_pro > false_pro:
            ans = 1
        else:
            ans = 0
        if ans == label:
            count += 1
    print('Acc: {:.6f}%'.format(float(count / total) * 100))
