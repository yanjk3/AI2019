# Author: Junkai-Yan
# Finished in 2019/11/23
# This file builds decision tree using 'num_pre_adult.data'.
# Then, uses the tree to predict the salary of every person in 'num_pre_adult.test'.
# Finally, calculates accuracy.

from math import log

# generate data set
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

# calculate the entropy of label
def cal_entropy(dataset):
    """
    :param dataset: data set
    :return: the entropy of the data set
    """
    total = len(dataset)
    label_dict = {}
    for data in dataset:
        label = data[-1]
        if label not in label_dict:
            label_dict[label] = 1
        else:
            label_dict[label] += 1
    entropy = 0.0
    for key in label_dict:
        prob = float(label_dict[key]) / total
        entropy -= prob * log(prob,2)
    return entropy

# choose the data with variable at position whose value is value
def split_dataset(dataset, position, value):
    """
    :param dataset: data set
    :param position: which variable to choose
    :param value: which value to choose
    :return: data set with variable[position] == value
    """
    dataset_split = list()
    for data in dataset:
        if data[position] == value:
            dataset_split.append(data)
    return dataset_split

# choose the best variable to split
def choose_feature(dataset):
    """
    :param dataset: data set
    :return: which position of data set to be chosen to split the set
    """
    total_feature = len(dataset[0]) - 1
    total_entropy = cal_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    # choose the variable who has the best information gain
    for i in range(total_feature):
        feat_list = [data[i] for data in dataset]
        feat_list = set(feat_list)
        new_entropy = 0.0
        # calculate the entropy of the data set split by variable[i]
        for value in feat_list:
            sub_set = split_dataset(dataset, i, value)
            prob = len(sub_set) / float(len(dataset))
            new_entropy += prob * cal_entropy(sub_set)
        # calculate the information gain
        info_gain = total_entropy - new_entropy
        # update the information gain and best feature
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature

if __name__=="__main__":
    training_data = get_data('num_pre_adult.data')
    # print(choose_feature(training_data))
