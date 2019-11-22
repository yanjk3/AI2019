# Author: Junkai-Yan
# Finished in 2019/11/22
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

# choose the data with attribute at position whose value is value
def split_dataset(dataset, position, value):
    """
    :param dataset: data set
    :param position: which attribute to choose
    :param value: which value to choose
    :return: data set split by attribute[position] == value
    """
    dataset_split = list()
    for data in dataset:
        if data[position] == value:
            new_data = data[0:position] + data[position+1:]
            dataset_split.append(new_data)
    return dataset_split

# choose the best attribute to split by using ID3
def choose_feature_id3(dataset):
    """
    :param dataset: data set
    :return: which position of data set to be chosen to split the set
    """
    total_feature = len(dataset[0]) - 1
    total_entropy = cal_entropy(dataset)
    best_info_gain = 0.0
    best_feature = -1
    # choose the attribute who has the best information gain
    for i in range(total_feature):
        feat_list = [data[i] for data in dataset]
        feat_list = set(feat_list)
        new_entropy = 0.0
        # calculate the entropy of the data set split by attribute[i]
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

# choose the best attribute to split by using C4.5
def choose_feature_c45(dataset):
    """
    :param dataset: data set
    :return: which position of data set to be chosen to split the set
    """
    total_feature = len(dataset[0]) - 1
    total_entropy = cal_entropy(dataset)
    best_feature = -1
    best_info_gain_ratio = 0.0
    # choose the attribute who has the best information gain ratio
    for i in range(total_feature):
        feat_list = [data[i] for data in dataset]
        feat_list = set(feat_list)
        new_entropy = 0.0
        intrinsic_value = 0.0
        # calculate the entropy and intrinsic value of the data set split by attribute[i]
        for value in feat_list:
            sub_set = split_dataset(dataset, i, value)
            prob = len(sub_set) / float(len(dataset))
            new_entropy += prob * cal_entropy(sub_set)
            intrinsic_value -= prob * log(prob, 2)
        # calculate the information gain ratio
        info_gain = total_entropy - new_entropy
        if intrinsic_value == 0:
            continue
        info_gain_ratio = info_gain / intrinsic_value
        # update the information gain ratio and best feature
        if info_gain_ratio > best_info_gain_ratio:
            best_info_gain_ratio = info_gain
            best_feature = i
    return best_feature

# choose the most labels to be the leaf
def plurality_value(class_result):
    """
    :param class_result: result after classification
    :return: the most frequent result
    """
    num_counter = {}
    # count the times every result takes place
    for result in class_result:
        if result not in num_counter:
            num_counter[result] = 1
        else:
            num_counter[result] += 1
    num = 0
    plurality = -1
    # choose the most one
    for key in num_counter:
        if num_counter[key] > num:
            num = num_counter[key]
            plurality = key
    return plurality

# build tree
def create_tree(datas, attribute, attribute_dict, flag):
    """
    :param datas: data set
    :param attribute: attribute list
    :param attribute_dict: dict, key is every attribute, value is a list represents there domain
    :param flag: using ID3 or C4.5
    :return: decision tree
    """
    # the last column is the classification result
    class_result = [data[-1] for data in datas]
    # if all the data have the same classification, return it
    if class_result.count(class_result[0]) == len(class_result):
        return class_result[0]
    # if attribute is empty, then return most frequent classification result
    if len(attribute) == 0:      # only has classification result(attribute is empty)
        return plurality_value(class_result)
    # else, choose the best feature to split
    if flag == 'ID3':
        best_feature = choose_feature_id3(datas)
    else:
        best_feature = choose_feature_c45(datas)
    best_attribute = attribute[best_feature]
    value_list = attribute_dict[best_attribute]
    # create the tree, root is best feature (tree type: dict)
    decision_tree = {best_attribute: {}}
    attribute.pop(best_feature)
    feat_list = [data[best_feature] for data in datas]
    feat_list = set(feat_list)
    # for each value, create sub tree
    majority = plurality_value(class_result)
    for value in value_list:
        # if this value has some examples, create sub tree
        if value in feat_list:
            new_attribute = attribute.copy()
            decision_tree[best_attribute][value] = create_tree(split_dataset(datas, best_feature, value), new_attribute, attribute_dict, flag)
        # if this value has not example, choose the majority to be the leaf
        else:
            decision_tree[best_attribute][value] = majority
    return decision_tree

# test tree
def test_tree(decision_tree, attribute, test_data):
    """
    :param decision_tree: tree
    :param attribute: attribute list
    :param test_data: data to be classified
    :return: classification of test_data
    """
    # get the first attribute to find a branch
    key = list(decision_tree.keys())[0]  # 获取树的第一个特征属性
    # get sub tree according to key above
    sub_tree = decision_tree[key]
    # get the position of key
    position = attribute.index(key)
    # traverse all branch to find the right node to search further
    result = -1
    for value in sub_tree:
        if test_data[position] == value:
            # type is dict means this node isn't leaf, so search this node
            if type(sub_tree[value]).__name__ == 'dict':
                result = test_tree(sub_tree[value], attribute, test_data)
            # else, is a leaf, this node is the result
            else:
                result = sub_tree[value]
    return result

# calculate the accuracy for datas
def cal_accuracy(data_name, datas, tree, attribute_list):
    total = len(datas)
    counter = 0
    for data in datas:
        ans = test_tree(tree, attribute_list, data)
        if ans == data[-1]:
            counter += 1
    print('Acc at '+data_name+': {:.6f}%'.format(float(counter/total)*100))

if __name__=="__main__":
    attribute_list = ['age', 'workplace', 'fnlwgt', 'education',
                        'education-num', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'capital-gain',
                        'capital-loss', 'hours-per-week', 'native-country']
    attribute_dict = {'age':[0, 1], 'workplace':[i for i in range(8)],
                      'fnlwgt':[0, 1], 'education':[i for i in range(16)],
                      'education-num':[0, 1], 'marital-status':[i for i in range(7)],
                      'occupation':[i for i in range(14)], 'relationship':[i for i in range(6)],
                      'race':[i for i in range(5)], 'sex':[i for i in range(2)],
                      'capital-gain':[0, 1], 'capital-loss':[0, 1],
                      'hours-per-week':[0, 1], 'native-country':[i for i in range(41)]}
    training_data = get_data('num_pre_adult.data')
    testing_data = get_data('num_pre_adult.test')

    print('Test for decision tree built by ID3:')
    id3_tree = create_tree(training_data, attribute_list.copy(), attribute_dict, 'ID3')
    cal_accuracy('training set', training_data, id3_tree, attribute_list)
    cal_accuracy('testing set', testing_data, id3_tree, attribute_list)

    print('\nTest for decision tree built by C4.5:')
    c45_tree = create_tree(training_data, attribute_list.copy(), attribute_dict, 'C4.5')
    cal_accuracy('training set', training_data, c45_tree, attribute_list)
    cal_accuracy('testing set', testing_data, c45_tree, attribute_list)

