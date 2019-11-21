# Author: Junkai-Yan
# Finished in 2019/11/21
# This file pre-processes data set by replacing the '?' by most frequent value or average value
# both for training set and testing set.

def preprocessing(filename):
    # all data
    dataset = list()
    # counter for data
    datadict = list()
    # list for calculating average
    continue_list = [0, 2, 4, 10, 11, 12]
    average_list = list()

    # initialize
    for i in range(15):
        dataset.append([])
        datadict.append({})
        average_list.append([])

    # get data set
    f = open(filename, 'r')
    for line in f:
        s = line.strip().split(', ')
        for i in range(len(s)):
            if s[i].isdigit():
                dataset[i].append(int(s[i]))
            else:
                dataset[i].append(s[i])
            if s[i] != '?':
                if s[i] not in datadict[i]:
                    datadict[i][s[i]] = 1
                else:
                    datadict[i][s[i]] += 1
    f.close()

    # calculate average for continuous variable
    for c in continue_list:
        total = 0
        for data in dataset[c]:
            total += data
        average_list[c].append(int(total/len(dataset[c])))

    # print(average_list)

    # replace '?' with most frequent value or average value
    for i in range(len(dataset)):
        if i not in continue_list:
            for data in dataset[i]:
                if data == '?':
                    position = dataset[i].index(data)
                    key = max(datadict[i], key=datadict[i].get)
                    dataset[i][position] = key
        else:
            for data in dataset[i]:
                if data == '?':
                    position = dataset[i].index(data)
                    key = average_list[i][0]
                    dataset[i][position] = key

    # write new data set
    f = open('pre_'+filename, 'w')
    for i in range(len(dataset[0])):
        for j in range(len(dataset)):
            if j != len(dataset)-1:
                f.write(str(dataset[j][i]) + ', ')
            else:
                f.write(str(dataset[j][i]) + '\n')
    f.close()

preprocessing('adult.data')
preprocessing('adult.test')
