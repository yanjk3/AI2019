# Author: Junkai-Yan
# Finished in 2019/12/11
# This file pre-processes data set by replacing the '?' by most frequent value or average value
# both for training set and testing set.

def preprocessing(filename):
    # all data
    dataset = list()
    # counter for data
    datadict = list()
    # list for calculating average
    continue_list = [3, 4, 5, 15, 18, 19, 21]
    average_list = list()

    # initialize
    for i in range(28):
        dataset.append([])
        datadict.append({})
        average_list.append([])

    # get data set
    f = open(filename, 'r')
    for line in f:
        s = line.strip().split(' ')
        for i in range(len(s)):
            if s[i] != '?':
                dataset[i].append(float(s[i]))
                if s[i] not in datadict[i]:
                    datadict[i][s[i]] = 1
                else:
                    datadict[i][s[i]] += 1
            else:
                dataset[i].append(s[i])
    f.close()

    # calculate average for continuous variable
    for c in continue_list:
        total = 0
        for data in dataset[c]:
            if data != '?':
                total += data
        average_list[c].append(total/len(dataset[c]))

    # replace '?' with most frequent value or average value
    for i in range(len(dataset)):
        if i not in continue_list:
            for data in dataset[i]:
                if data == '?':
                    position = dataset[i].index(data)
                    key = max(datadict[i], key=datadict[i].get)
                    dataset[i][position] = float(key)
        else:
            for data in dataset[i]:
                if data == '?':
                    position = dataset[i].index(data)
                    key = average_list[i][0]
                    dataset[i][position] = key

    # write new data set and put '23: outcome' to the last of the data(the last column is label)
    # also, change the interval of outcome from [1, 3] into [0, 2]
    f = open('pre_'+filename, 'w')
    for i in range(len(dataset[0])):
        s = ''
        for j in range(len(dataset)):
            if j != 22 :
                f.write(str(dataset[j][i]) + ' ')
            else:
                s = str(int(dataset[j][i]-1))
        f.write(s + '\n')
    f.close()

preprocessing('horse-colic.data')
preprocessing('horse-colic.test')