# Author: Junkai-Yan
# Finished in 2019/11/28
# This file generates number for each value of every attribute.
# Then, it changes the string into num and get digital data set.

def change_to_num(filename, datadict):
    """
    :param filename: name of file
    :param datadict: string to num dictionary
    :return: none
    """
    f = open(filename, 'r')
    fw = open('num_'+filename, 'w')
    for line in f:
        s = line.strip().split(', ')
        for i in range(len(s)):
            if i not in continue_list:
                if i != len(s)-1:
                    fw.write(datadict[i][s[i]]+' ')
                else:
                    fw.write(datadict[i][s[i]]+'\n')
            else:
                fw.write(s[i]+' ')
    f.close()
    fw.close()

continue_list = [0, 2, 4, 10, 11, 12]

datadict = list()
for i in range(15):
    datadict.append({})

# build dictionary(key: str, value: num)
f = open('pre_adult.data', 'r')
for line in f:
    s = line.strip().split(', ')
    for i in range(len(s)):
        if i not in continue_list:
            if s[i] not in datadict[i]:
                datadict[i][s[i]] = str(len(datadict[i]))
f.close()

# change string into num
change_to_num('pre_adult.data', datadict)
change_to_num('pre_adult.test', datadict)
