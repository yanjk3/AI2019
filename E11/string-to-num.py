# Author: Junkai-Yan
# Finish in 2019/11/21
# This file divide continuous value into two sections and generate number for each value
# Finally, change the string into num and get digital data set

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
                if int(s[i]) <= average_list[i]:
                    fw.write('0 ')
                else:
                    fw.write('1 ')
    f.close()
    fw.close()

continue_list = [0, 2, 4, 10, 11, 12]
# average_list from preprocessing
average_list = [38, 0, 189778, 0, 10, 0, 0, 0, 0, 0, 1077, 87, 40, 0, 0]
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
