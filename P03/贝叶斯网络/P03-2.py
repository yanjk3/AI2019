# Author: Junkai-Yan
# Finish in 2019/11/15
class VE(object):
    @staticmethod
    def inference(factorList, queryVariables,
                 orderedListOfHiddenVariables, evidenceList):
        """
        :param factorList: Variable List
        :param queryVariables: Query List
        :param orderedListOfHiddenVariables: Random Variable
        :param evidenceList: Given Variable
        """
        for ev in evidenceList:
            for i in range(len(factorList)):
                if ev in factorList[i].var_list:
                    factorList[i] = factorList[i].restrict((ev, evidenceList[ev]))
        for var in orderedListOfHiddenVariables:
            relative_list = [factor for factor in factorList if var in factor.var_list]
            if len(relative_list) == 0:
                continue
            # Merge with multiply
            res_0 = relative_list[0]
            factorList.remove(res_0)
            for factor in relative_list[1:]:
                factorList.remove(factor)
                res_0 = res_0.multiply(factor)
            # Merge with sumout to eliminate var
            res_0 = res_0.sumout(var)
            factorList.append(res_0)
        print('RESULT:')
        res = factorList[0]
        for factor in factorList[1:]:
            res = res.multiply(factor)
        # Normalization
        total = sum(res.cpt.values())
        res.cpt = {k: v/total for k, v in res.cpt.items()}
        # output the result
        res.printfInf()

    @staticmethod
    def printfFactors(factorList):
        for factor in factorList:
            factor.printInf()

class Util(object):
    @staticmethod
    def to_binary(num, length ) :
        return format (num, '0'+str(length )+'b')

class Node(object):
    def __init__(self, name, var_list):
        self.name = name
        self.var_list = var_list
        self.cpt = dict()

    def setCpt(self, cpt):
        self.cpt = cpt

    def printfInf(self):
        print('Name = ' + self.name)
        print('  vars '+str(self.var_list))
        for key in self.cpt:
            print('    key: '+key+' val: '+str(self.cpt[key]))
        print()

    def multiply(self, factor):
        both = list()
        for a in self.var_list:
            if a in factor.var_list:
                both.append(a)
        # has no the same variable, multiply directly to calculate new cpt
        if len(both) == 0:
            new_cpt = dict()
            newList = self.var_list + factor.var_list
            for key_a in self.cpt:
                for key_b in factor.cpt:
                    new_cpt[key_a+key_b] = self.cpt[key_a]*factor.cpt[key_b]
        # else delete the same variable from self and calculate new cpt
        else:
            position_a = list()
            position_b = list()
            for both_v in both:
                position_a.append(self.var_list.index(both_v))
                position_b.append(factor.var_list.index(both_v))
            new_cpt = dict()
            newList = self.var_list + factor.var_list
            for both_v in both:
                newList.remove(both_v)
            for key_a in self.cpt:
                for key_b in factor.cpt:
                    flag = 1
                    for i in range(len(both)):
                        if key_a[position_a[i]] != key_b[position_b[i]]:
                            flag = 0
                            break
                    if flag == 0:
                        continue
                    else:
                        new_key = ''
                        for i in range(len(key_a)):
                            if i not in position_a:
                                new_key += key_a[i]
                        new_key += key_b
                        new_cpt[new_key] = self.cpt[key_a] * factor.cpt[key_b]
        new_node = Node('f'+str(newList), newList)
        new_node.setCpt(new_cpt)
        return new_node

    def sumout(self, var):
        newList = self.var_list.copy()
        position = self.var_list.index(var)
        newList.remove(var)
        new_cpt = dict()
        for key in self.cpt:
            new_key = key[0:position] + key[position + 1:]
            if new_key not in new_cpt:
                new_cpt[new_key] = self.cpt[key]
            else:
                new_cpt[new_key] += self.cpt[key]
        new_node = Node('f' + str(newList), newList)
        new_node.setCpt(new_cpt)
        return new_node

    def restrict(self, factor):
        variable, value = factor[0], factor[1]
        position = self.var_list.index(variable)
        newList = self.var_list.copy()
        newList.remove(variable)
        new_cpt = dict()
        for key in self.cpt:
            if key[position] == value:
                new_key = key[0:position] + key[position+1:]
                new_cpt[new_key] = self.cpt[key]
        new_node = Node('f' + str(newList), newList)
        new_node.setCpt(new_cpt)
        return new_node

# create nodes for BN
PatientAge = Node('PatientAge', ['PatientAge'])
CTScanResult = Node('CTScanResult', ['CTScanResult'])
MRIScanResult = Node('MRIScanResult', ['MRIScanResult'])
Anticoagulants = Node('Anticoagulants', ['Anticoagulants'])
StrokeType = Node('StrokeType', ['CTScanResult', 'MRIScanResult', 'StrokeType'])
Mortality = Node('Mortality', ['StrokeType', 'Anticoagulants', 'Mortality'])
Disability = Node('Disability', ['StrokeType', 'PatientAge', 'Disability'])
# Generate cpt for each node
PatientAge.setCpt({'0':0.1, '1':0.3, '2':0.6})
CTScanResult.setCpt({'0':0.7, '1':0.3})
MRIScanResult.setCpt({'0':0.7, '1':0.3})
Anticoagulants.setCpt({'0':0.5, '1':0.5})
StrokeType.setCpt({'000':0.8, '010':0.5, '100':0.5, '110':0.0,
                   '001':0.0, '011':0.4, '101':0.4, '111':0.9,
                   '002':0.2, '012':0.1, '102':0.1, '112':0.1})
Mortality.setCpt({'000':0.28, '100':0.99, '200':0.1, '010':0.56, '110':0.58, '210':0.05,
                  '001':0.72, '101':0.01, '201':0.9, '011':0.44, '111':0.42, '211':0.95})
Disability.setCpt({'000':0.8, '100':0.7, '200':0.90, '010':0.6, '110':0.5, '210':0.4, '020':0.3, '120':0.2, '220':0.1,
                   '001':0.1, '101':0.2, '201':0.05, '011':0.3, '111':0.4, '211':0.3, '021':0.4, '121':0.2, '221':0.1,
                   '002':0.1, '102':0.1, '202':0.05, '012':0.1, '112':0.1, '212':0.3, '022':0.3, '122':0.6, '222':0.8})
#
# print('*'*15+'TEST FOR E09'+'*'*15)
# print('P1')
# VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
#              ['Mortality'],
#              ['MRIScanResult', 'Anticoagulants', 'StrokeType', 'Disability'],
#              {'CTScanResult':'0', 'PatientAge':'1'})
# print('P2')
# VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
#              ['Disability'],
#              ['Mortality', 'Anticoagulants', 'StrokeType', 'CTScanResult'],
#              {'MRIScanResult':'1', 'PatientAge':'2'})
# print('P3')
# VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
#              ['StrokeType'],
#              ['Anticoagulants', 'Mortality', 'Disability'],
#              {'MRIScanResult':'0', 'CTScanResult':'1', 'PatientAge':'2'})
# print('P4')
# VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
#              ['Anticoagulants'],
#              ['MRIScanResult', 'CTScanResult', 'StrokeType', 'Disability', 'Mortality'],
#              {'PatientAge':'0'})

print('*'*15+'TEST FOR P03'+'*'*15)
print('P1')
VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
             ['CTScanResult', 'Mortality'],
             ['MRIScanResult', 'Anticoagulants', 'StrokeType', 'Disability'],
             {'PatientAge':'1'})
print('P2')
VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
             ['Disability', 'CTScanResult'],
             ['Anticoagulants', 'StrokeType', 'Mortality'],
             {'PatientAge':'2', 'MRIScanResult':'1'})
print('P3')
VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
             ['StrokeType'],
             ['Mortality', 'Anticoagulants', 'Disability'],
             {'PatientAge':'2', 'CTScanResult':'1', 'MRIScanResult':'0'})
print('P4')
VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
             ['Anticoagulants'],
             ['MRIScanResult', 'Mortality', 'CTScanResult', 'StrokeType', 'Disability'],
             {'PatientAge':'1'})
print('P5')
VE.inference([PatientAge, CTScanResult, MRIScanResult, Anticoagulants, StrokeType, Mortality, Disability],
             ['Disability'],
             ['MRIScanResult', 'Anticoagulants', 'StrokeType', 'CTScanResult', 'PatientAge', 'Mortality'],
             {})