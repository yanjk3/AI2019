# Author: Junkai-Yan
# Finish in 2019/11/14
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
#
# # create nodes for BN
B = Node('B', ['B'])
E = Node('E', ['E'])
A = Node('A', ['A', 'B', 'E'])
J = Node('J', ['J', 'A'])
M = Node('M', ['M', 'A'])

# Generate cpt for each node
B.setCpt({'0':0.999, '1':0.001})
E.setCpt({'0':0.998, '1':0.002})
A.setCpt({'111':0.95, '011':0.05, '110':0.94, '010':0.06, '101':0.29, '001':0.71, '100':0.001, '000':0.999})
J.setCpt({'11':0.9, '01':0.1, '10':0.05, '00':0.95})
M.setCpt({'11':0.7, '01':0.3, '10':0.01, '00':0.99})

# print('P(A)∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗')
# VE.inference([B, E, A, J, M], ['A'], ['B', 'E', 'J', 'M'], {})
# print('P(J~M)∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗')
# VE.inference([B, E, A, J, M], ['J', 'M'], ['A', 'B', 'E'], {})
# print('P(A|J~M)∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗')
# VE.inference([B, E, A, J, M], ['A'], ['B', 'E'], {'J':'1', 'M':'0'})
# print('P(B|A)∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗')
# VE.inference([B, E, A, J, M], ['B'], ['E', 'J', 'M'], {'A':'1'})
print('P(B|J~M)∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗')
VE.inference([B, E, A, J, M], ['B'], ['E', 'A'], {'J':'1', 'M':'0'})
# print('P(J~M|~B)∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗∗')
# VE.inference([B, E, A, J, M], ['J', 'M'], ['E', 'A'], {'B':'0'})

# A = Node('A', ['A'])
# B = Node('B', ['A', 'B'])
# C = Node('C', ['A', 'C'])
# D = Node('D', ['B', 'C', 'D'])
# E = Node('E', ['C', 'E'])
#
# A.setCpt({'0':0.8, '1':0.2})
# B.setCpt({'00':0.8, '01':0.2, '10':0.2, '11':0.8})
# C.setCpt({'00':0.95, '01':0.05, '10':0.8, '11':0.2})
# D.setCpt({'000':0.95, '001':0.05, '010':0.2, '011':0.8, '100':0.2, '101':0.8, '110':0.2, '111':0.8})
# E.setCpt({'00':0.4, '01':0.6, '10':0.2, '11':0.8})
#
# VE.inference([A,B,C,D,E], ['A'], ['B', 'C'], {'D':'0', 'E':'1'})