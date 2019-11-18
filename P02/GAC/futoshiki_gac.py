from numpy import *
from copy import deepcopy
import time

def mrv(assigned, choice, length):
    num = 10
    pos = ()
    for i in range(length):
        for j in range(length):
            if len(choice[i][j]) < num and assigned[i][j] == 0:
                num = len(choice[i][j])
                pos = (i, j)
    return pos

def GAC_check(gac_queue, constraint, choice):
    while len(gac_queue) != 0:
        C = gac_queue.pop(0)

        x0, y0 = C[0]
        x1, y1 = C[1]

        for d in choice[x0][y0]:
            flag = 0
            for a in choice[x1][y1]:
                if d < a:
                    flag = 1
                    break
            if flag == 0:
                choice[x0][y0].remove(d)
                if len(choice[x0][y0]) == 0:
                    return False
                else:
                    for c in constraint:
                        if (x1, y1) in c and c not in gac_queue:
                            gac_queue.append(c)

        for d in choice[x1][y1]:
            flag = 0
            for a in choice[x0][y0]:
                if a < d:
                    flag = 1
                    break
            if flag == 0:
                choice[x1][y1].remove(d)
                if len(choice[x1][y1]) == 0:
                    return False
                else:
                    for c in constraint:
                        if (x0, y0) in c and c not in gac_queue:
                            gac_queue.append(c)
    return True


def GAC_main(board, assigned, choice, constraint, length, check):
    count = 0
    for row in assigned:
        for col in row:
            if col == 0:
                count += 1
    if count == 0:
        for item in board:
            print(item)
        check[0] = 1
        return

    # choose a V from not_assigned
    x, y = mrv(assigned, choice, length)
    assigned[x][y] = 1

    for d in choice[x][y]:
        if check[0] == 1:
            return
        gac_queue = list()
        flag = 0
        choice_t = deepcopy(choice)
        board[x][y] = str(d)
        # update row
        for col in choice_t[x]:
            if int(board[x][y]) in col:
                col.remove(int(board[x][y]))
        # update the col
        for k in range(length):
            if int(board[x][y]) in choice_t[k][y]:
                choice_t[k][y].remove(int(board[x][y]))
        choice_t[x][y] = [d]
        for i in range(length):
            for j in range(length):
                if len(choice_t[i][j]) == 0:
                    continue

        # GAC check
        for C in constraint:
            if (x, y) in C:
                gac_queue.append(C)

        if not GAC_check(gac_queue, constraint, choice_t):
            flag = 1

        if flag == 0:
            GAC_main(board, assigned, choice_t, constraint, length, check)
    assigned[x][y] = 0
    return


def generate_constraint_set(case):
    unequal_constraint = []
    inFile = open('c' + str(case) + '.txt', 'r')
    for line in inFile.readlines():
        s = line.strip()
        s = s.split()
        unequal_constraint.append([(int(s[0]), int(s[1])), (int(s[2]), int(s[3]))])
    inFile.close()
    return unequal_constraint

if __name__ == '__main__':
    test = [1, 2, 3, 4, 5]
    for case in test:
        start = time.time()
        print('case:', case)
        board = []
        choice = []
        assigned = []
        inFile = open('b' + str(case) + '.txt', 'r')
        for line in inFile.readlines():
            s = line.strip()
            board.append(s.split())
        inFile.close()
        length = len(board)

        # generate constraint set
        constraint = generate_constraint_set(case)

        # generate assign_set and Domain
        for i in range(length):
            t = []
            for j in range(length):
                t.append(0)
            assigned.append(t)
        for i in range(length):
            t = []
            for j in range(length):
                c = [k for k in range(1, length+1)]
                t.append(c)
            choice.append(t)

        for i in range(length):
            for j in range(length):
                if board[i][j] != '*':
                    # update the row
                    for col in choice[i]:
                        if int(board[i][j]) in col:
                            col.remove(int(board[i][j]))
                    # update the col
                    for k in range(length):
                        if int(board[i][j]) in choice[k][j]:
                            choice[k][j].remove(int(board[i][j]))
                    choice[i][j] = [int(board[i][j])]
                    assigned[i][j] = 1
        check = [0]
        GAC_main(board, assigned, choice, constraint, length, check)
        end = time.time()
        print('case'+str(case)+' uses %.4fs\n'%(end-start))