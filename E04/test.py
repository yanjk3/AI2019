from numpy import *
from copy import deepcopy

def mrv(assigned, choice):
    num = 10
    pos = ()
    for i in range(9):
        for j in range(9):
            if len(choice[i][j]) < num and assigned[i][j] == 0:
                num = len(choice[i][j])
                pos = (i, j)
    return pos

def FC_check(board, C, choice, V):
    (x, y) = V
    if V == C[0]:
        x_1, y_1 = C[1]
        for c in choice[x_1][y_1]:
            if c <= int(board[x][y]):
                choice[x_1][y_1].remove(c)
    else:
        x_1, y_1 = C[0]
        for c in choice[x_1][y_1]:
            if c > int(board[x][y]):
                choice[x_1][y_1].remove(c)
    if len(choice[x_1][y_1]) == 0:
        return False
    return True

def forward_checking_main(board, assigned, choice, constraint, check):
    count = 0
    for row in assigned:
        for col in row:
            if col == 0:
                count += 1
    if count == 0:
        check[0] = 1
        for i in board:
            print(i)
        return
    # choose a V from not_assigned
    x, y = mrv(assigned, choice)
    assigned[x][y] = 1
    for d in choice[x][y]:
        if check[0] == 1:
            return
        flag = 0
        choice_t = deepcopy(choice)
        board[x][y] = str(d)
        # update row
        for col in choice_t[x]:
            if int(board[x][y]) in col:
                col.remove(int(board[x][y]))
        # update the col
        for k in range(9):
            if int(board[x][y]) in choice_t[k][y]:
                choice_t[k][y].remove(int(board[x][y]))
        choice_t[x][y] = [d]
        # update unequal_constraint
        for C in constraint:
            if (x, y) in C:
                if not FC_check(board, C, choice_t, (x, y)):
                    flag = 1
                    break
        if flag == 0:
            forward_checking_main(board, assigned, choice_t, constraint, check)
    assigned[x][y] = 0
    return

def generate_constraint_set():
    unequal_constraint = []
    inFile = open('constraint.txt', 'r')
    for line in inFile.readlines():
        s = line.strip()
        s = s.split()
        unequal_constraint.append([(int(s[0]), int(s[1])), (int(s[2]), int(s[3]))])
    inFile.close()
    return unequal_constraint

if __name__ == '__main__':
    board = []
    choice = []
    assigned = []
    inFile = open('board.txt', 'r')
    for line in inFile.readlines():
        s = line.strip()
        board.append(s.split())
    inFile.close()

    # generate constraint set
    constraint = generate_constraint_set()

    # generate assign_set and Domain
    for i in range(9):
        t = []
        for j in range(9):
            t.append(0)
        assigned.append(t)
    for i in range(9):
        t = []
        for j in range(9):
            c = [k for k in range(1, 10)]
            t.append(c)
        choice.append(t)

    for i in range(9):
        for j in range(9):
            if board[i][j] != '*':
                # update the row
                for col in choice[i]:
                    if int(board[i][j]) in col:
                        col.remove(int(board[i][j]))
                # update the col
                for k in range(9):
                    if int(board[i][j]) in choice[k][j]:
                        choice[k][j].remove(int(board[i][j]))
                choice[i][j] = [int(board[i][j])]
                assigned[i][j] = 1
    check = [0]
    forward_checking_main(board, assigned, choice, constraint, check)