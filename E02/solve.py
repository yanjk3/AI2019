import queue
puzzle = []
qu = queue.PriorityQueue()
min_step = 999999
path = []
step_set = [(-1, 0), (0, -1), (0, 1), (1, 0)]  # up, left, right, down
bound = 0
def cal_successor_h(position):
    x, y = position
    costs = []
    successors = []
    for step in step_set:
        x_1 = x + step[0]
        y_1 = y + step[1]
        successors.append((x_1, y_1))
        if not( 0 <= x_1 <= 3 and 0 <= y_1 <= 3):
            costs.append(999999)
            continue
        puzzle[x][y] = puzzle[x_1][y_1]
        puzzle[x_1][y_1] = '*'
        costs.append(h())
        puzzle[x_1][y_1] = puzzle[x][y]
        puzzle[x][y] = '*'
    return costs, successors

def h():
    cost = 0
    for i in range(4):
        for j in range(4):
            if i==3 and j==3 and puzzle[i][j]=='*':
                continue
            elif puzzle[i][j] != str(4*i + j + 1):
                cost += 1
    return cost

def search(g):
    global min_step
    global bound
    node = path[-1]
    f = h() + g
    if f > bound:
        return f
    if h() == 0:
        min_step = g
        return -1
    f_list = []
    successor_h, successors = cal_successor_h(node)
    for i in range(4):
        f_list.append(g + successor_h[i])
    # sort
    for i in range(4):
        for j in range(3 - i):
            if f_list[j] > f_list[j+1]:
                temp = f_list[j+1]
                f_list[j+1] = f_list[j]
                f_list[j] = temp
                temp1 = successors[j+1]
                successors[j+1] = successors[j]
                successors[j] = temp1
    for i in range(4):
        if f_list[i] >= 999999:
            break
        min_value = 999999
        next_position = successors[i]
        puzzle[node[0]][node[1]] = puzzle[next_position[0]][next_position[1]]
        puzzle[next_position[0]][next_position[1]] = '*'
        path.append(next_position)
        re = search(g + 1)
        if re == -1:
            return -1
        elif re < min_value:
            min_value = re
        path.pop()
        puzzle[next_position[0]][next_position[1]] = puzzle[node[0]][node[1]]
        puzzle[node[0]][node[1]] = '*'
    return min_value

def ida_star(position):
    global bound
    bound = h()
    path.append(position)
    while True:
        re = search(0)
        if re == -1:
            break
        bound = re

def cal_disorder():
    l = []
    num = 0
    for row in puzzle:
        for ele in row:
            if ele != '*':
                l.append(int(ele))
    for i in range(len(l)):
        for j in range(i+1, len(l)):
            if l[i] > l[j]:
                num += 1
    return num

if __name__ == "__main__":
    for i in range(4):
        row = input()
        row = row.split()
        puzzle.append(row)
    for i in range(4):
        for j in range(4):
            if puzzle[i][j] == '*':
                x, y = i, j
    num_disorder = cal_disorder()
    dif = abs(x - 3)
    has_solution = 1
    if num_disorder % 2 == 0:
        if dif % 2 != 0:
            print("Has no solution.")
            has_solution = 0
    if num_disorder % 2 != 0:
        if dif % 2 == 0:
            print("Has no solution")
            has_solution = 0

    if has_solution:
        ida_star((x, y))
        print("Best path length:", min_step)
        print("An optimal solution is:")
        for i in range(1, len(path)):

            if path[i][0] > path[i-1][0]:
                print("down ", end=" ")
            if path[i][0] < path[i - 1][0]:
                print("up   ", end=" ")
            if path[i][1] > path[i - 1][1]:
               print("right", end=" ")
            if path[i][1] < path[i - 1][1]:
                print("left ", end=" ")
            if i % 5 == 0:
                print()
