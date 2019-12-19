# Author: Junkai-Yan
# Finished in 2019/12/19

import numpy as np
import random

def get_reward():
    """Get the reward matrix"""
    f = open('reward.txt')
    data = list()
    for line in f:
        line = line.strip().split()
        line = list(map(int, line))
        data.append(line)
    return np.array(data)

def q_learning(reward_table, gamma, end):
    """
    Q learning function
    :param reward_table: reward table, size is m*n, m for state, n for action
    :param gamma: coefficient < 1
    :param end: the target position
    :return: Q table
    """
    m, n = reward_table.shape
    q_table = np.zeros((m, n))
    epoch = 0
    while epoch < 500:
        start = random.randint(0, n-1)
        while True:
            if start == end:
                break
            actions = list()
            for action in range(n):
                if reward_table[start][action] != -1:
                    actions.append(action)
            action = actions[random.randint(0, len(actions)-1)]
            q_table[start][action] = reward_table[start][action] + gamma*max(q_table[action])
            # if start == end:
            #     break
            start = action
        epoch += 1
    return np.array(q_table, dtype='int32')

def find_path(q_table, start, end):
    """
    Find the path from start to end according to Q table
    :param q_table: Q table
    :param start: start state
    :param end: end state
    :return: None
    """
    path = [start]
    while start != end:
        start = np.argmax(q_table[start])
        path.append(start)
    print(path)

if __name__=="__main__":
    reward = get_reward()
    q_table = q_learning(reward, 0.8, 5)
    print(q_table)
    find_path(q_table, start=2, end=5)
