def update_scores(self, dump_qvalues=True):
    """
    Update qvalues via iterating over experiences
    """
    history = list(reversed(self.moves))

    # Flag if the bird died in the top pipe
    high_death_flag = True if int(history[0][2].split("_")[1]) > 120 else False

    # Q-learning score updates
    t = 1
    # traversal for moves
    for time_step in history:
        state = time_step[0]
        action = time_step[1]
        next_state = time_step[2]

        reward = self.r[0]
        # 第一步和第二步直接导致失败，必被惩罚
        if t == 1 or t == 2:
            reward = self.r[1]
        # 如果因为撞到上面的管子，又跳了，则第三步也要被惩罚
        if t == 3 and high_death_flag and action == 1:
            reward = self.r[1]
        # Q learning 迭代
        max_value = max(self.qvalues[next_state][0], self.qvalues[next_state][1])
        self.qvalues[state][action] *= 1 - self.lr
        self.qvalues[state][action] += self.lr * (reward + self.discount * max_value)
        t += 1

    self.gameCNT += 1  # increase game count
    if dump_qvalues:
        self.dump_qvalues()  # Dump q values (if game count % DUMPING_N == 0)
    self.moves = []  # clear history after updating strategies
