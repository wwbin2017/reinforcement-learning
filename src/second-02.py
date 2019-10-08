#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Date    :   19/10/08 21:34:08
#   Desc    :
#
# coding: UTF-8

import random

class GridExample:
    def __init__(self):

        self.states = range(1, 17)  # 状态空间
        self.terminate_states = dict()  # 终止状态为字典格式
        self.terminate_states[10] = 1
        self.terminate_states[11] = 1
        self.terminate_states[12] = 1
        self.terminate_states[16] = 1

        self.actions = ["up", "down", "left", "right"]
        self.rewards = dict()        # 回报的数据结构为字典
        self.rewards['6_down'] = -1.0
        self.rewards['7_down'] = -1.0
        self.rewards['8_down'] = -1.0
        self.rewards['14_up'] = -1.0
        self.rewards['15_up'] = -1.0
        self.rewards['16_up'] = -1.0
        self.rewards['9_right'] = -1.0
        self.rewards['15_right'] = 1.0

        self.t = dict()     # 状态转移的数据格式为字典

        self.gamma = 0.8    # 折扣因子

        self.qfunc = dict()  # 动作值函数

    def transfrom(self, s, a):
        c_s = self.next_state(s, a)
        r = 0
        is_terminate = False
        if c_s in self.terminate_states:
            is_terminate = True
        key = "%d_%s" % (s, a)
        if key in self.rewards:
            r = self.rewards[key]
        if c_s == -1:
            is_terminate = True
            r = -10
        return is_terminate, c_s, r

    def next_state(self, current_state, action):
        j = (current_state - 1) / 4
        k = (current_state - 1) % 4
        if action == "up":
            if j == 0:
                return -1
            else:
                j = j - 1
        if action == "down":
            if j == 3:
                return -1
            else:
                j += 1
        if action == "left":
            if k == 0:
                return -1
            else:
                k -= 1
        if action == "right":
            if k == 3:
                return -1
            else:
                k += 1
        return 4 * j + k + 1

    def qlearning(self, inter_num, epsilon, gamma):
        qfunc = dict()  # 动作值函数
        # 初始化行为值函数为0
        for s in self.states:
            for a in self.actions:
                key = "%d_%s" % (s, a)
                qfunc[key] = 0.0
        count_s_a = dict()
        for i in range(inter_num):
            s_sample = []
            a_sample = []
            r_sample = []
            s = self.states[int(random.random()*len(self.states))]
            is_terminate = False
            count = 0
            while not is_terminate and count < 400:
                a = self.epsilon_greedy(qfunc, s, epsilon)
                is_terminate, c_s, r = self.transfrom(s, a)
                s_sample.append(s)
                a_sample.append(a)
                r_sample.append(r)
                s = c_s
                count += 1
            # 更新动作值函数
            g = 0.0
            for j in range(len(s_sample), -1, -1):
                g *= gamma
                g += r_sample[i]

            for j in range(len(s_sample)):
                key = "%d_%s" % (s_sample[j], a_sample[j])
                if key not in count_s_a:
                    count_s_a[key] = 0
                count_s_a[key] += 1
                qfunc[key] = (qfunc[key]*(count_s_a[key]-1) + g) /count_s_a[key]
                g -= r_sample[i]
                g /= gamma
        return qfunc


    #  贪婪策略
    def greedy(self, qfunc, state):
        amax = 0
        key = "%d_%s" % (state, self.actions[0])
        qmax = qfunc[key]
        for i in range(len(self.actions)):  # 扫描动作空间得到最大动作值函数
            key = "%d_%s" % (state, self.actions[i])
            q = qfunc[key]
            if qmax < q:
                qmax = q
                amax = i
        return self.actions[amax]

    def epsilon_greedy(self, qfunc, state, epsilon):
        amax = 0
        key = "%d_%s" % (state, self.actions[0])
        qmax = qfunc[key]
        for i in range(len(self.actions)):  # 扫描动作空间得到最大动作值函数
            key = "%d_%s" % (state, self.actions[i])
            q = qfunc[key]
            if qmax < q:
                qmax = q
                amax = i
        # 概率部分
        pro = [0.0 for i in range(len(self.actions))]
        pro[amax] += 1 - epsilon
        for i in range(len(self.actions)):
            pro[i] += epsilon / len(self.actions)

        ##选择动作
        r = random.random()
        s = 0.0
        for i in range(len(self.actions)):
            s += pro[i]
            if s >= r: return self.actions[i]
        return self.actions[len(self.actions) - 1]

    def qlearning_td(self, inter_num, epsilon, gamma, alpha):
        qfunc = dict()  # 动作值函数
        # 初始化行为值函数为0
        for s in self.states:
            for a in self.actions:
                key = "%d_%s" % (s, a)
                qfunc[key] = 0.0
        for i in range(inter_num):
            s = self.states[int(random.random() * len(self.states))]
            is_terminate = False
            count = 0
            while not is_terminate and count < 400:
                a = self.epsilon_greedy(qfunc, s, epsilon)
                is_terminate, c_s, r = self.transfrom(s, a)
                key = "%d_%s" % (s, a)
                if is_terminate:
                    qfunc[key] = qfunc[key] + alpha * (r - qfunc[key])
                else:
                    a_max = self.greedy(qfunc, c_s)
                    key1 = "%d_%s" % (c_s, a_max)
                    qfunc[key] = qfunc[key] + alpha*(r +
                             gamma*qfunc[key1] - qfunc[key])
                s = c_s
                count += 1
        return qfunc

    def route(self, s, inter_num):
        route = list()
        for i in range(inter_num):
            is_terminate = False
            count = 0
            while not is_terminate and count < 400:
                a = self.epsilon_greedy(qfunc, s, 0)
                is_terminate, c_s, r = self.transfrom(s, a)
                route.append([s, a])
                if is_terminate:
                    return route
                s = c_s
                count += 1


if __name__ == "__main__":
    g = GridExample()
    qfunc = g.qlearning_td(100000, 0.1, 0.8, 0.01)
    print g.route(4, 400)
    print qfunc

