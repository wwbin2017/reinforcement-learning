# -*- coding:utf-8 -*-
#
#   Author  :   寒江雪
#   E-mail  :
#   Date    :   19/11/10 11:03:30
#   Desc    : 使用PPO的方法玩flappy bird
#
"""
Dependencies:
tensorflow r1.14
pygame 1.9.4
"""
from __future__ import print_function

import tensorflow as tf
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import os

class Config(object):
    GAME = 'bird'
    ACTIONS = 2
    GAMMA = 0.99
    OBSERVE = 100000.0
    EXPLORE = 2000000.0
    FINAL_EPSILON = 0.0001
    INITIAL_EPSILON = 0.06
    REPLAY_MEMORY = 50000
    BATCH = 32
    FRAME_PER_ACTION = 1
    SAVE_MODEL_EVERY = 10000
    METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2)][1]
    BATCH = 500
    EP_LEN = 100000
    A_UPDATE_STEPS = 1
    C_UPDATE_STEPS = 1


S = tf.placeholder("float", [None, 80, 80, 4])

class Advantage(object):
    def __init__(self, scope='estimator', log_dir=None, config=Config):
        self.scope = scope
        self.summary_writer = None
        self.config = config
        with tf.variable_scope(scope):
            self.build_graph()

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(slef, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def build_graph(self):
        with tf.variable_scope("network"):
            # network weights
            W_conv1 = self.weight_variable([8, 8, 4, 32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4, 4, 32, 64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3, 3, 64, 64])
            b_conv3 = self.bias_variable([64])

            W_fc1 = self.weight_variable([1600, 512])
            b_fc1 = self.bias_variable([512])

            W_fc2 = self.weight_variable([512, 1])
            b_fc2 = self.bias_variable([1])

            h_conv1 = tf.nn.relu(self.conv2d(S, W_conv1, 4) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
            #h_pool2 = max_pool_2x2(h_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            #h_pool3 = max_pool_2x2(h_conv3)

            #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            # 输出动作的最大值
            self.readout = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.readout
            self.loss = tf.reduce_mean(tf.square(self.advantage))
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)

    def get_v(self, sess, s):
        return sess.run(self.readout, {S: s})[0, 0]


class PolicyNet(object):
    def __init__(self, scope='estimator', config=Config):
        self.scope = scope
        self.summary_writer = None
        self.config = config
        with tf.variable_scope(scope):
            self.build_graph()
            with tf.variable_scope('sample_action'):
                self.sample_op = tf.squeeze(self.norm_dist.sample(1), axis=0)       # choosing action

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def conv2d(slef, x, W, stride):
        return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

    def build_graph(self):
        scope = "network"
        with tf.variable_scope(scope):
            # network weights
            W_conv1 = self.weight_variable([8, 8, 4, 32])
            b_conv1 = self.bias_variable([32])

            W_conv2 = self.weight_variable([4, 4, 32, 64])
            b_conv2 = self.bias_variable([64])

            W_conv3 = self.weight_variable([3, 3, 64, 64])
            b_conv3 = self.bias_variable([64])

            W_fc1 = self.weight_variable([1600, 512])
            b_fc1 = self.bias_variable([512])

            W_fc2 = self.weight_variable([512, self.config.ACTIONS])
            b_fc2 = self.bias_variable([self.config.ACTIONS])

            W_fc3 = self.weight_variable([2*self.config.ACTIONS, 1])
            b_fc3 = self.bias_variable([1])

            h_conv1 = tf.nn.relu(self.conv2d(S, W_conv1, 4) + b_conv1)
            h_pool1 = self.max_pool_2x2(h_conv1)

            h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
            #h_pool2 = max_pool_2x2(h_conv2)

            h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
            #h_pool3 = max_pool_2x2(h_conv3)

            #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
            h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

            h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            mu = 2 * tf.layers.dense(h_fc1, self.config.ACTIONS, tf.nn.tanh)
            self.sigma = tf.layers.dense(h_fc1, self.config.ACTIONS, tf.nn.softplus)
            self.norm_dist = tf.distributions.Normal(loc=mu, scale=self.sigma)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(self.norm_dist.sample(1), axis=0)


    def predict(self, sess, s_t):
        a = sess.run(self.sample_op, {S: s_t})[0]
        return np.argmax(a)


def n_step_copy_model_parameters(q_value, target_q):
    def get_params(estimator):
        params = [t for t in tf.trainable_variables() if t.name.startswith(estimator.scope)]
        params = sorted(params, key=lambda t: t.name)
        return params
    params = get_params(q_value)
    target_params = get_params(target_q)

    assign_ops = []
    for t, target_t in zip(params, target_params):
        assign_op = tf.assign(ref=target_t, value=t)
        assign_ops.append(assign_op)
    """
    for t, target_t in zip(params, target_params):
        assign_op = tf.assign(target_t, (1-0.9)*t + 0.9*target_t)
        assign_ops.append(assign_op)
    """
    return assign_ops


def preprocess_state(x_t):
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t


def train_ppo(sess, advantage_eval_net, actor_eval_net, actor_target_net, actor_assign_ops):
    tfa = tf.placeholder(tf.float32, [None, Config.ACTIONS], 'action')
    tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
    with tf.variable_scope('loss'):
        with tf.variable_scope('surrogate'):
            # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
            ratio = actor_eval_net.norm_dist.prob(tfa) / actor_target_net.norm_dist.prob(tfa)
            surr = ratio * tfadv
        if Config.METHOD['name'] == 'kl_pen':
            tflam = tf.placeholder(tf.float32, None, 'lambda')
            kl = tf.distributions.kl_divergence(actor_target_net.norm_dist, actor_eval_net.norm_dist)
            kl_mean = tf.reduce_mean(kl)
            aloss = -(tf.reduce_mean(surr - tflam * kl))
        else:   # clipping method, find this is better
            aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-Config.METHOD['epsilon'], 1.+ Config.METHOD['epsilon'])*tfadv))

    with tf.variable_scope('atrain'):
        atrain_op = tf.train.AdamOptimizer(1e-6).minimize(aloss)

    # 开始游戏
    game_state = game.GameState()
    # 存储样本

    do_nothing = np.zeros(Config.ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    t = 0
    while "flappy bird" != "angry bird":
        readout_t = actor_eval_net.predict(sess, [s_t])
        #  值域： [1,0]：什么都不做； [0,1]：提升Bird
        a_t = np.zeros([Config.ACTIONS])
        a_t[readout_t] = 1  # do nothing

        # run the selected action and observe next state and reward
        # 表示界面图像数据，得分以及是否结束游戏
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        buffer_s.append(s_t)
        buffer_a.append(a_t)
        buffer_r.append(r_t)
        ep_r += r_t

        # update ppo
        if terminal or (t+1) % Config.BATCH == 0 or t == Config.EP_LEN-1:
            v_s_ = advantage_eval_net.get_v(sess, [s_t1])
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + Config.GAMMA * v_s_
                discounted_r.append([v_s_])
            discounted_r.reverse()
            # 更新分布
            sess.run(actor_assign_ops)
            adv = sess.run(advantage_eval_net.advantage, {S: buffer_s, advantage_eval_net.tfdc_r: discounted_r})


            # update actor
            if Config.METHOD['name'] == 'kl_pen':
                for _ in range(Config.A_UPDATE_STEPS):
                    _, kl = sess.run(
                        [atrain_op, kl_mean],
                        {S: buffer_s, tfa: buffer_a, tfadv: adv, tflam: Config.METHOD['lam']})
                    if kl > 4*Config.METHOD['kl_target']:  # this in in google's paper
                        break
                if kl < Config.METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                    Config.METHOD['lam'] /= 2
                elif kl > Config.METHOD['kl_target'] * 1.5:
                    Config.METHOD['lam'] *= 2
                Config.METHOD['lam'] = np.clip(Config.METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
            else:   # clipping method, find this is better (OpenAI's paper)
                [sess.run(atrain_op, {S: buffer_s, tfa: buffer_a, tfadv: adv}) for _ in range(Config.A_UPDATE_STEPS)]

            # update critic
            [sess.run(advantage_eval_net.train_op, {S: buffer_s, advantage_eval_net.tfdc_r: discounted_r}) for _ in range(Config.C_UPDATE_STEPS)]
            ######
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            t += 1

        s_t = s_t1


def main():
    advantage_eval_net = Advantage("advantage_eval_net", Config)

    actor_eval_net = PolicyNet("actor_eval_net", Config)
    actor_target_net = PolicyNet("actor_target_net",  Config)
    actor_assign_ops = n_step_copy_model_parameters(actor_eval_net, actor_target_net)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_ppo(sess, advantage_eval_net, actor_eval_net, actor_target_net, actor_assign_ops)


if __name__ == "__main__":
    main()
