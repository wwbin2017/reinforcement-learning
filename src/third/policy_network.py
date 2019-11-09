# -*- coding:utf-8 -*-
#
#   Author  :   寒江雪
#   E-mail  :
#   Date    :   19/11/09 00:03:44
#   Desc    : 使用策略梯度的方法玩flappy bird
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


class PolicyGradient(object):
    def __init__(self, scope='estimator', log_dir=None, config=Config):
        self.scope = scope
        self.summary_writer = None
        self.config = config
        # 一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs = [], [], []
        with tf.variable_scope(scope):
            self.build_graph()
            if log_dir:
                summary_dir = os.path.join(log_dir, scope)
                if os.path.exists(summary_dir):
                    print(summary_dir + " is exits.")
                else:
                    os.makedirs(summary_dir)
                self.summary_writer = tf.summary.FileWriter(summary_dir)

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

        self.s = tf.placeholder("float", [None, 80, 80, 4])
        self.tf_acts = tf.placeholder(tf.int32, [None, ], name="actions_num")
        self.tf_vt = tf.placeholder(tf.float32, [None, ], name="actions_value")



        h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
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

        #利用softmax函数得到每个动作的概率
        self.all_act_prob = tf.nn.softmax(self.readout, name='act_prob')
        #定义损失函数
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.all_act_prob,labels=self.tf_acts)
            self.loss = tf.reduce_mean(neg_log_prob*self.tf_vt)
        #定义训练,更新参数
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(1e-6).minimize(self.loss)
        #self.summaries = tf.summary.merge([
        #    tf.summary.scalar('loss_total', self.loss),
        #    tf.summary.scalar('max_action', tf.reduce_max(self.readout))
        #])

    def predict(self, sess, s_t):
        all_act_prob = sess.run(self.all_act_prob, feed_dict={self.s: s_t})
        #按照给定的概率采样
        action = np.random.choice(range(all_act_prob.shape[1]), p=all_act_prob.ravel())
        return action

    def greedy(self, observation):
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.tf_obs: observation[np.newaxis, :]})
        action = np.argmax(prob_weights.ravel())
        return action

    #定义存储，将一个回合的状态，动作和回报都保存在一起
    def store_transition(self, s, a, r):
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    #学习，以便更新策略网络参数，一个episode之后学一回
    def train(self, sess):
        #计算一个episode的折扣回报
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        #调用训练函数更新参数
        sess.run(self.train_op, feed_dict={
            self.s: self.ep_obs,
            self.tf_acts: self.ep_as,
            self.tf_vt: discounted_ep_rs_norm
        })
        #清空episode数据
        self.ep_obs, self.ep_as, self.ep_rs = [], [],[]
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        #折扣回报和
        discounted_ep_rs =np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.config.GAMMA + self.ep_rs[t]
            discounted_ep_rs[t] = running_add
        #归一化
        discounted_ep_rs-= np.mean(discounted_ep_rs)
        discounted_ep_rs /= np.std(discounted_ep_rs)
        return discounted_ep_rs


def preprocess_state(x_t):
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t


def train_policy_network(sess, policy_gradient_net, model_dir=None):
    # 开始游戏
    game_state = game.GameState()

    do_nothing = np.zeros(Config.ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = preprocess_state(x_t)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks")
    if False and checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    # start training
    epsilon = Config.INITIAL_EPSILON
    t = 0
    i_episode = 0
    while "flappy bird" != "angry bird":
        readout_t = policy_gradient_net.predict(sess, [s_t])
        #  值域： [1,0]：什么都不做； [0,1]：提升Bird
        a_t = np.zeros([Config.ACTIONS])
        a_t[readout_t] = 1
        print(readout_t, a_t)
        # run the selected action and observe next state and reward
        # 表示界面图像数据，得分以及是否结束游戏
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        #将观测，动作和回报存储起来
        policy_gradient_net.store_transition(s_t, readout_t, r_t)
        s_t = s_t1
        # 游戏完成一轮eposide，进行一次训练
        if terminal:
            ep_rs_sum = sum(policy_gradient_net.ep_rs)
            print("episode:", i_episode, "rewards:", ep_rs_sum)
            #每个episode学习一次
            vt = policy_gradient_net.train(sess)
            i_episode

def main():
    log_dir = "log_dir/"
    policy_gradient_net = PolicyGradient(scope='estimator', log_dir=log_dir)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_policy_network(sess, policy_gradient_net, None)


if __name__ == "__main__":
    main()
