# -*- coding:utf-8 -*-
#
#   Author  :   寒江雪
#   E-mail  :
#   Date    :   19/10/10 00:03:44
#   Desc    :
#
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


class QValueEvaluation(object):
    def __init__(self, scope='estimator', log_dir=None, config=Config):
        self.scope = scope
        self.summary_writer = None
        self.config = config
        with tf.variable_scope(scope):
            self.build_graph()
            if log_dir:
                summary_dir = os.path.join(log_dir, scope)
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

        h_conv1 = tf.nn.relu(self.conv2d(self.s, W_conv1, 4) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2, 2) + b_conv2)
        #h_pool2 = max_pool_2x2(h_conv2)

        h_conv3 = tf.nn.relu(self.conv2d(h_conv2, W_conv3, 1) + b_conv3)
        #h_pool3 = max_pool_2x2(h_conv3)

        #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        # readout layer
        self.readout = tf.matmul(h_fc1, W_fc2) + b_fc2
        # define the cost function
        self.a = tf.placeholder("float", [None, self.config.ACTIONS])
        self.y = tf.placeholder("float", [None])
        readout_action = tf.reduce_sum(tf.multiply(self.readout, self.a), reduction_indices=1)
        self.cost = tf.reduce_mean(tf.square(self.y - readout_action))
        self.train_step = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        self.summaries = tf.summary.merge([
            tf.summary.scalar('loss', self.cost),
            tf.summary.scalar('max_q', tf.reduce_max(self.readout))
        ])

    def predict(self, sess, s_t):
        q_values = sess.run(self.readout, feed_dict={self.s: s_t})
        return q_values

    def train(self, sess, s_j_batch, a_batch, y_batch):
        _, summaries, global_step = sess.run(
            [self.train_step, self.summaries, tf.train.get_global_step()],
            feed_dict={
                self.y: y_batch,
                self.a: a_batch,
                self.s: s_j_batch}
        )
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
            self.summary_writer.flush()


def n_step_copy_model_parameters(q_value_evaluation, target_q_value_evaluation):
    def get_params(estimator):
        params = [t for t in tf.trainable_variables() if t.name.startswith(estimator.scope)]
        params = sorted(params, key=lambda t: t.name)
        return params
    params = get_params(q_value_evaluation)
    target_params = get_params(target_q_value_evaluation)

    assign_ops = []
    for t, target_t in zip(params, target_params):
        assign_op = tf.assign(ref=target_t, value=t)
        assign_ops.append(assign_op)
    return assign_ops


def preprocess_state(x_t):
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return x_t


def train_dqn(sess, q_value_evaluation, target_q_value_evaluation, model_dir=None,
              fix_target=True, pretrained_model_dir=None, assign_ops=None):
    # 开始游戏
    game_state = game.GameState()
    # 存储样本
    D = deque()

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
    while "flappy bird" != "angry bird":
        readout_t = q_value_evaluation.predict(sess, [s_t])[0]
        #  值域： [1,0]：什么都不做； [0,1]：提升Bird
        a_t = np.zeros([Config.ACTIONS])
        action_index = 0
        if t % Config.FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                print("----------Random Action----------")
                action_index = random.randrange(Config.ACTIONS)
                a_t[random.randrange(Config.ACTIONS)] = 1
            else:
                action_index = np.argmax(readout_t)
                a_t[action_index] = 1
        else:
            a_t[0] = 1  # do nothing

        # scale down epsilon
        if epsilon > Config.FINAL_EPSILON and t > Config.OBSERVE:
            epsilon -= (Config.INITIAL_EPSILON - Config.FINAL_EPSILON) / Config.EXPLORE

        # run the selected action and observe next state and reward
        # 表示界面图像数据，得分以及是否结束游戏
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = preprocess_state(x_t1_colored)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # store the transition in D
        D.append((s_t, a_t, r_t, s_t1, terminal))
        if len(D) > Config.REPLAY_MEMORY:
            D.popleft()

        # only train if done observing
        if t > Config.OBSERVE:
            # sample a minibatch to train on
            minibatch = random.sample(D, Config.BATCH)

            # get the batch variables
            s_j_batch = [d[0] for d in minibatch]
            a_batch = [d[1] for d in minibatch]
            r_batch = [d[2] for d in minibatch]
            s_j1_batch = [d[3] for d in minibatch]

            y_batch = []
            readout_j1_batch = q_value_evaluation.predict(sess, s_j1_batch)
            for i in range(0, len(minibatch)):
                terminal = minibatch[i][4]
                if terminal:
                    y_batch.append(r_batch[i])
                else:
                    y_batch.append(r_batch[i] + Config.GAMMA * np.max(readout_j1_batch[i]))
            q_value_evaluation.train(sess, s_j_batch, a_batch, y_batch)

        s_t = s_t1
        t += 1
        # train q_estimator
        if t > Config.OBSERVE:
            if t % Config.UPDATE_TARGET_ESTIMATOR_EVERY == 0 and fix_target:
                sess.run(assign_ops)

        # 10000 迭代保存一次模型
        if t % 10000 == 0:
            saver.save(sess, 'saved_networks/' + Config.GAME + '-dqn', global_step = t)

        if t <= Config.OBSERVE:
            state = "observe"
        elif t > Config.OBSERVE and t <= Config.OBSERVE + Config.EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX %e" % np.max(readout_t))
        # write info to files
        '''
        if t % 10000 <= 100:
            a_file.write(",".join([str(x) for x in readout_t]) + '\n')
            h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
            cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
        '''


def main():
    log_dir = "log_dir/"
    q_value_evaluation = QValueEvaluation(scope='estimator', log_dir=log_dir)
    target_q_value_evaluation = QValueEvaluation(scope='target_estimator', log_dir=log_dir)
    assign_ops = n_step_copy_model_parameters(q_value_evaluation, target_q_value_evaluation)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_dqn(sess, q_value_evaluation, target_q_value_evaluation, None,
                  True, None, assign_ops)


if __name__ == "__main__":
    main()


