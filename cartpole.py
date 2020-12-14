#!/usr/bin/env python3

import os, sys
import gym
import numpy as np
import math
import random

env = gym.make('CartPole-v1')


def simulate(episodes, test):
    # State space discretization
    NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # (left, right)

    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    STATE_BOUNDS[1] = [-0.5, 0.5]
    STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]

    ## load Q table
    q_table = np.load('q_table.npy')
    # print(q_table)

    # record data
    rec_state = np.zeros(shape=(500, 4))
    rec_action = np.zeros(shape=(500, 1))

    # set as converged state
    EPS = 0.01
    LR = 0.1
    # discount = 0.99

    rec_game_rew = []
    avg_rew = 0
    num_success = 0

    for i in range(episodes):

        obs = env.reset()
        current_state = state_to_bucket(obs)

        count = 0
        done = False
        game_rew = 0
        while not done:
            # record state
            rec_state[count] = current_state

            # choose action
            if np.random.random() < EPS:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[current_state])

            # record action
            rec_action[count] = action

            # feed action to environment
            obs, rew, done, info = env.step(action)
            count += 1
            game_rew += rew

            # discretize observation
            state = state_to_bucket(obs)

            # state transition
            current_state = state

            if done:
                #print("episode {} reached {} step.".format(i, count))
                rec_game_rew.append(game_rew)
                game_rew = 0
                """
                if test == False:
                    np.save('./data/ep{}_state'.format(i), rec_state)
                    np.save('./data/ep{}_action'.format(i), rec_action)
                else:
                    np.save('./data/ep{}_state_test'.format(i), rec_state)
                    np.save('./data/ep{}_action_test'.format(i), rec_action)
                """

    env.close()
    avg_rew = sum(rec_game_rew) / episodes
    num_success = sum(x == 500 for x in rec_game_rew)
    print("avg_rew: ", avg_rew)
    print("num_success: ", num_success)

def state_to_bucket(state):
    # State space discretization
    NUM_BUCKETS = (1, 1, 6, 3)  # (x, x', theta, theta')

    # Number of discrete actions
    NUM_ACTIONS = env.action_space.n  # (left, right)

    # Bounds for each discrete state
    STATE_BOUNDS = list(zip(env.observation_space.low, env.observation_space.high))
    STATE_BOUNDS[1] = [-0.5, 0.5]
    STATE_BOUNDS[3] = [-math.radians(50), math.radians(50)]
    bucket_indice = []

    for i in range(len(state)):
        if state[i] <= STATE_BOUNDS[i][0]:
            bucket_index = 0
        elif state[i] >= STATE_BOUNDS[i][1]:
            bucket_index = NUM_BUCKETS[i] - 1
        else:
            # Mapping the state bounds to the bucket array
            bound_width = STATE_BOUNDS[i][1] - STATE_BOUNDS[i][0]
            offset = (NUM_BUCKETS[i] - 1) * STATE_BOUNDS[i][0] / bound_width
            scaling = (NUM_BUCKETS[i] - 1) / bound_width
            bucket_index = int(round(scaling * state[i] - offset))
        bucket_indice.append(bucket_index)
    return tuple(bucket_indice)


if __name__ == '__main__':
    print("This package contains functions for simulating cart pole.")
    simulate(100000, True)



