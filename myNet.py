
import cartpole as cart
import math, random
import numpy as np
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

import matplotlib.pyplot as plt


q_action = np.load('./data/ep0_action.npy')
q_state = np.load('./data/ep0_state.npy')
for i in range(1, 10):
    q_action = np.concatenate((q_action, np.load('./data/ep'+ str(i) + '_action.npy')), axis=0)
    q_state = np.concatenate((q_state, np.load('./data/ep'+ str(i) + '_state.npy')), axis=0)


env = gym.make('CartPole-v1')

class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(DQN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128), #4, 128
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, env.action_space.n)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            state   = torch.FloatTensor(state).unsqueeze(0)
            q_value = self.forward(state)

            action  = q_value.argmax().item()

        else:
            action = random.randrange(env.action_space.n) #0 or 1
        return action


def update_target(current_model, target_model):
    target_model.load_state_dict(current_model.state_dict())


def train(current_model, optimizer, batch_size, gamma):
    #training dataset으로부터 sampling
    #index 뽑아오기
    index_sample = random.sample(list(range(0,len(q_state))),batch_size)
    state_sample = []
    action_sample = []
    rewards = []
    next_states = []
    for i in index_sample:
        state_sample.append(q_state[i])
        env.reset()
        env.state = q_state[i]
        action_sample.append(q_action[i])
        obs, reward, _, _ = env.step(int(q_action[i].item()))
        rewards.append(reward)

        next_states.append(cart.state_to_bucket(obs))

    state_sample = torch.FloatTensor(np.float32(state_sample))
    action_sample = torch.LongTensor(np.int32(action_sample))
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(np.float32(next_states))
    q_value = current_model(state_sample)
    next_q_values = current_model(next_states)

    ##### target model 유무
    #target_next_q_values = target_model(next_states)
    #####

    q_value = q_value.gather(1, action_sample).squeeze(1)
    #지금 모델 기준으로 다음 state에서 각 action별로의 q value를 가지고 argmax action을 결정한 다음,
    #타겟 모델에서 다음 state의 해당 action의 q value로 로스를 계산함
    #########
    #next_q_value = target_next_q_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.gather(1,torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
    ########

    expected_q_value = rewards + gamma * next_q_value

    loss = (q_value - expected_q_value).pow(2).mean()


    optimizer.zero_grad()
    loss.backward()

    """
    nn.utils.clip_grad_norm_(current_model.parameters(),1.0)
    """

    optimizer.step()

    return loss

def test_accuracy(current_model):
    q_action_test = np.load('./data/ep0_action_test.npy')
    q_state_test = np.load('./data/ep0_state_test.npy')
    for i in range(1, 2):
        q_action_test = np.concatenate((q_action_test, np.load('./data/ep' + str(i) + '_action_test.npy')), axis=0)
        q_state_test = np.concatenate((q_state_test, np.load('./data/ep' + str(i) + '_state_test.npy')), axis=0)

    missed = 0
    for i in range(0, len(q_action_test)):
        action = current_model(torch.FloatTensor(q_state_test[i])).argmax().item()

        if action != int(q_action_test[i].item()):
            missed = missed + 1

    return 100 - (missed/len(q_action_test))*100


gamma = 0.99


for bs in [32]:
    losses = []
    current_model = DQN(env.observation_space.shape[0], env.action_space.n)
    target_model = DQN(env.observation_space.shape[0], env.action_space.n)
    update_target(current_model, target_model)
    optimizer = optim.Adam(current_model.parameters())

    for i in range((int)(len(q_state)/bs)):
        losses.append(train(current_model, optimizer, bs,gamma))

        #if i % 10 == 0:
        #    update_target(current_model, target_model)
    plt.plot(losses)
    plt.show()
    print(bs," test_accuracy:",test_accuracy(current_model),"%")
    trained_model = current_model

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 10

epsilon_by_game = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * frame_idx / epsilon_decay)

games = 250
rec_game_rew = []
avg_rew = 0
num_success = 0
for i in range(games):
    obs = env.reset()
    current_state = cart.state_to_bucket(obs)

    game_rew = 0
    done = False
    while not done:
        action = trained_model.act(current_state, -1)
        obs,rew,done,info = env.step(action)
        game_rew += rew

        current_state = cart.state_to_bucket(obs)

        if done:
            rec_game_rew.append(game_rew)

avg_rew = sum(rec_game_rew) / games
num_success = sum(x == 500 for x in rec_game_rew)
print("avg_rew: ",avg_rew)
print("num_success: ", num_success)

#gradient clipping, DQN,discount, replay buffer, temporal difference, target model 고민해보기













