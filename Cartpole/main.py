#imports
import gym
import random
import numpy as np
#tensorflow imports
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizer_v2.adam import Adam
#Agent imports (RL)
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory



print (gym.__version__)
print (np.__version__)

#create ENV

env = gym.make('CartPole-v1')

#get states and actions number

states = env.observation_space.shape[0]
actions = env.action_space.n

episodes = 10

for episode in range (0,episodes):
    state = env.reset()
    done = False
    score = 0
    while not done:
        env.render()
        action = random.choice([1, 0])
        n_state, reward, done, info = env.step(action)
        score += reward
    print("Episode: {},Score: {} ".format(episode+1, score))


# building model

def build_model(states,actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(48,activation='relu'))
    model.add(Dense(48,activation='relu'))
    model.add(Dense(actions,activation='linear'))
    return model

model = build_model(states, actions)
model.summary()

#building agent

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, nb_actions=actions, nb_steps_warmup=1000,target_model_update=1e-2)
    return dqn

dqn = build_agent(model, actions)

dqn.compile(Adam(learning_rate=0.001,name='Adam'),metrics=['mae'])
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)


scores = dqn.test(env, nb_episodes=5, visualize=True)
print(np.mean(scores.history['episode_reward']))