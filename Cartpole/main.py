# Required libraries for the environment, agent, and model are imported.
import gym  # Gym library for reinforcement learning environments.
import random  # To generate random numbers for choosing actions.
import numpy as np  # For numerical operations.
# Tensorflow and Keras for building the neural network model.
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Flatten, Input
from keras.optimizer_v2.adam import Adam
# Keras-RL for reinforcement learning components.
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory

# Prints the versions of Gym and Numpy for debugging and compatibility checks.
print(gym.__version__)
print(np.__version__)

# Environment setup: Initializes the CartPole environment.
env = gym.make('CartPole-v1')

# Retrieves the shape of the environment's observation space (state space) and the number of possible actions.
states = env.observation_space.shape[0]
actions = env.action_space.n

# Number of episodes for the initial random play.
episodes = 10

# Loop through each episode.
for episode in range(0, episodes):
    state = env.reset()  # Resets the environment state at the start of each episode.
    done = False  # Initializes the 'done' flag to False.
    score = 0  # Initializes the score for the current episode.
    while not done:  # Continue until the episode is finished.
        env.render()  # Renders the environment's current state to the screen.
        action = random.choice([0, 1])  # Randomly chooses an action (0 or 1).
        n_state, reward, done, info = env.step(action)  # Takes the action and observes the new state and reward.
        score += reward  # Updates the score with the reward received from taking the action.
    # Prints the score at the end of the episode.
    print("Episode: {}, Score: {}".format(episode + 1, score))

# Defines a function to build the neural network model.
def build_model(states, actions):
    model = Sequential()  # Initializes a Sequential model.
    model.add(Flatten(input_shape=(1, states)))  # Flattens the input.
    model.add(Dense(48, activation='relu'))  # Adds a fully connected layer with 48 units and ReLU activation.
    model.add(Dense(48, activation='relu'))  # Adds another fully connected layer.
    model.add(Dense(actions, activation='linear'))  # Output layer with 'actions' units for Q-values of each action.
    return model

# Builds the model using the state and action dimensions.
model = build_model(states, actions)
model.summary()  # Prints the model summary to check the architecture.

# Defines a function to build the DQN agent.
def build_agent(model, actions):
    policy = BoltzmannQPolicy()  # Uses Boltzmann policy for action selection.
    memory = SequentialMemory(limit=50000, window_length=1)  # Sets up memory for experience replay.
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                   nb_actions=actions, nb_steps_warmup=1000, target_model_update=1e-2)
    return dqn

# Builds the DQN agent using the model and number of actions.
dqn = build_agent(model, actions)

# Compiles the DQN agent with Adam optimizer and mean absolute error as a metric.
dqn.compile(Adam(learning_rate=0.001, name='Adam'), metrics=['mae'])
# Trains the agent. This process does not render the environment's visuals to speed up training.
dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Tests the trained agent on the environment for 5 episodes with visuals.
scores = dqn.test(env, nb_episodes=5, visualize=True)
# Prints the average score of the testing phase.
print(np.mean(scores.history['episode_reward']))