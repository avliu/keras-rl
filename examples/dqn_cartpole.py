import numpy as np
import gym
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


ENV_NAME = 'CartPole-v0'


def build_train_test(timesteps=50000, display=False):

    # Get the environment and extract the number of actions.
    env = gym.make(ENV_NAME)
    np.random.seed(123)
    env.seed(123)
    nb_actions = env.action_space.n

    # Next, we build a very simple model.
    model = Sequential()
    model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(nb_actions))
    model.add(Activation('linear'))
    print(model.summary())

    # Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
    # even the metrics!
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=10,
                   target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    # Okay, now it's time to learn something! We visualize the training here for show, but this
    # slows down training quite a lot. You can always safely abort the training prematurely using
    # Ctrl + C.
    dqn.fit(env, nb_steps=timesteps, visualize=display, verbose=2)

    # After training is done, we save the final weights.
    dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    # Finally, evaluate our algorithm for 5 episodes.
    result = dqn.test(env, nb_episodes=5, visualize=display)
    return result

if __name__ == "__main__":

    timesteps = [10**i for i in range(2, 8)] + [50**i for i in range(2, 8)]

    output = pd.Series(index=timesteps)
    for t in timesteps:
        result = build_train_test(t)
        rewards = pd.Series(result.history['episode_reward'])
        avg_reward = rewards.mean(0)
        output[t] = avg_reward
        print(f'average reward: {avg_reward}')

    output.to_csv(f'{ENV_NAME}_results.csv', index_label=['timesteps'], header=['average reward'])
