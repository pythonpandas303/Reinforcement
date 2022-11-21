import time
import gym
env = gym.make('CartPole-v0')
env.reset()
for _ in range(1000):
#    time.sleep(0.1)
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
