import gym

env = gym.make('MountainCar-v0')
env.reset()
step = 0
scroe = 0



while(True):
    env.render()