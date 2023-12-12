import torch
import gym
import gym_miniworld
import time
from agent import Agent
from skip_and_frames_env import SkipAndFramesEnv

def test(arguments):
    learning_rate = arguments.lr
    n_updates = arguments.epochs
    clip = arguments.clip
    c1 = arguments.c1
    c2 = arguments.c2
    minibatch_size = arguments.minibatch_size
    in_channels = arguments.in_channels
    n_outputs = arguments.n_outputs
    alpha = arguments.alpha
    beta = arguments.beta
    env = arguments.env
    
    agente = Agent(in_channels, n_outputs, learning_rate, n_updates, clip, minibatch_size, c1, c2, alpha, beta)
    
    raw_env = gym.make(env)
    env = SkipAndFramesEnv(raw_env, in_channels)

    while True:
        done = False
        observation = env.reset()
        hx = torch.zeros(512)

        while not done: 
            env.render()
            time.sleep(0.1)
            
            action, next_hx = agente.get_action_max_prob(observation, hx)
            #_, _, action, next_hx = agente.get_action(observation, hx)

            next_state, reward, done = env.step(action.item())

            observation = next_state
            hx = next_hx

            if done:
                print('Gano' if reward > 0 else 'Perdio')


