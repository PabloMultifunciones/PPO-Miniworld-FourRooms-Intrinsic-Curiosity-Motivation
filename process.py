import gym
import gym_miniworld
import torch
import numpy as np
import os
from agent import Agent
from skip_and_frames_env import SkipAndFramesEnv
from worker import Worker
from datetime import datetime
from auxiliars import draw_plot, calculate_generalized_advantage_estimate, calculate_discounted_rewards

def process(thread, n_outputs, episodes, batch_size, lam, in_channels, learning_rate, gamma, n_updates, clip, minibatch_size, c1, c2, env_name, alpha, beta, global_agent):

    local_agent = Agent(in_channels, n_outputs, learning_rate, n_updates, clip, minibatch_size, c1, c2, alpha, beta)

    raw_env = gym.make(env_name)
    env = SkipAndFramesEnv(raw_env, in_channels)
    worker = Worker(env, local_agent, batch_size)

    history_loss_curiosity = []
    history_loss_actor_critic = []
    history_extrinsic_rewards = []
    history_intrinsic_rewards = []

    for ciclo in range(episodes):
        observations_batch, actions_batch, extrinsic_rewards_batch, dones_batch, values_batch, old_log_probs_batch, hxs_batch, intrinsic_rewards_batch, last_value = worker.run()

        observations = torch.stack(observations_batch[:-1])
        next_observations = torch.stack(observations_batch[1:])
        actions = torch.stack(actions_batch[:-1])
        old_log_probs = torch.stack(old_log_probs_batch[:-1])
        hxs = torch.stack(hxs_batch[:-1]).detach_()
        values = torch.tensor(values_batch)
        dones = torch.tensor(dones_batch)
        extrinsic_rewards = torch.tensor(extrinsic_rewards_batch)
        intrinsic_rewards = torch.tensor(intrinsic_rewards_batch)

        combine_rewards = extrinsic_rewards + intrinsic_rewards

        values = torch.cat((values, last_value), dim=0)
        advantages = calculate_generalized_advantage_estimate(combine_rewards, values, dones, gamma, lam)
        
        #advantages = (advantages - torch.mean(advantages) ) / (torch.std(advantages) + 1e-8)

        combine_rewards = torch.cat((combine_rewards, last_value), dim=0)
        value_targets = calculate_discounted_rewards(combine_rewards, dones, gamma)[:-1]

        actor_critic_loss, curiosity_losses = local_agent.update(observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, global_agent)

        if(thread == 0):
            history_intrinsic_rewards.append(np.array(intrinsic_rewards).mean())
            history_loss_curiosity.append(curiosity_losses)
            history_loss_actor_critic.append(actor_critic_loss)
            history_extrinsic_rewards.append(np.array(extrinsic_rewards).sum())
            print('Ciclo: ', ciclo, 'Promedio: ', np.mean(history_extrinsic_rewards[-100:]))

    if(thread == 0):
        local_agent.save_models()

        now = datetime.now()
        path = 'plots/' + now.strftime("%d%m%Y%H%M%S")

        os.mkdir(path) 

        draw_plot("Recompensa Intrinseca Historial", "Ciclos", "Recompensa", path + "/recompensa_intrinseca_historial.png", history_intrinsic_rewards)
        draw_plot("Recompensas Extrinseca Historial", "Ciclos", "Recompensa", path + "/recompensa_extrinseca_historial.png", history_extrinsic_rewards)
        draw_plot("Perdidas Curiosity", "Ciclos", "Perdida", path + "/perdida_curiosity_historial.png", history_loss_curiosity)
        draw_plot("Perdidas Actor Critic", "Ciclos", "Perdida", path + "/perdida_actor_critic_historial.png", history_loss_actor_critic)
