import os
import torch
import numpy as np
from torch.distributions import Categorical
from torch.utils.data import DataLoader
from actor_critic_network import ActorCriticNetwork
from curiosity_network import CuriosityNetwork
from batch_dataset import Batch_DataSet
from shared_adam import SharedAdam

class Agent():
    def __init__(self, in_channels, n_output, learning_rate, n_updates, clip, minibatch_size, c1, c2, alpha, beta, is_global = False):

        self.actor_critic = ActorCriticNetwork(in_channels, n_output, clip, c1, c2)
        self.curiosity = CuriosityNetwork(in_channels, n_output, alpha, beta)
        
        self.n_updates_per_iteration = n_updates
        self.minibatch_size = minibatch_size
        self.clip = clip
        self.c1 = c1
        self.c2 = c2

        self.model_name = 'models/modelo_actor_critico.pt'
        self.model_name_curiosity = "models/modelo_curiosity.pt"

        if(is_global == True):
            self.actor_critic.share_memory()
            self.curiosity.share_memory()

            self.optimizer = SharedAdam(list(self.curiosity.parameters()) + list(self.actor_critic.parameters()), lr=learning_rate)

        self.load_models()

    def save_models(self):
        print('Guardando modelos...')
        torch.save(self.actor_critic.state_dict(), self.model_name)
        torch.save(self.curiosity.state_dict(), self.model_name_curiosity)

    def load_models(self):
        if(os.path.isfile(self.model_name)):
            print('Se ha cargado un modelo para la red neuronal')
            self.actor_critic.load_state_dict(torch.load(self.model_name))
        else:
            print('No se ha encontrado ningun podelo para la red neuronal')

        if(os.path.isfile(self.model_name_curiosity)):
            print('Se ha cargado un modelo de curiosidad para la red neuronal')
            self.curiosity.load_state_dict(torch.load(self.model_name_curiosity))
        else:
            print('No se ha encontrado ningun modelo de curiosidad para la red neuronal')

    def get_action(self, observation, hx):
        distribution, value, next_hx = self.actor_critic(observation, hx)

        m = Categorical(distribution)
        action = m.sample()
        log_prob = m.log_prob(action)

        return log_prob, value.squeeze(0), action, next_hx

    def get_action_max_prob(self, observation, hx):
        distribution, _, next_hx = self.actor_critic(observation, hx)

        action = torch.argmax(distribution)

        return action, next_hx

    def update(self, observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets, global_agent):
        actor_critic_losses = []
        curiosity_losses = []

        dataset = Batch_DataSet(observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets)
        dataloader = DataLoader(dataset, batch_size=self.minibatch_size, num_workers=0, shuffle=True)

        for _ in range(self.n_updates_per_iteration):
                
            for i, batch in enumerate(dataloader):
                
                observations_batch, next_observations, actions_batch, advantages_batch, old_log_probs_batch, hxs_batch, value_targets_batch = batch 

                actor_critic_loss = self.actor_critic.calc_loss(observations_batch, actions_batch, old_log_probs_batch, value_targets_batch, hxs_batch, advantages_batch)
                curiosity_loss = self.curiosity.calc_loss(observations_batch, next_observations, actions_batch)  

                loss = actor_critic_loss + curiosity_loss

                global_agent.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(global_agent.actor_critic.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(global_agent.curiosity.parameters(), 0.5)

                for local_param, global_param in zip(self.actor_critic.parameters(), global_agent.actor_critic.parameters()):
                        global_param._grad = local_param.grad

                for local_param, global_param in zip(self.curiosity.parameters(), global_agent.curiosity.parameters()):
                        global_param._grad = local_param.grad

                global_agent.optimizer.step()

                self.actor_critic.load_state_dict(global_agent.actor_critic.state_dict())
                self.curiosity.load_state_dict(global_agent.curiosity.state_dict())

                actor_critic_losses.append(actor_critic_loss.item())
                curiosity_losses.append(curiosity_loss.item())

        return np.array(actor_critic_losses).mean(), np.array(curiosity_losses).mean()
