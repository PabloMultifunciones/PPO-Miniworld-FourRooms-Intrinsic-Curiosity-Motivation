import torch.nn as nn
import torch
from auxiliars import Swish
from torch.distributions import Categorical

class ActorCriticNetwork(nn.Module):
    def __init__(self, in_chanels, n_actions, clip, c1, c2):
        super(ActorCriticNetwork, self).__init__()

        self.size_output_layer = 288
        self.clip = clip
        self.c1 = c1
        self.c2 = c2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chanels, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            Swish(),
            nn.Flatten(1, 3),
        )
        
        self.actor_output = nn.Sequential(
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, n_actions), 
            nn.Softmax(dim=-1)
        )
        
        self.critic_output = nn.Sequential(            
            nn.Linear(512, 256),
            Swish(),
            nn.Linear(256, 1), 
        )

        self.gru = nn.GRUCell(self.size_output_layer, 512)

    def forward(self, observation, hx):
        network_output = self.encoder(observation)

        if(network_output.size(dim=0) == 1):
            network_output = network_output.view(self.size_output_layer)

        new_hx = self.gru(network_output, (hx))

        value = self.critic_output(new_hx)
        distribution = self.actor_output(new_hx)

        return distribution, value, new_hx

    def calc_loss(self, observations_batch, actions_batch, old_log_probs_batch, value_targets, hxs_batch, advantages_batch):
        #advantages_batch = (advantages_batch - torch.mean(advantages_batch) ) / (torch.std(advantages_batch) + 1e-8)

        distributions, current_values, _ = self.forward(observations_batch, hxs_batch)

        m = Categorical(distributions)
        current_log_probs_batch = m.log_prob(actions_batch)     
        entropys = m.entropy()

        current_values = current_values.squeeze(1)
                
        ratios = torch.exp(current_log_probs_batch - old_log_probs_batch)

        surr1 = ratios * advantages_batch
        surr2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * advantages_batch

        actor_loss = -torch.min(surr1, surr2)

        critic_loss = torch.pow(current_values - value_targets, 2)
    
        ac_loss = actor_loss.mean() + self.c1 * critic_loss.mean() - self.c2 * entropys.mean()

        return ac_loss



