import torch

class Batch_DataSet(torch.utils.data.Dataset):

    def __init__(self, observations, next_observations, actions, advantages, old_log_probs, hxs, value_targets):
        super().__init__()
        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.advantages = advantages
        self.old_log_probs = old_log_probs
        self.hxs = hxs
        self.value_targets = value_targets

    def __len__(self):
        return self.observations.shape[0]
    
    def __getitem__(self, i):
        return self.observations[i], self.next_observations[i], self.actions[i], self.advantages[i], self.old_log_probs[i], self.hxs[i], self.value_targets[i]