import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def calculate_generalized_advantage_estimate(rewards, values, dones, gamma, tau):
    delta_t = rewards + gamma * values[1:] * dones - values[:-1]

    return calculate_discounted_rewards(delta_t, dones, gamma*tau)

def calculate_discounted_rewards(rewards, dones, gamma):
    value_targets = []
    old_value_target = 0
    
    for t in reversed(range(len(rewards)-1)):
        old_value_target = rewards[t] + gamma*old_value_target*dones[t]

        value_targets.append(old_value_target)
        
    value_targets.reverse()

    return torch.tensor(value_targets)

def draw_plot(title, xlabel, ylabel, title_file, history_score, color = "red"):

    new_array = []

    for i in range(len(history_score)):
        new_array.append(np.array(history_score[max(0, i-100):(i+1)]).mean())

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(new_array, color)
    plt.title(title)
    plt.savefig(title_file)
    plt.close()

def show_img(observation):
    window_title = "Juego"

    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)        
    cv2.imshow(window_title, observation)
    
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        cv2.destroyAllWindows()

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * nn.Sigmoid()(x)
