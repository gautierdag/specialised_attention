import torch
import torch.nn as nn
import torch.nn.functional as F


class PositioningGenerator(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()

        n_building_blocks_mu = 3
        self.mu_weights_generator = nn.Linear(input_size, n_building_blocks_mu)
        self.sigma_generator = nn.Linear(input_size, 1)

    def forward(self, x):
        sigma = self.sigma_generator(x)
        mu_weights = F.softmax(self.mu_weights_generator(x))
        return sigma, mu_weights
