import torch.nn as nn
import torch


class LearnableDropout(nn.Module):
    def __init__(self, initial_rate=0.5):
        super(LearnableDropout, self).__init__()
        # Initialize the dropout rate as a learnable parameter
        self.dropout_rate = nn.Parameter(torch.tensor(initial_rate))

    def forward(self, x):
        if self.training:
            # Constrain the dropout rate to [0, 1] using sigmoid
            effective_rate = torch.sigmoid(self.dropout_rate)

            # Generate dropout mask
            mask = (torch.rand(x.shape) < effective_rate).to(x.device)

            # Apply dropout
            return mask * x / effective_rate
        else:
            # No dropout during evaluation
            return x
