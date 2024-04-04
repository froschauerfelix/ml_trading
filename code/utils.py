# Functions and classes that are executed repeatedly

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super().__init__()
        self.activation = nn.SiLU() # hyperparameter
        self.fcs = []

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        for i in range(self.num_layers):
            input_size = self.input_dim if i == 0 else self.hidden_dim
            fc = nn.Linear(input_size, self.hidden_dim)
            setattr(self, 'fc%i' % i, fc)
            self._set_init(fc)
            self.fcs.append(fc)

        self.predict = nn.Linear(self.hidden_dim, self.output_dim)
        self._set_init(self.predict)

    def _set_init(self, layer):
        nn.init.kaiming_normal_(layer.weight)

    def forward(self, x):
        pre_activation = [x]
        layer_input = [x]

        for i in range(self.num_layers):
            x = self.fcs[i](x)
            pre_activation.append(x)

            x = self.activation(x)
            layer_input.append(x)

        out = self.predict(x)
        out = torch.sigmoid(out)   # Binary Softmax Activation Function

        return out, layer_input, pre_activation
