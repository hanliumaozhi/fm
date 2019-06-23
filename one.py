import torch
import torch.nn as nn


class FMLayer(nn.Module):
    def __init__(self, n=10, k=5):
        super(FMLayer, self).__init__()
        self.n = n
        self.k = k
        self.linear = nn.Linear(self.n, 1)
        self.V = nn.Parameter(torch.randn(self.n, self.k))

    def forward(self, x):
        linear_part = self.linear(x)
        interaction_part_1 = torch.mm(x, self.V)
        interaction_part_1 = torch.pow(interaction_part_1, 2)
        interaction_part_2 = torch.mm(torch.pow(x, 2), torch.pow(self.V, 2))
        return linear_part + torch.sum(0.5 * interaction_part_2 - interaction_part_1)


if __name__ == '__main__':
    fm = FMLayer(10, 5)
    x_val = torch.randn(1, 10)
    output = fm(x_val)
    print(output)
