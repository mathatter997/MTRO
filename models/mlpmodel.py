import numpy as np
import torch


class MLPModel(torch.nn.Module):
    """
    construct the nn model for RankNet
    """

    def __init__(self, n_features, mlp_dims, lr,
    ):
        super(MLPModel, self).__init__()

        layers = []
        layers.append(torch.nn.Linear(n_features, mlp_dims[0]))
        layers.append(torch.nn.ReLU())

        for idx in range(len(mlp_dims) - 1):
            layers.append(torch.nn.Linear(mlp_dims[idx], mlp_dims[idx + 1]))
            layers.append(torch.nn.ReLU())

        layers.append(torch.nn.Linear(mlp_dims[-1], 1))

        self.lr = lr
        self.model = torch.nn.Sequential(*layers)
        self.output_sig = torch.nn.Sigmoid()
        self.total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self.optim = torch.optim.Adam(self.parameters(), lr=self.lr)

    def reset_parameters(self):
        for i in range(len(self.model)):
            if hasattr(self.model[i], "reset_parameters"):
                self.model[i].reset_parameters()

    def forward(self, input_):
        out = self.model(input_)
        return out

    def predict(self, input_):
        s = self.model(input_)
        n = s.data.cpu().numpy()
        return n