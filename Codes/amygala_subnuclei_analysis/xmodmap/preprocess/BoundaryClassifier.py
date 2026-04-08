import torch
from torch import nn
import pickle
from skorch import NeuralNetClassifier


class MyModule(nn.Module):
    def __init__(self, nonlin=nn.ELU(), interface_scale=1.):
        super().__init__()

        self.interface_scale = interface_scale

        self.dense0 = nn.Linear(3, 15)
        self.nonlin = nonlin
        # self.dropout = nn.Dropout(0.5)
        self.dense1 = nn.Linear(15, 5)
        self.output = nn.Linear(5, 1)
        self.softmax = nn.Tanh()

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X))
        # X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        y_pred = self.softmax(self.interface_scale * self.output(X))

        return torch.hstack([0.5 - y_pred / 2., 0.5 + y_pred / 2.])


class BoundaryClassifier:
    def __init__(self, param_file=None):
        self.mod = None
        if param_file is not None:
            self.mod = MyModule()
            self.mod.load_state_dict(torch.load(param_file))
            self.mod.eval()
            self.mod.requires_grad_(False)
            self.mod = self.mod.to('cuda:0')

    def train(self, trainingFile):
        info = torch.load(trainingFile)

        old_device = torch.tensor(0).device
        torch.set_default_device('cpu')

        nnet = NeuralNetClassifier(
            MyModule,
            max_epochs=500,
            lr=0.1,
            iterator_train__shuffle=True,
            train_split=None
        )

        nnet.fit(info['X'], info['y'], sample_weight=info['sample_weight'])

        torch.set_default_device(old_device)

        self.mod = nnet.module_
        self.mod.eval()
        self.mod.requires_grad_(False)
        self.mod = self.mod.to('cuda:0')
        
        return

    def predict(self, bandwidth, qx):
        self.mod.interface_scale = bandwidth
        p1 = self.mod.forward(qx)[:, 1]
        # convolve output with Gaussian
        
        return p1
        
    def save(self, filename):
        torch.save(self.mod.state_dict(), filename)
