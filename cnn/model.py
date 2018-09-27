import torch
import torch.nn as nn
import numpy as np

class Model(nn.Module):
    def __init__(self, n_mels, n_samp, n_lbls, verbose=False):
        super().__init__()

        n_filter = 256
        n_hidden = 256
        n_linear = 256

        probe = torch.zeros(1, n_mels, n_samp)
        parts = [nn.BatchNorm1d(n_mels)]
        for i in range(3):
            seq = nn.Sequential(nn.Conv1d(n_mels if i == 0 else n_filter, n_filter, 5), nn.ReLU(), nn.MaxPool1d(2))
            probe = seq(probe)
            parts.append(nn.Sequential(*seq, nn.BatchNorm1d(probe.shape[1]), nn.Dropout(0.1*(i+1))))
        self.conv = nn.Sequential(*parts)
        self.lstm = nn.LSTM(n_filter, n_hidden)
        self.fc   = nn.Sequential(nn.Linear(n_hidden, n_linear), nn.Dropout(0.5), nn.Linear(n_linear, n_lbls))
        if verbose:
            print(f"Created model for input {n_mels}x{n_samp} and {n_lbls} classes.\n"
                  f"parameter count: {self.param_count():,}")

    def param_count(self):
        return sum([sum([y.numel() for y in x.parameters()]) for x in self.modules() if type(x) not in {nn.Sequential, Model}])
    
    def use_lstm(self, x):
        return self.lstm(x.permute(2, 0, 1))[0] # => format: (nt, bs, nf)
    
    def time_dist_linear(self, x):
        nt, bs, nf = x.size()
        return self.fc(x.view(-1, nf)).view(nt, bs, -1).permute(1,2,0)
        # => format (bs, nf, nt)
    
    def average_linears(self, x):
        """ #IDEA: make weighted average
        nt = x.size(-1)
        weight = torch.arange(nt).view(1, 1, -1) + 1
        """

    def forward(self, x):
        x = self.conv(x)
        x = self.use_lstm(x)
        x = self.time_dist_linear(x)
        x = x.mean(-1)
        return x


def main():
    model = Model(128, 646, 10, verbose=True)
    print(f"parameter count: {sum([sum([y.numel() for y in x.parameters()]) for x in model.modules() if type(x) not in {nn.Sequential, Model}]):,}")
    inp = torch.ones(64, 128, 646)
    print(model(inp).size())

if __name__ == '__main__':
    main()