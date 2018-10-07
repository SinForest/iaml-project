import torch
import torch.nn as nn
import numpy as np
from scipy.misc import imshow

class Model(nn.Module):
    def __init__(self, n_mels, n_samp, n_lbls, verbose=False):
        super().__init__()

        n_filter = 256
        size_img =  16
        n_hidden = size_img ** 2
        n_linear = 256

        self.size_img = size_img
        probe = torch.zeros(1, n_mels, n_samp)
        parts = [nn.BatchNorm1d(n_mels)]
        for i in range(3):
            seq = nn.Sequential(nn.Conv1d(n_mels if i == 0 else n_filter, n_filter, 5), nn.ReLU(), nn.MaxPool1d(2))
            probe = seq(probe)
            parts.append(nn.Sequential(*seq, nn.BatchNorm1d(probe.shape[1]), nn.Dropout(0.1*(i+1))))
        
        self.conv   = nn.Sequential(*parts)
        self.lstm   = nn.LSTM(n_filter, n_hidden)
        #probe = self.use_lstm(probe).reshape(-1, 1, self.size_img, self.size_img)
        parts = [nn.BatchNorm2d(1)]
        for i in range(3):
            seq = nn.Sequential(nn.Conv2d(1 if i == 0 else 3, 3, 5, padding=2), nn.ReLU())
            #probe = seq(probe)
            parts.append(nn.Sequential(*seq, nn.BatchNorm2d(3), nn.Dropout(0.1)))
        
        self.imconv = nn.Sequential(*parts)
        self.fc     = nn.Sequential(nn.Linear(3*n_hidden, n_linear), nn.Dropout(0.5), nn.Linear(n_linear, n_lbls))
        if verbose:
            print(f"Created model for input {n_mels}x{n_samp} and {n_lbls} classes.\n"
                  f"parameter count: {self.param_count():,}")
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.LSTM):
                nn.init.xavier_normal_(m.weight_ih_l0, gain=nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(m.weight_hh_l0, gain=nn.init.calculate_gain('relu'))
        self.init_gaussian(2)
            

    def param_count(self):
        return sum([sum([y.numel() for y in x.parameters()]) for x in self.modules() if type(x) not in {nn.Sequential, Model}])
    
    def use_lstm(self, x):
        return self.lstm(x.permute(2, 0, 1))[0] # => format: (nt, bs, nf)
    
    def time_dist_conv(self, x):
        nt, bs, nf = x.size()
        x = self.imconv(x.reshape(-1, 1, self.size_img, self.size_img))
        return self.blur(x).view(nt*bs, -1), nt, bs
    
    def time_dist_linear(self, x, nt, bs):
        nt_bs, nf = x.size()
        return self.fc(x).view(nt, bs, -1).permute(1,2,0)
        # => format (bs, nf, nt)
    
    def average_linears(self, x):
        """ #IDEA: make weighted average
        nt = x.size(-1)
        weight = torch.arange(nt).view(1, 1, -1) + 1
        """

    def forward(self, x):
        #TODO: testen
        #TODO: blur und bild extract
        #TODO: probe
        x = self.conv(x)
        x = self.use_lstm(x)
        x, nt, bs = self.time_dist_conv(x)
        x = self.time_dist_linear(x, nt, bs)
        x = x.mean(-1)
        return x
    
    def init_gaussian(self, n):
        x, y = np.mgrid[-n:n+1, -n:n+1]
        g = np.exp(-(x**2/float(n)+y**2/float(n)))
        g =  torch.tensor(g / g.sum(), requires_grad=False)

        self.gaussian = torch.zeros(3,3,2*n+1,2*n+1, requires_grad=False);
        print(g.size(), self.gaussian.size())
        self.gaussian[0,0] = g
        self.gaussian[1,1] = g
        self.gaussian[2,2] = g
    
    def gen_images(self, x):
        self.eval()
        x = self.conv(x)
        x = self.use_lstm(x)
        x, nt, bs = self.time_dist_conv(x)
        x = torch.sigmoid(x)
        return x.view(nt, bs, 3, self.size_img, self.size_img).permute(1,0,2,3,4)

    
    def blur(self, x):
        return torch.nn.functional.conv2d(x, self.gaussian, padding=2)


def main():
    model = Model(128, 646, 10, verbose=True)
    print(f"parameter count: {sum([sum([y.numel() for y in x.parameters()]) for x in model.modules() if type(x) not in {nn.Sequential, Model}]):,}")
    inp = torch.ones(64, 128, 646)
    print(model(inp).size())
    ims = model.gen_images(inp)
    import matplotlib.pyplot as plt
    for i in range(ims.size(0)):
        for ii in range(ims.size(1)):
            print(ims[i,ii].permute(1,2,0))
            plt.imshow(ims[i,ii].permute(1,2,0).detach().numpy())
            plt.show()

if __name__ == '__main__':
    main()