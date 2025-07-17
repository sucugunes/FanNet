import torch
import torch.nn as nn


class FanConv(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length, dim=1):
        super(FanConv, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = seq_length

        self.layer = nn.Linear(in_channels * self.seq_length, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x, indices):
        #n_nodes, _ = indices.size()
        #import pdb; pdb.set_trace()
        n_nodes = indices.size(dim=1)
        if x.dim() == 2:
            x = torch.index_select(x, 0, indices.view(-1))
            x = x.view(n_nodes, -1)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, indices.view(-1))
            #######x_new = x.view(bs,n_nodes,self.seq_length,-1)
            #######x = torch.mean(x_new,dim=2)
            x = x.view(bs, n_nodes, -1)
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2 or 3, but received {}'.format(
                    x.dim()))
        #import pdb; pdb.set_trace()
        x = self.layer(x)
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)
