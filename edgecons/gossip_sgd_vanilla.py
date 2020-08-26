# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from .contract import Contract


class GossipSGDVanilla(SGD):
    def __init__(self, name, nodes, device, model, interval=10, offset=0,
                 lr=0.05, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(GossipSGDVanilla, self).__init__(model.parameters(), lr, momentum, dampening, weight_decay, nesterov)
        self._contract = Contract(name, nodes, device, model, interval, offset)
        self._diff = torch.tensor(0., device=device)
        self._criterion = nn.MSELoss()
        for group in self.param_groups:
            group["initial_lr"] = lr

    def __setstate__(self, state):
        super(GossipSGDVanilla, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def update(self):
        edges = self._contract.edges()
        torch.nn.init.zeros_(self._diff)
        for edge in edges.values():
            edge.params_lock()
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    d_p = p.data
                    p.data = torch.div((d_p + edge.rcv_state()[i]), 2)
                    edge.state[i] = p.data.clone()
                    self._diff += self._criterion(p.data, edge.rcv_state()[i])
            edge.params_release()

        self._contract.swap()
        return self._diff

