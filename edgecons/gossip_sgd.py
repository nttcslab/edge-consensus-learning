# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim.sgd import SGD
from .contract import Contract


class GossipSGD(SGD):
    def __init__(self, name, nodes, device, model, interval=10, offset=0,
                 lr=0.05, momentum=0, dampening=0, weight_decay=0, nesterov=False):
        super(GossipSGD, self).__init__(model.parameters(), lr, momentum, dampening, weight_decay, nesterov)
        self._contract = Contract(name, nodes, device, model, interval, offset, is_dual=False)
        self._diff = torch.tensor(0., device=device)
        self._criterion = nn.MSELoss()
        for group in self.param_groups:
            group["initial_lr"] = lr

    def __setstate__(self, state):
        super(GossipSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def update(self):
        edges = self._contract.edges()
        for edge in edges.values():
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    d_p = p.data
                    p.data = torch.div((d_p + edge.rcv_state()[i]), 2)
                    edge.update(p.data, i)

        self._contract.swap()

    @torch.no_grad()
    def diff(self):
        edges = self._contract.edges()
        torch.nn.init.zeros_(self._diff)

        for edge in edges.values():
            diff_buf = edge.diff_buff()
            buf_name_list = list(diff_buf)
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    self._diff += self._criterion(p.data, diff_buf[buf_name_list[i]])

        return self._diff
