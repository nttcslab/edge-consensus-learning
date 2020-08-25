# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from .contract import Contract


class PdmmSGDVanilla(Optimizer):
    def __init__(self, name, nodes, device, model, interval=10, offset=0, mu=200, eta=1.0, rho=0.1):
        lr = 1 / mu
        eta_rate = eta / mu
        defaults = dict(lr=lr, eta=eta, rho=rho, initial_lr=lr, eta_rate=eta_rate)
        super(PdmmSGDVanilla, self).__init__(model.parameters(), defaults)
        self._contract = Contract(name, nodes, device, model, interval, offset)
        self._diff = torch.tensor(0., device=device)
        self._criterion = nn.MSELoss()

        m_state = model.state_dict()
        dim_num_ary = []
        prev_dim_num = 1
        for name in m_state:
            dim_num = 1
            if name.endswith(".weight"):
                if m_state[name].ndim > 1:
                    for i, dim in enumerate(m_state[name].shape):
                        # except out_dim
                        if i > 0:
                            dim_num *= dim
                else:
                    # for GrpNorm
                    dim_num *= m_state[name].shape[0]
                prev_dim_num = dim_num
            else:
                # bias
                dim_num = prev_dim_num
            dim_num_ary.append(dim_num)

        for group in self.param_groups:
            group["dim_num"] = dim_num_ary

    def __setstate__(self, state):
        super(PdmmSGDVanilla, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        return loss

    @torch.no_grad()
    def update(self):
        edges = self._contract.edges()
        edge_num = float(len(edges))
        torch.nn.init.zeros_(self._diff)

        for edge in edges.values():
            edge.params_lock()

        for group in self.param_groups:
            mu = 1 / group["lr"]
            group["eta"] = mu * group["eta_rate"]

            for i, p in enumerate(group['params']):
                vs_metric = math.sqrt(group["dim_num"][i])
                consensus = torch.zeros_like(p)
                proximity = torch.zeros_like(p)

                p_data = p.data
                m_grad = p.grad.data
                v_grad = torch.zeros_like(p)
                vs_metric_eta = vs_metric * group["eta"]
                vs_metric_rho = vs_metric * group["rho"]
                torch.nn.init.constant_(v_grad, mu)
                coefficient = v_grad.clone()

                for edge in edges.values():
                    consensus += vs_metric_eta / edge_num * edge.prm_a() * edge.rcv_dual()[i]
                    proximity += vs_metric_rho / edge_num * edge.rcv_state()[i]
                    coefficient += vs_metric_eta / edge_num + vs_metric_rho / edge_num

                p.data = (v_grad * p_data - m_grad + consensus + proximity) / coefficient

                for edge in edges.values():
                    edge.state[i] = p.data.clone()
                    edge.dual[i] = edge.rcv_dual()[i] - 2 * edge.prm_a() * edge.state[i]
                    self._diff += self._criterion(p.data, edge.rcv_state()[i])

        for edge in edges.values():
            edge.params_release()

        self._contract.swap()
        return self._diff

