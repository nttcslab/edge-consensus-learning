# -*- coding: utf-8 -*-
import copy
import grpc
import io
import torch
import threading
from .pb.mnw_pb2 import SwapParams
from .pb.mnw_pb2_grpc import MnwServiceStub


class Edge:
    def __init__(self, edge_info, self_index, self_name, device, state_dict, grpc_buf_size, err_max_cnt=10):
        self._node_addr = edge_info["addr"]
        self._node_idx = edge_info["index"]
        self._self_idx = self_index
        self._self_name = self_name
        self._device = device
        self._params_lock = threading.Lock()
        self._grpc_err_cnt = 0
        self._grpc_buf_size = grpc_buf_size
        self._err_max_cnt = err_max_cnt

        self.state = []
        self.dual = []
        self._state_r = []
        self._dual_r = []
        self._dual_avg = []
        for param_tensor in state_dict:
            self.state.append(copy.deepcopy(state_dict[param_tensor]))
            self.dual.append(torch.zeros(state_dict[param_tensor].size(), device=device))
            self._state_r.append(copy.deepcopy(state_dict[param_tensor]))
            self._dual_r.append(torch.zeros(state_dict[param_tensor].size(), device=device))
            # for ADMM_SGD
            self._dual_avg.append(copy.deepcopy(state_dict[param_tensor]))

        self._prm_a = 1
        if self_index > self._node_idx:
            self._prm_a = -1

    def swap(self):
        is_connected = True
        with grpc.insecure_channel(self._node_addr) as channel:
            try:
                stub = MnwServiceStub(channel)
                swap_req_iter = self.SwapReqIter(self._self_name, self.state, True, self._grpc_buf_size)
                responses = stub.Swap(swap_req_iter)
                params_buf = io.BytesIO()
                for res in responses:
                    params_buf.write(res.params)
                params_buf.seek(0)
                self._state_r = torch.load(params_buf)

                swap_req_iter = self.SwapReqIter(self._self_name, self.dual, False, self._grpc_buf_size)
                responses = stub.Swap(swap_req_iter)
                params_buf = io.BytesIO()
                for res in responses:
                    params_buf.write(res.params)
                params_buf.seek(0)
                self._dual_r = torch.load(params_buf)

            except grpc.RpcError:
                self._grpc_err_cnt += 1
                if self._err_max_cnt < self._grpc_err_cnt:
                    is_connected = False

        return is_connected

    class SwapReqIter(object):
        def __init__(self, self_name, params, is_state, grpc_buf_size):
            self._self_name = self_name
            self._is_state = is_state
            self._grpc_buf_size = grpc_buf_size
            self._params_buf = io.BytesIO()
            self._responses = []
            torch.save(params, self._params_buf)
            self._params_buf.seek(0)

        def __iter__(self):
            return self

        def __next__(self):
            read_buf = self._params_buf.read(self._grpc_buf_size)
            if not read_buf:
                raise StopIteration()
            return SwapParams(src=self._self_name, is_state=self._is_state, params=read_buf)

    def get_send_params(self, is_state):
        send_buf = io.BytesIO()
        if is_state:
            torch.save(self.state, send_buf)
        else:
            torch.save(self.dual, send_buf)
        return send_buf

    def set_recv_params(self, params, is_state):
        if is_state:
            self._state_r = torch.load(io.BytesIO(params.getvalue()))
        else:
            self._dual_r = torch.load(io.BytesIO(params.getvalue()))

    def rcv_state(self):
        return self._state_r

    def rcv_dual(self):
        return self._dual_r

    def dual_avg(self, index):
        self._dual_avg[index] = torch.div((self._dual_avg[index] + self._dual_r[index]), 2)
        return self._dual_avg[index]

    def prm_a(self):
        return self._prm_a

    def params_lock(self):
        self._params_lock.acquire()

    def params_release(self):
        self._params_lock.release()
