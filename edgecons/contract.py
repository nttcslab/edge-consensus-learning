# -*- coding: utf-8 -*-
import grpc
import io
import time
import torch
import logging
from collections import OrderedDict
from concurrent import futures
from .edge import Edge
from .pb.mnw_pb2 import Status as PbStatus
from .pb.mnw_pb2 import Params as PbParams
from .pb.mnw_pb2 import StateParams, SwapParams
from .pb.mnw_pb2_grpc import MnwServiceStub
from .pb.mnw_pb2_grpc import MnwServiceServicer, add_MnwServiceServicer_to_server


class MnwGateway(MnwServiceServicer):
    def __init__(self, name, self_index, edges, edge_info, device, model,
                 is_state=True, is_dual=True, is_avg=False, grpc_buf_size=524288, grpc_timeout=1.0):
        self._name = name
        self._index = self_index
        self._edges = edges
        self._edge_info = edge_info
        self._device = device
        self._model = model
        self._is_state = is_state
        self._is_dual = is_dual
        self._is_avg = is_avg
        self._grpc_buf_size = grpc_buf_size
        self._grpc_timeout = grpc_timeout

    def Hello(self, request, context):
        if request.src not in self._edges:
            self._edges[request.src] = Edge(self._edge_info[request.src], self._index, self._name,
                                            self._device, self._model.state_dict(),
                                            self._is_state, self._is_dual, self._is_avg,
                                            self._grpc_buf_size, self._grpc_timeout)
        return PbStatus(status=200)

    def GetState(self, request, context):
        r_buf = io.BytesIO()
        torch.save(self._model.state_dict(), r_buf)
        r_buf.seek(0)
        while True:
            read_buf = r_buf.read(self._grpc_buf_size)
            if read_buf:
                yield StateParams(params=read_buf)
            else:
                break

    def Swap(self, request_iter, context):
        edge = None
        read_buf = io.BytesIO()

        for req in request_iter:
            if edge is None:
                edge = self._edges[req.src]
                send_buf = edge.get_send_params()
                send_buf.seek(0)

            bin_params = send_buf.read(self._grpc_buf_size)
            read_buf.write(req.params)
            yield SwapParams(src=self._name, params=bin_params)

        edge.set_recv_params(read_buf)


class Contract:
    def __init__(self, name, nodes, device, model, interval=10, offset=0,
                 is_state=True, is_dual=True, is_avg=False, grpc_buf_size=524288, grpc_timeout=1.0):
        self._update_interval = interval
        self._update_cnt = offset
        self._next_edge = 0
        self._edges = OrderedDict()
        self._server = None

        # edge add
        node_name_list = []
        for node in nodes:
            node_name_list.append(node["name"])
        self_index = node_name_list.index(name)
        if self_index == 0:
            state_req = False
        else:
            state_req = True

        edge_info = {}
        for edge_name in nodes[self_index]["edges"]:
            keys = list(self._edges.keys())
            if edge_name not in keys:
                edge_index = node_name_list.index(edge_name)
                edge_addr = nodes[edge_index]["addr"] + ":" + nodes[edge_index]["port"]
                edge_info[edge_name] = {"index": edge_index, "addr": edge_addr}

        # gRPC Server start
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=len(node_name_list)*2))
        add_MnwServiceServicer_to_server(MnwGateway(name, self_index, self._edges, edge_info, device,
                                                    model, is_state, is_dual, is_avg, grpc_buf_size, grpc_timeout),
                                         self._server)
        port_str = '[::]:' + nodes[self_index]["port"]
        self._server.add_insecure_port(port_str)
        self._server.start()

        for edge_name in nodes[self_index]["edges"]:
            keys = list(self._edges.keys())
            if edge_name not in keys:
                con = self.hello(name, edge_info[edge_name]["addr"], model, state_req)
                if con:
                    self._edges[edge_name] = Edge(edge_info[edge_name], self_index, name, device,
                                                  model.state_dict(), is_state, is_dual, is_avg,
                                                  grpc_buf_size, grpc_timeout)
                    state_req = False

    def __del__(self):
        self._server.stop(0)

    @staticmethod
    def hello(name, addr, model, state_req, timeout_sec=10):
        connected = False
        cnt = 0
        while cnt < timeout_sec:
            with grpc.insecure_channel(addr) as channel:
                try:
                    req = PbParams(src=name)
                    stub = MnwServiceStub(channel)
                    response = stub.Hello(req)

                    if response.status == 200:
                        connected = True
                        if state_req:
                            device = next(model.parameters()).device
                            params_buf = io.BytesIO()
                            responses = stub.GetState(req)
                            for res in responses:
                                params_buf.write(res.params)
                            params_buf.seek(0)
                            model.load_state_dict(torch.load(io.BytesIO(params_buf.getvalue())))
                            model = model.to(device)
                        break
                except grpc.RpcError:
                    time.sleep(1)
                    cnt += 1

        return connected

    def swap(self):
        if len(list(self._edges.values())) > 0:
            self._update_cnt += 1
            if self._update_cnt >= self._update_interval:
                self._update_cnt = 0
                pull_edge = list(self._edges.values())[self._next_edge]
                is_connected = pull_edge.swap()
                if not is_connected:
                    remove_edge_name = list(self._edges.keys())[self._next_edge]
                    self._edges.pop(remove_edge_name)
                    logging.info("%s : edge removed.", remove_edge_name)

                if len(list(self._edges.values())) > 0:
                    self._next_edge += 1
                    if self._next_edge >= len(self._edges):
                        self._next_edge = 0

    def edges(self):
        return self._edges

