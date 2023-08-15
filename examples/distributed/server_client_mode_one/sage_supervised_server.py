import argparse
import json
import os.path as osp

import graphlearn_torch as glt
import torch
import torch.distributed


# CUDA_VISIBLE_DEVICES=3 python examples/distributed/server_client_mode_one/sage_supervised_server.py
def run_server_proc(handle, server_rank, dataset):
    glt.distributed.init_server(
        num_servers=handle["num_servers"],
        num_clients=handle["num_clients"],
        server_rank=server_rank,
        dataset=dataset,
        master_addr=handle["master_addr"],
        master_port=handle["server_client_master_port"],
        num_rpc_threads=16,
        server_group_name="dist_train_supervised_sage_server",
    )

    print(f"-- [Server {server_rank}] Waiting for exit ...")
    glt.distributed.wait_and_shutdown_server()
    print(f"-- [Server {server_rank}] Exited ...")


def launch_graphlearn_torch_server(handle, config, server_index):
    # TODO(hongyi): hard code arxiv for test now
    dataset_name = "ogbn-arxiv"
    dataset_root_dir = "/home/hongyizhang/arxiv"
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
    dataset = glt.distributed.DistDataset()
    dataset.load(
        root_dir=osp.join(root_dir, f"{dataset_name}-partitions"),
        partition_idx=0,
        graph_mode="ZERO_COPY",
        whole_node_label_file=osp.join(root_dir, f"{dataset_name}-label", "label.pt"),
    )
    server_rank = server_index

    import torch

    mp_context = torch.multiprocessing.get_context("spawn")
    print(f"-- [Server {server_rank}] Initializing server ...")

    proc = mp_context.Process(
        target=run_server_proc, args=(handle, server_rank, dataset)
    )
    proc.start()
    proc.join()


if __name__ == "__main__":
    handle = {
        "master_addr": "localhost",
        "num_servers": 1,
        "num_clients": 1,
        "server_client_master_port": 11110,
    }
    config = {"a": "b"}
    launch_graphlearn_torch_server(handle, config, 0)
