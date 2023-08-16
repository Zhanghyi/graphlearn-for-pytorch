import json
import base64

import os.path as osp

import graphlearn_torch as glt
import torch

# python examples/distributed/server_client_mode_one/sage_supervised_server.py
def decode_arg(arg):
    if isinstance(arg, dict):
        return arg
    return json.loads(
        base64.b64decode(arg.encode("utf-8", errors="ignore")).decode(
            "utf-8", errors="ignore"
        )
    )

def run_server_proc(proc_rank, handle, config, server_rank, dataset):
    glt.distributed.init_server(
        num_servers=handle["num_servers"],
        num_clients=handle["num_clients"],
        server_rank=server_rank,
        dataset=dataset,
        master_addr=handle["master_addr"],
        master_port=handle["server_client_master_port"],
        num_rpc_threads=16,
        # server_group_name="dist_train_supervised_sage_server",
    )

    print(f"-- [Server {server_rank}] Waiting for exit ...")
    glt.distributed.wait_and_shutdown_server()
    print(f"-- [Server {server_rank}] Exited ...")


def launch_graphlearn_torch_server(handle, config, server_rank):
    # TODO(hongyi): hard code arxiv for test now
    dataset_name = "ogbn-arxiv"
    dataset_root_dir = "/home/hongyizhang/arxiv"
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), dataset_root_dir)
    dataset = glt.distributed.DistDataset()
    dataset.load(
        root_dir=osp.join(root_dir, f"{dataset_name}-partitions"),
        partition_idx=0,
        graph_mode="CPU",
        whole_node_label_file=osp.join(root_dir, f"{dataset_name}-label", "label.pt"),
    )

    print(f"-- [Server {server_rank}] Initializing server ...")

    torch.multiprocessing.spawn(
        fn=run_server_proc, args=(handle, config, server_rank, dataset), nprocs=1
    )


if __name__ == "__main__":
    handle = {
        "master_addr": "localhost",
        "num_servers": 1,
        "num_clients": 1,
        "server_client_master_port": 11110,
    }
    config = {"a": "b"}
    launch_graphlearn_torch_server(handle, config, 0)
