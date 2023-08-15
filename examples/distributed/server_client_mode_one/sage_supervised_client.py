import argparse
import os.path as osp
import time

import graphlearn_torch as glt
import torch
import torch.distributed
import torch.nn.functional as F

from ogb.nodeproppred import Evaluator
from torch_geometric.nn import GraphSAGE
from typing import List


@torch.no_grad()
def test(model, test_loader, dataset_name):
    evaluator = Evaluator(name=dataset_name)
    model.eval()
    xs = []
    y_true = []
    for i, batch in enumerate(test_loader):
        if i == 0:
            device = batch.x.device
        x = model(batch.x, batch.edge_index)[: batch.batch_size]
        xs.append(x.cpu())
        y_true.append(batch.y[: batch.batch_size].cpu())
        del batch

    xs = [t.to(device) for t in xs]
    y_true = [t.to(device) for t in y_true]
    y_pred = torch.cat(xs, dim=0).argmax(dim=-1, keepdim=True)
    y_true = torch.cat(y_true, dim=0).unsqueeze(-1)
    test_acc = evaluator.eval(
        {
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )["acc"]
    return test_acc


def run_client_proc(
    num_servers: int,
    num_clients: int,
    dataset_name: str,
    train_path_list: List[str],
    test_path_list: List[str],
    epochs: int,
    batch_size: int,
    master_addr: str,
    server_client_port: int,
    train_loader_master_port: int,
    test_loader_master_port: int,
):
    print(f"-- Initializing client ...")
    glt.distributed.init_client(
        num_servers=num_servers,
        num_clients=num_clients,
        client_rank=0,
        master_addr=master_addr,
        master_port=server_client_port,
        num_rpc_threads=4,
        client_group_name="dist_train_supervised_sage_client",
    )

    # Initialize training process group of PyTorch.
    current_ctx = glt.distributed.get_context()
    current_device = torch.device(current_ctx.rank % torch.cuda.device_count())

    print(
        f"--  Initializing training process group of PyTorch ..."
    )

    # Create distributed neighbor loader on remote server for training.
    print(f"Creating training dataloader ...")
    train_loader = glt.distributed.DistNeighborLoader(
        data=None,
        num_neighbors=[15, 10, 5],
        input_nodes=train_path_list,
        batch_size=batch_size,
        shuffle=True,
        collect_features=True,
        to_device=current_device,
        worker_options=glt.distributed.RemoteDistSamplingWorkerOptions(
            server_rank=[0],
            num_workers=1,
            worker_devices=["cpu"],
            worker_concurrency=1,
            master_addr=master_addr,
            master_port=train_loader_master_port,
            buffer_size="1GB",
            prefetch_size=2,
            worker_key="train",
        ),
    )

    # Create distributed neighbor loader on remote server for testing.
    print(f"--  Creating testing dataloader ...")
    test_loader = glt.distributed.DistNeighborLoader(
        data=None,
        num_neighbors=[15, 10, 5],
        input_nodes=test_path_list,
        batch_size=batch_size,
        shuffle=False,
        collect_features=True,
        to_device=current_device,
        worker_options=glt.distributed.RemoteDistSamplingWorkerOptions(
            server_rank=[0],
            num_workers=1,
            worker_devices=["cpu"],
            worker_concurrency=1,
            master_addr=master_addr,
            master_port=test_loader_master_port,
            buffer_size="1GB",
            prefetch_size=2,
            worker_key="test",
        ),
    )

    # Define model and optimizer.
    print(f"-- Initializing model and optimizer ...")
    torch.cuda.set_device(current_device)
    model = GraphSAGE(
        in_channels=128,
        hidden_channels=256,
        num_layers=3,
        out_channels=47,
    ).to(current_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Train and test.
    print(f"--  Start training and testing ...")
    for epoch in range(0, epochs):
        model.train()
        start = time.time()
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)[: batch.batch_size].log_softmax(
                dim=-1
            )
            loss = F.nll_loss(out, batch.y[: batch.batch_size])
            loss.backward()
            optimizer.step()

        end = time.time()
        print(
            f"--  Epoch: {epoch:03d}, Loss: {loss:.4f}, Epoch Time: {end - start}"
        )
        # Test accuracy.
        if epoch == 0 or epoch > (epochs // 2):
            test_acc = test(model, test_loader, dataset_name)
            print(f"--  Test Accuracy: {test_acc:.4f}")

    print(f"--  Shutdowning ...")
    glt.distributed.shutdown_client()

    print(f"--  Exited ...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments for distributed training of supervised SAGE with servers."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        help="The name of ogbn dataset.",
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default="../../data/products",
        help="The root directory (relative path) of partitioned ogbn dataset.",
    )
    parser.add_argument(
        "--num_dataset_partitions",
        type=int,
        default=2,
        help="The number of partitions of the dataset.",
    )
    parser.add_argument(
        "--num_server_nodes",
        type=int,
        default=2,
        help="Number of server nodes for remote sampling.",
    )
    parser.add_argument(
        "--num_client_nodes",
        type=int,
        default=2,
        help="Number of client nodes for training.",
    )
    parser.add_argument(
        "--node_rank",
        type=int,
        default=0,
        help="The node rank of the current role.",
    )
    parser.add_argument(
        "--num_server_procs_per_node",
        type=int,
        default=1,
        help="The number of server processes for remote sampling per server node.",
    )
    parser.add_argument(
        "--num_client_procs_per_node",
        type=int,
        default=2,
        help="The number of client processes for training per client node.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of training epochs. (client option)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for the training and testing dataloader.",
    )
    parser.add_argument(
        "--master_addr",
        type=str,
        default="localhost",
        help="The master address for RPC initialization.",
    )
    parser.add_argument(
        "--server_client_master_port",
        type=int,
        default=11110,
        help="The port used for RPC initialization across all servers and clients.",
    )
    parser.add_argument(
        "--training_pg_master_port",
        type=int,
        default=11111,
        help="The port used for PyTorch's process group initialization across all training processes.",
    )
    parser.add_argument(
        "--train_loader_master_port",
        type=int,
        default=11112,
        help="The port used for RPC initialization across all sampling workers of training loader.",
    )
    parser.add_argument(
        "--test_loader_master_port",
        type=int,
        default=11113,
        help="The port used for RPC initialization across all sampling workers of testing loader.",
    )
    args = parser.parse_args()

    print(
        f"--- Distributed training example of supervised SAGE with server-client mode. Client {args.node_rank} ---"
    )
    print(f"* dataset: {args.dataset}")
    print(f"* dataset root dir: {args.dataset_root_dir}")
    print(f"* total server nodes: {args.num_server_nodes}")
    print(f"* total client nodes: {args.num_client_nodes}")
    print(f"* node rank: {args.node_rank}")
    print(
        f"* number of server processes per server node: {args.num_server_procs_per_node}"
    )
    print(
        f"* number of client processes per client node: {args.num_client_procs_per_node}"
    )
    print(f"* master addr: {args.master_addr}")
    print(f"* server-client master port: {args.server_client_master_port}")
    print(f"* number of dataset partitions: {args.num_dataset_partitions}")

    num_servers = args.num_server_nodes * args.num_server_procs_per_node
    num_clients = args.num_client_nodes * args.num_client_procs_per_node
    root_dir = osp.join(osp.dirname(osp.realpath(__file__)), args.dataset_root_dir)

    print(f"* epochs: {args.epochs}")
    print(f"* batch size: {args.batch_size}")
    print(f"* training process group master port: {args.training_pg_master_port}")
    print(f"* training loader master port: {args.train_loader_master_port}")
    print(f"* testing loader master port: {args.test_loader_master_port}")

    print("--- Loading training and testing seeds ...")
    train_path_list = []
    for data_pidx in range(args.num_dataset_partitions):
        train_path_list.append(
            osp.join(
                root_dir, f"{args.dataset}-train-partitions", f"partition{data_pidx}.pt"
            )
        )

    test_path_list = []
    for data_pidx in range(args.num_dataset_partitions):
        test_path_list.append(
            osp.join(
                root_dir, f"{args.dataset}-test-partitions", f"partition{data_pidx}.pt"
            )
        )

    print("--- Launching client processes ...")
    mp_context = torch.multiprocessing.get_context("spawn")
    client_procs = []
    for local_proc_rank in range(args.num_client_procs_per_node):
        cproc = mp_context.Process(
            target=run_client_proc,
            args=(
                num_servers,
                num_clients,
                args.dataset,
                train_path_list,
                test_path_list,
                args.epochs,
                args.batch_size,
                args.master_addr,
                args.server_client_master_port,
                args.train_loader_master_port,
                args.test_loader_master_port,
            ),
        )
        client_procs.append(cproc)
    for cproc in client_procs:
        cproc.start()
    for cproc in client_procs:
        cproc.join()
