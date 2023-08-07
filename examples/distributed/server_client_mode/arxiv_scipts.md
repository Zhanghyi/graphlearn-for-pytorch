python setup.py bdist_wheel && pip install dist/* --force-reinstall
python partition_ogbn_dataset.py --dataset=ogbn-arxiv --num_partitions=2 --root_dir=/home/hongyizhang/arxiv
# server node 0:
CUDA_VISIBLE_DEVICES=3,4 python examples/distributed/server_client_mode/sage_supervised_server.py \
  --num_server_nodes=2 --num_client_nodes=2 --node_rank=0 --num_dataset_partitions=2 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost --dataset=ogbn-arxiv --dataset_root_dir=/home/hongyizhang/arxiv 

# server node 1:
CUDA_VISIBLE_DEVICES=5,6 python examples/distributed/server_client_mode/sage_supervised_server.py \
  --num_server_nodes=2 --num_client_nodes=2 --node_rank=1 --num_dataset_partitions=2 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost --dataset=ogbn-arxiv --dataset_root_dir=/home/hongyizhang/arxiv

# client node 0:
CUDA_VISIBLE_DEVICES=3,4 python examples/distributed/server_client_mode/sage_supervised_client.py \
  --num_server_nodes=2 --num_client_nodes=2 --node_rank=0 --num_dataset_partitions=2 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost --dataset=ogbn-arxiv --dataset_root_dir=/home/hongyizhang/arxiv --epochs=2

# client node 1:
CUDA_VISIBLE_DEVICES=5,6 python examples/distributed/server_client_mode/sage_supervised_client.py \
  --num_server_nodes=2 --num_client_nodes=2 --node_rank=1 --num_dataset_partitions=2 \
  --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost --dataset=ogbn-arxiv --dataset_root_dir=/home/hongyizhang/arxiv --epochs=2

### server 0
root@j63c09243:/home/hongyizhang/tmp/graphlearn-for-pytorch# CUDA_VISIBLE_DEVICES=3,4 python examples/distributed/server_client_mode/sage_supervised_server.py   --num_server_nodes=2 --num_client_nodes=2 --node_rank=0 --num_dataset_partitions=2   --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost --dataset=ogbn-arxiv --dataset_root_dir=/home/hongyizhang/arxiv 
--- Distributed training example of supervised SAGE with server-client mode. Server 0 ---
* dataset: ogbn-arxiv
* dataset root dir: /home/hongyizhang/arxiv
* total server nodes: 2
* node rank: 0
* number of server processes per server node: 1
* number of client processes per client node: 2
* master addr: localhost
* server-client master port: 11110
* number of dataset partitions: 2
--- Loading data partition ...
--- Launching server processes ...
-- [Server 0] Initializing server ...
-- [Server 0] Waiting for exit ...

### server 1
root@j63c09243:/home/hongyizhang/tmp/graphlearn-for-pytorch# CUDA_VISIBLE_DEVICES=5,6 python examples/distributed/server_client_mode/sage_supervised_server.py   --num_server_nodes=2 --num_client_nodes=2 --node_rank=1 --num_dataset_partitions=2   --num_server_procs_per_node=1 --num_client_procs_per_node=2 --master_addr=localhost --dataset=ogbn-arxiv --dataset_root_dir=/home/hongyizhang/arxiv
--- Distributed training example of supervised SAGE with server-client mode. Server 1 ---
* dataset: ogbn-arxiv
* dataset root dir: /home/hongyizhang/arxiv
* total server nodes: 2
* node rank: 1
* number of server processes per server node: 1
* number of client processes per client node: 2
* master addr: localhost
* server-client master port: 11110
* number of dataset partitions: 2
--- Loading data partition ...
--- Launching server processes ...
-- [Server 1] Initializing server ...
Process SpawnProcess-1:
Traceback (most recent call last):
  File "/usr/lib/python3.8/multiprocessing/process.py", line 315, in _bootstrap
    self.run()
  File "/usr/lib/python3.8/multiprocessing/process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "/home/hongyizhang/tmp/graphlearn-for-pytorch/examples/distributed/server_client_mode/sage_supervised_server.py", line 30, in run_server_proc
    glt.distributed.init_server(
  File "/usr/local/lib/python3.8/dist-packages/graphlearn_torch/distributed/dist_server.py", line 212, in init_server
    init_rpc(master_addr, master_port, num_rpc_threads, request_timeout, dynamically=True)
  File "/usr/local/lib/python3.8/dist-packages/graphlearn_torch/distributed/rpc.py", line 264, in init_rpc
    rpc.init_rpc(
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/rpc/__init__.py", line 196, in init_rpc
    _init_rpc_backend(backend, store, name, rank, world_size, rpc_backend_options)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/rpc/__init__.py", line 231, in _init_rpc_backend
    rpc_agent = backend_registry.init_backend(
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/rpc/backend_registry.py", line 101, in init_backend
    return backend.value.init_backend_handler(*args, **kwargs)
  File "/usr/local/lib/python3.8/dist-packages/torch/distributed/rpc/backend_registry.py", line 371, in _tensorpipe_init_backend_handler
    agent = TensorPipeAgent(
ValueError: stoi
