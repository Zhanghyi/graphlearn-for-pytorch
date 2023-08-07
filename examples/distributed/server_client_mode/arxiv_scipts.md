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