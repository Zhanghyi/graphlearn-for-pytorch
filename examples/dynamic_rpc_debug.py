import argparse
import torch.distributed.rpc as rpc
import time
import multiprocessing as mp

# python examples/dynamic_rpc_debug.py --rank 0,1
# python examples/dynamic_rpc_debug.py --rank 2,3

def worker(rank):
    options = rpc.TensorPipeRpcBackendOptions(
        _transports=['ibv', 'uv'],
        _channels=['mpt_uv', 'basic'],
        num_worker_threads=16,
        rpc_timeout=180,
        init_method='tcp://localhost:11111'
    )

    rpc.init_rpc(
        name=f'worker_{rank}',
        rank=rank,
        # world_size=None,
        rpc_backend_options=options
    )
    print(f'rank {rank} initialized')
    if rank == 0:
        time.sleep(5)
    time.sleep(5)
    rpc.shutdown()
    print(f'rank {rank} exited')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=str, help='Ranks of the processes')
    args = parser.parse_args()

    ranks = args.rank.split(',')
    num_processes = len(ranks)

    processes = []
    for rank in ranks:
        rank = int(rank)
        process = mp.Process(target=worker, args=(rank, ))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
