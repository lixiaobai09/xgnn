import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dgl.nn.pytorch import GraphConv
import dgl.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import os
import sys
import samgraph.torch as sam
import datetime
from common_config import *


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GraphConv(in_feats, n_hidden, activation=activation, allow_zero_in_degree=True))
        # hidden layers
        for _ in range(n_layers - 2):
            self.layers.append(
                GraphConv(n_hidden, n_hidden, activation=activation, allow_zero_in_degree=True))
        # output layer
        self.layers.append(
            GraphConv(n_hidden, n_classes, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(blocks[i], h)
        return h


def parse_args(default_run_config):
    argparser = argparse.ArgumentParser("GCN Training")

    add_common_arguments(argparser, default_run_config)

    argparser.add_argument('--fanout', nargs='+',
                           type=int, default=default_run_config['fanout'])
    argparser.add_argument('--lr', type=float,
                           default=default_run_config['lr'])
    argparser.add_argument('--dropout', type=float,
                           default=default_run_config['dropout'])
    argparser.add_argument('--weight-decay', type=float,
                           default=default_run_config['weight_decay'])
    argparser.add_argument('--use-dist-graph', type=float,
                           default=0.0)
    argparser.add_argument('--use-ics22-song-solver', action='store_true',
                           default=False)
    argparser.add_argument('--clique-size', type=int, default=0)
    argparser.add_argument('--ics22-song-alpha', type=float, default=0.0)
    argparser.add_argument('--dist-graph-part-cpu', type=int, default=0)
    argparser.add_argument('--unified-memory', action='store_true',
                           default=False)
    argparser.add_argument('--unified-memory-percentage', type=float, nargs='+', default=argparse.SUPPRESS)
    argparser.add_argument('--part-cache', action='store_true', default=False)
    argparser.add_argument('--gpu-extract', action='store_true', default=False)

    return vars(argparser.parse_args())


def get_run_config():
    run_config = {}

    run_config.update(get_default_common_config(run_mode=RunMode.SGNN))
    run_config['sample_type'] = 'khop2'

    run_config['fanout'] = [5, 10, 15]
    run_config['lr'] = 0.003
    run_config['dropout'] = 0.5
    run_config['weight_decay'] = 0.0005

    run_config.update(parse_args(run_config))

    process_common_config(run_config)
    assert(run_config['arch'] == 'arch6')
    assert(run_config['sample_type'] != 'random_walk')

    run_config['num_fanout'] = run_config['num_layer'] = len(
        run_config['fanout'])

    if 'PEEK_MEMORY' in dict(os.environ) and dict(os.environ)['PEEK_MEMORY'] == '1':
        run_config['peek_memory'] = True
    else:
        run_config['peek_memory'] = False

    print_run_config(run_config)

    return run_config


def run_init(run_config):
    sam.config(run_config)
    sam.data_init()

    if run_config['validate_configs']:
        sys.exit()


def run(worker_id, run_config):
    num_worker = run_config['num_worker']
    global_barrier = run_config['global_barrier']

    ctx = run_config['workers'][worker_id]
    device = torch.device(ctx)

    print('[Worker {:d}/{:d}] Started with PID {:d}({:s})'.format(
        worker_id, num_worker, os.getpid(), torch.cuda.get_device_name(ctx)))
    init_t1 = time.time()
    sam.sample_init(worker_id, ctx)
    init_t2 = time.time()
    sam.train_init(worker_id, ctx)
    init_t3 = time.time()
    print("sample init time: ", (init_t2 - init_t1))
    print("train init time : ", (init_t3 - init_t2))

    if num_worker > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = num_worker
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=worker_id,
                                             timeout=datetime.timedelta(seconds=get_default_timeout()))

    in_feat = sam.feat_dim()
    num_class = sam.num_class()
    num_layer = run_config['num_layer']

    model = GCN(in_feat, run_config['num_hidden'], num_class,
                num_layer, F.relu, run_config['dropout'])
    model = model.to(device)
    if num_worker > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device)

    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(
        model.parameters(), lr=run_config['lr'], weight_decay=run_config['weight_decay'])

    num_epoch = sam.num_epoch()
    num_step = sam.num_local_step()

    model.train()

    epoch_sample_total_times = []
    epoch_sample_nodes = []
    epoch_sample_times = []
    epoch_sample_coo_times = []
    epoch_get_cache_miss_index_times = []
    epoch_copy_times = []
    epoch_convert_times = []
    epoch_train_times = []
    epoch_total_times_python = []
    epoch_train_total_times_profiler = []
    epoch_cache_hit_rates = []
    epoch_cache_local_hit_rates = []

    copy_times = []
    convert_times = []
    train_times = []
    total_times = []

    # run start barrier
    global_barrier.wait()
    print('[Worker {:d}] run for {:d} epochs with {:d} steps'.format(
        worker_id, num_epoch, num_step))
    run_start = time.time()

    peek_memory = 0

    for epoch in range(num_epoch):
        # epoch start barrier
        global_barrier.wait()

        tic = time.time()

        for step in range(worker_id, num_step * num_worker, num_worker):
            t0 = time.time()
            if not run_config['pipeline']:
              sam.sample_once()
            elif epoch + step == worker_id:
              sam.extract_start(0)
            batch_key = sam.get_next_batch()
            t1 = time.time()
            blocks, batch_input, batch_label = sam.get_dgl_blocks(
                batch_key, num_layer)
            t2 = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_input)
            loss = loss_fcn(batch_pred, batch_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            event_sync()
            batch_input = None
            batch_label = None
            blocks = None

            t3 = time.time()

            copy_time = sam.get_log_step_value(epoch, step, sam.kLogL1CopyTime)
            convert_time = t2 - t1
            train_time = t3 - t2
            total_time = t3 - t1

            sam.log_step(epoch, step, sam.kLogL1TrainTime, train_time)
            sam.log_step(epoch, step, sam.kLogL1ConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochConvertTime, convert_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTrainTime, train_time)
            sam.log_epoch_add(epoch, sam.kLogEpochTotalTime, total_time)

            copy_times.append(copy_time)
            convert_times.append(convert_time)
            train_times.append(train_time)
            total_times.append(total_time)

            # sam.report_step_average(epoch, step)

            if run_config['peek_memory']:
                mem_info = torch.cuda.mem_get_info(device)
                peek_memory = max(peek_memory, mem_info[1] - mem_info[0])

        event_sync()

        # sync the train workers
        if num_worker > 1:
            torch.distributed.barrier()

        toc = time.time()

        epoch_total_times_python.append(toc - tic)

        # epoch end barrier
        global_barrier.wait()

        feat_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochFeatureBytes)
        miss_nbytes = sam.get_log_epoch_value(
            epoch, sam.kLogEpochMissBytes)
        epoch_cache_hit_rates.append(
            (feat_nbytes - miss_nbytes) / feat_nbytes)
        if run_config['part_cache'] or run_config['use_ics22_song_solver']:
            local_cache_nbytes = sam.get_log_epoch_value(epoch, sam.kLogEpochLocalCacheBytes)
            epoch_cache_local_hit_rates.append(local_cache_nbytes / feat_nbytes)
        epoch_sample_total_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTotalTime)
        )
        epoch_sample_nodes.append(sam.get_log_epoch_value(epoch, sam.kLogEpochNumSample))
        epoch_sample_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochSampleTime)
        )
        epoch_sample_coo_times.append(sam.get_log_epoch_value(epoch, sam.kLogEpochSampleCooTime))
        epoch_get_cache_miss_index_times.append(
            sam.get_log_epoch_value(
                epoch, sam.KLogEpochSampleGetCacheMissIndexTime)
        )
        epoch_copy_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochCopyTime))
        epoch_convert_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochConvertTime))
        epoch_train_times.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTrainTime))
        epoch_train_total_times_profiler.append(
            sam.get_log_epoch_value(epoch, sam.kLogEpochTotalTime))
        (free, total) = torch.cuda.mem_get_info()
        peek_memory = max(peek_memory, total - free)
        if worker_id == 0:
            used = (total - free) / 1024 / 1024 / 1024
            torch_mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024 / 1024
            print('Epoch {:05d} | Epoch Time {:.4f} | Sample {:.4f} | Copy {:.4f} | Total Train(Profiler) {:.4f} | GPU memory {:.2f} | torch memory used {:.2f}'.format(
                epoch, epoch_total_times_python[-1], epoch_sample_total_times[-1], epoch_copy_times[-1], epoch_train_total_times_profiler[-1],
                used, torch_mem_reserved))
            # print("stats:\n", torch.cuda.memory_stats())
            # print("summary:\n", torch.cuda.memory_summary())
            # print("snapshot:\n", torch.cuda.memory_snapshot())

    # sync the train workers
    if num_worker > 1:
        torch.distributed.barrier()

    # run end barrier
    global_barrier.wait()
    run_end = time.time()

    print('[Train  Worker {:d}] Avg Epoch {:.4f} | Sample {:.4f} | Copy {:.4f} | Train Total (Profiler) {:.4f}'.format(
          worker_id, np.mean(epoch_total_times_python[1:]), np.mean(epoch_sample_total_times[1:]), np.mean(epoch_copy_times[1:]), np.mean(epoch_train_total_times_profiler[1:])))
    if run_config['part_cache'] or run_config['use_ics22_song_solver']:
        print('[Train Worker {}] Partition Cache Hit Rate {:.2f} | Local Hit Rate {:.2f}'.format(
            worker_id, np.mean(epoch_cache_hit_rates[1:]), np.mean(epoch_cache_local_hit_rates[1:])))

    global_barrier.wait()  # barrier for pretty print

    if worker_id == 0:
        sam.report_step_average(epoch, step)
        sam.report_init()
        test_result = []
        test_result.append(('sample_time', np.mean(epoch_sample_times[1:])))
        test_result.append(('get_cache_miss_index_time', np.mean(
            epoch_get_cache_miss_index_times[1:])))
        test_result.append(
            ('epoch_time:sample_total', np.mean(epoch_sample_total_times[1:])))
        test_result.append(('epoch_time:sample_coo_time', np.mean(epoch_sample_coo_times[1:])))
        test_result.append(
            ('epoch_time:sample_no_mark', np.mean(epoch_sample_total_times[1:]) - np.mean(epoch_get_cache_miss_index_times[1:])))
        test_result.append(('epoch_time:copy_time',
                           np.mean(epoch_copy_times[1:])))
        test_result.append(('epoch_time:mark_cache_copy_time',
            np.mean(epoch_copy_times[1:]) + np.mean(epoch_get_cache_miss_index_times[1:])))
        test_result.append(('convert_time', np.mean(epoch_convert_times[1:])))
        test_result.append(('train_time', np.mean(epoch_train_times[1:])))
        test_result.append(('epoch_time:train_total', np.mean(
            epoch_train_total_times_profiler[1:])))
        # test_result.append(('epoch_time:mark_cache_train_total',
        #     np.mean(epoch_train_total_times_profiler[1:]) + np.mean(epoch_get_cache_miss_index_times[1:])))
        test_result.append(
            ('cache_percentage', run_config['cache_percentage']))
        test_result.append(('cache_hit_rate', np.mean(
            epoch_cache_hit_rates[1:])))
        test_result.append(('epoch:sample_nodes', np.mean(epoch_sample_nodes[1:])))
        # thpt, M SEPS
        test_result.append(('epoch:sample_thpt', np.mean(np.array(epoch_sample_nodes[1:]) / np.array(epoch_sample_times[1:])) / 1e6))
        test_result.append(
            ('epoch_time:total', np.mean(epoch_total_times_python[1:])))
        test_result.append(('run_time', run_end - run_start))
        for k, v in test_result:
            print('test_result:{:}={:.2f}'.format(k, v))

        # sam.dump_trace()

    sam.shutdown()

    print(f"memory({device}):graph={sam.get_log_init_value(sam.kLogInitL1GraphMemory)}")
    print(f"memory({device}):feature={sam.get_log_init_value(sam.kLogInitL1FeatMemory)}")
    print(f"memory({device}):workspace_total={sam.get_log_init_value(sam.kLogInitL1WorkspaceTotalMemory)}")
    print(f'memory({device}):peek_memory={peek_memory}')


if __name__ == '__main__':
    run_config = get_run_config()
    init_t0 = time.time()
    run_init(run_config)
    print("init run time: ", (time.time() - init_t0))

    num_worker = run_config['num_worker']

    # global barrier is used to sync all the sample workers and train workers
    run_config['global_barrier'] = mp.Barrier(
        num_worker, timeout=get_default_timeout())

    if num_worker == 1:
        run(0, run_config)
    else:
        workers = []
        # sample processes
        for worker_id in range(num_worker):
            p = mp.Process(target=run, args=(worker_id, run_config))
            p.start()
            workers.append(p)

        ret = sam.wait_one_child()
        if ret != 0:
            for p in workers:
                p.kill()
        for p in workers:
            p.join()

        if ret != 0:
            sys.exit(1)
