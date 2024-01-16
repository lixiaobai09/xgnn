#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import argparse
import re
import os

model_list = ['gcn', 'graphsage', 'pinsage']
system_list = ['sgnn', 'xgnn']
dataset_list = ['tw', 'pa', 'uk', 'cf']
dataset_node_size = {
        'tw': 41652230,
        'pa': 111059956,
        'uk': 77741046,
        'cf': 65608366,
        }
dataset_feat_dim = {
        'tw': 256,
        'pa': 128,
        'uk': 256,
        'cf': 140,
        }
dataset_feat_size = {
        'tw': 39.7227,
        'pa': 52.9575,
        'uk': 74.1396,
        'cf': 34.2174,
        }

mock = False
gpu_mem = 16
gpu_num = 4

def parse_args():
    argparser = argparse.ArgumentParser('Search Max Memory occupy')
    argparser.add_argument('-d', '--directory', type=str,
            help='the log file directory path to parse')
    ret = vars(argparser.parse_args())
    if (ret['directory'] == None):
        argparser.error('Add --file argument')
    return ret

def find_one(file_name):
    pattern_cuda_mem = r'cuda(\d+): usage: (\d+\.\d+) GB'
    max_mem = 0.0
    if (os.path.exists(file_name)):
        with open(file_name, 'r') as file:
            for line in file:
                m = re.search(pattern_cuda_mem, line)
                if (m):
                    # print(file_name, ' mach: ', m.group(0))
                    mem_usage = float(m.group(2))
                    if (max_mem < mem_usage):
                        max_mem = mem_usage
    return max_mem



if __name__ == '__main__':
    arguments = parse_args()
    directory = arguments['directory']
    dic = {}
    for system in system_list:
        for dataset in dataset_list:
            for model in model_list:
                file_name = directory + "/" + system + "_" + model + "_" + dataset + ".log.err"
                ret = find_one(file_name)
                if (mock):
                    print(f'{system} {dataset} {model} max_mem: {ret} GB')
                key = (system + dataset + model)
                dic[key] = ret
    for dataset in dataset_list:
        for model in model_list:
            sgnn_max_mem = dic["sgnn" + dataset + model]
            xgnn_max_mem = dic["xgnn" + dataset + model]
            hash_mem = dataset_node_size[dataset] * 4 / 1024 / 1024 / 1024
            feat_mem = dataset_feat_size[dataset]
            sgnn_pert = 0.0
            xgnn_pert = 0.0
            if (sgnn_max_mem != 0):
                sgnn_pert = (gpu_mem - sgnn_max_mem - hash_mem) / dataset_feat_size[dataset]
            if (xgnn_max_mem != 0):
                xgnn_pert = (gpu_mem - xgnn_max_mem - hash_mem) / dataset_feat_size[dataset] * gpu_num
            if (mock):
                print(f'{dataset} hash_mem: {hash_mem}, feat_mem: {feat_mem}')
            print(f'{dataset}_{model}: ',
                    str(sgnn_max_mem) + ", " + '{:.4f}'.format(sgnn_pert) + " | " +
                    str(xgnn_max_mem) + ", " + '{:.4f}'.format(xgnn_pert)
                    )

