#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import argparse
import re
import os

model_list = ['graphsage']
dataset_list = ['tw', 'cf']
system_list = ['xgnn']
topo_list = [['4g_1', '4G_f'],
             ['4g_2', '4G_c'],
             ['6g', '6G'],
             ['8g', '8G']]

mock = True
if (mock):
    OOM = 'OOM'
    split_in  = '\t'
    split_out = ' | '
    split_end = '\n'
else:
    OOM = '\\texttt{OOM}'
    split_in  = '} & {'
    split_out = '}\n&&{'
    split_end = '} \\\\\n'

def parse_args():
    argparser = argparse.ArgumentParser('Acc Timeline Parser')
    argparser.add_argument('-d', '--directory', type=str,
            help='the log file directory path to parse')
    argparser.add_argument('--dataset', type=str,
            help=f'the dataset to parse, only in {dataset_list}')
    ret = vars(argparser.parse_args())
    if (ret['directory'] == None):
        argparser.error('Add --directory argument')
    if (ret['dataset'] not in dataset_list):
        argparser.error('Add --dataset argument')
    return ret

def parse_result_sgnn(file_name, show_list):
    pattern_time  = r'^test_result:epoch_time:(.+)=([0-9]+\.[0-9]+)$'
    pattern_cache = r'^test_result:(cache_.+)=([0-9]+\.[0-9]+)$'
    ret = []
    res = {}
    if (os.path.exists(file_name)):
        with open(file_name, 'r') as file:
            for line in file:
                m = re.match(pattern_time, line)
                if (m):
                    res[m.group(1)] = float(m.group(2))
                m = re.match(pattern_cache, line)
                if (m):
                    res[m.group(1)] = int(float(m.group(2)) * 100.0)
    for show_var in show_list:
        if (res):
            ret.append(res[show_var])
        else:
            ret.append(0.0)
    return ret

if __name__ == '__main__':
    arguments   = parse_args()
    directory   = arguments['directory']
    dataset     = arguments['dataset']
    print("mode\ttopo\tClique\tSolver\tCliqueOpt")
    for (log_name, topo_name) in topo_list:
        file = f"{directory}/xgnn_graphsage_{dataset}_{log_name}_origin_break.log"
        origin_time = parse_result_sgnn(file, ["mark_cache_copy_time"])[0]
        file = f"{directory}/xgnn_graphsage_{dataset}_{log_name}_clique_break.log"
        clique_time = parse_result_sgnn(file, ["mark_cache_copy_time"])[0]
        file = f"{directory}/xgnn_graphsage_{dataset}_{log_name}_ics22_break.log"
        ics22_time = parse_result_sgnn(file, ["mark_cache_copy_time"])[0]
        print("graphsage\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}".format(
            dataset, topo_name,
            clique_time, origin_time, ics22_time))
