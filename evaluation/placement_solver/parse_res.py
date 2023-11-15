#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import argparse
import re
import os

model_list = ['graphsage']
dataset_list = ['tw', 'cf']
system_list = ['xgnn']
topo_list = ['4g_1', '4g_2', '6g', '8g']

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
    ret = vars(argparser.parse_args())
    if (ret['directory'] == None):
        argparser.error('Add --file argument')
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
    print("model\tdataset\ttopo\tsample_time\textract_time\tcache_ratio\tcache_hit\ttrain_time\te2e")
    for dataset in dataset_list:
        prefix_str = f"xgnn_graphsage_{dataset}_"
        for topo in topo_list:
            break_file_name = directory + "/" + prefix_str + topo + "_ics22_break.log"
            e2e_file_name = directory + "/" + prefix_str + topo + ".log"
            # print("file_name: ", file_name)
            print(f"graphsage\t{dataset}\t{topo}", end="")
            ret = parse_result_sgnn(
                    break_file_name,
                    ["sample_no_mark", "mark_cache_copy_time", "train_total",
                     "cache_percentage", "cache_hit_rate"])
            print("\t{:.2f}\t{:.2f}\t{:d}\t{:d}\t{:.2f}".format(ret[0], ret[1], ret[3], ret[4], ret[2]), end="")
            ret = parse_result_sgnn(
                    e2e_file_name,
                    ["total"])
            print("\t{:.2f}".format(ret[0]))
