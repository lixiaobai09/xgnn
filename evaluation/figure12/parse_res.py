#!/usr/bin/python3
# -*- coding: UTF-8 -*-
import argparse
import re
import os

dataset_short_name = {
        'twitter': 'tw',
        'papers': 'pa',
        'uk-2006-05': 'uk',
        'com-friendster': 'cf'
        }
model_list = ['gcn', 'graphsage', 'pinsage']
dataset_list = ['pa', 'uk']
system_list = ['dgl', 'sgnn', 'xgnn']

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
    pattern_num_step = r'.*epochs with ([0-9]+) steps$'
    pattern_memory = r'^memory\(cuda:.+\):(.+)=([0-9]+)'
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
                m = re.match(pattern_num_step, line)
                if (m):
                    res['num_step'] = int(m.group(1))
                m = re.match(pattern_memory, line)
                if (m):
                    res[m.group(1)] = float(m.group(2))
    for show_var in show_list:
        if (res):
            ret.append(res[show_var])
        else:
            ret.append(0.0)
    return ret

if __name__ == '__main__':
    arguments   = parse_args()
    directory   = arguments['directory']
    print("# worker\tfeature\tgraph\tsample_time\textract_time\ttrain_time\tepoch_time\tstep")
    for num_gpu in [1,2,4,8]:
        file_name = f"{directory}/xgnn_graphsage_uk_{num_gpu}wk-bk.log"
        break_list = parse_result_sgnn(file_name,
                ["feature", "graph", "sample_no_mark", "mark_cache_copy_time", "train_total", "num_step"])
        file_name = f"{directory}/xgnn_graphsage_uk_{num_gpu}wk.log"
        epoch_time = parse_result_sgnn(file_name, ["total"])[0]
        print("{:d}\t{:.1f}\t{:.1f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:d}".format(
            num_gpu,
            break_list[0] * num_gpu,
            break_list[1] * num_gpu,
            break_list[2], break_list[3], break_list[4],
            epoch_time,
            break_list[5]))
