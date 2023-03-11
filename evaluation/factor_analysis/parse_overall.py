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
dataset_list = ['tw', 'pa', 'uk', 'cf']
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

def parse_result_sgnn(file_name, show_list, split_str):
    pattern_time  = r'^test_result:epoch_time:(.+)=([0-9]+\.[0-9]+)$'
    pattern_cache = r'^test_result:(cache_.+)=([0-9]+\.[0-9]+)$'
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
            if (show_var == 'mark_cache_copy_time'):
                print('{:.2f}({:d},{:d})'.format(
                    res[show_var],
                    res['cache_percentage'],
                    res['cache_hit_rate']),
                    end=split_str)
            else:
                print('{:.2f}'.format(res[show_var]), end=split_str)
        else:
            print(OOM, end=split_str)

if __name__ == '__main__':
    arguments   = parse_args()
    directory   = arguments['directory']
    for dataset in dataset_list:
        prefix_str = f"xgnn_graphsage_{dataset}_"
        if (dataset in ["tw", "pa"]):
            prefix_str = f"xgnn_gcn_{dataset}_"
        for i in range(1, 5):
            file_name = directory + "/" + prefix_str + str(i) + ".log"
            # print("file_name: ", file_name)
            parse_result_sgnn(
                    file_name,
                    [
                        "sample_no_mark",
                        "mark_cache_copy_time",
                    ],
                    split_str = split_in)
            parse_result_sgnn(file_name, ['train_total'], split_out)
