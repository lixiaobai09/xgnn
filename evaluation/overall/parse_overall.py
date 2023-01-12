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

def parse_args():
    argparser = argparse.ArgumentParser('Acc Timeline Parser')
    argparser.add_argument('-d', '--directory', type=str,
            help='the log file directory path to parse')
    ret = vars(argparser.parse_args())
    if (ret['directory'] == None):
        argparser.error('Add --file argument')
    return ret

def parse_result(file_name, show_list, split_str):
    pattern = r'^test_result:epoch_time:(.+)=([0-9]+\.[0-9]+)$'
    res = {}
    if (os.path.exists(file_name)):
        with open(file_name, 'r') as file:
            for line in file:
                m = re.match(pattern, line)
                if m:
                    res[m.group(1)] = float(m.group(2))
    for show_var in show_list:
        if (res):
            print('{}'.format(res[show_var]), end=split_str)
        else:
            print('none', end=split_str)

if __name__ == '__main__':
    arguments   = parse_args()
    directory   = arguments['directory']
    for model in model_list:
        for dataset in dataset_list:
            for system in system_list:
                file_name_pipe = f'{directory}/{system}_{model}_{dataset}.log'
                file_name_break = \
                        f'{directory}/{system}_{model}_{dataset}_break.log'
                parse_result(
                        file_name_break,
                        [
                            'sample_no_mark',
                            'mark_cache_copy_time',
                            'train_total'
                        ],
                        split_str = '\t')
                parse_result(file_name_pipe, ['total'], '\n')
