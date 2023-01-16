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

mock = False
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

def parse_result_dgl(file_name, show_list, split_str, end_str):
    pattern_time  = r'^test_result:(.+)=([0-9]+\.[0-9]+)$'
    res = {}
    if (os.path.exists(file_name)):
        with open(file_name, 'r') as file:
            for line in file:
                m = re.match(pattern_time, line)
                if (m):
                    res[m.group(1)] = float(m.group(2))
    for show_var in show_list:
        end = split_str
        if (show_var == show_list[-1]):
            end = end_str
        if (res):
            print('{:.2f}'.format(res[show_var]), end=end)
        else:
            print(OOM, end=end)

if __name__ == '__main__':
    arguments   = parse_args()
    directory   = arguments['directory']
    if (mock):
        print('model\tdataset\tdgl\tsgnn\txgnn')
    for model in model_list:
        for dataset in dataset_list:
            if (mock):
                print(f'{model}\t{dataset}\t', end='')
            else:
                print('\n&&{', end='')
            for system in system_list:
                file_name_pipe = f'{directory}/{system}_{model}_{dataset}.log'
                if (system == 'dgl'):
                    parse_result_dgl(file_name_pipe,
                            [
                                'sample_time',
                                'copy_time',
                                'train_time',
                                'epoch_time'
                            ],
                            split_str = split_in,
                            end_str = split_out)
                else:
                    file_name_break = \
                            f'{directory}/{system}_{model}_{dataset}_break.log'
                    parse_result_sgnn(
                            file_name_break,
                            [
                                'sample_no_mark',
                                'mark_cache_copy_time',
                                'train_total'
                            ],
                            split_str = split_in)
                    if (system == 'xgnn'):
                        parse_result_sgnn(file_name_pipe, ['total'], split_end)
                    else:
                        parse_result_sgnn(file_name_pipe, ['total'], split_out)
