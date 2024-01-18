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
system_list = ['quiver']

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
    argparser.add_argument('--quiver-dir', type=str)
    ret = vars(argparser.parse_args())
    return ret

def get_value_in_quiver_log(file, re_pattern):
    if (os.path.exists(file)):
        with open(file, 'r') as f:
            for line in f:
                m = re.match(re_pattern, line)
                if (m):
                    return m.group(1)

def parse_result_quiver(quiver_dir, model, ds, split_str, end_str):
    if model == 'pinsage':
        print("X X X X", end=end_str)
        return
    if model == 'graphsage':
        model = 'sage'
    if ds == 'cf':
        print("{}{}{}{}{}{}".format(
                "OOM", split_str,
                "OOM", split_str,
                "OOM", split_str
            ), end=end_str)
        return
    # e2e
    quiver_log_pl0 = f'{quiver_dir}/{model}-{ds}-pl0'
    re_p = r'^\[Avg Epoch Time\] ([0-9]+\.[0-9]+)$'
    e2e_time = float(get_value_in_quiver_log(quiver_log_pl0, re_p))
    # breakdown
    quiver_log_pl1 = f'{quiver_dir}/{model}-{ds}-pl1'
    re_p = r'^\[Avg Sample Time\] ([0-9]+\.[0-9]+)$'
    s_time = float(get_value_in_quiver_log(quiver_log_pl1, re_p))
    re_p = r'^\[Avg Extract Time\] ([0-9]+\.[0-9]+)$'
    e_time = float(get_value_in_quiver_log(quiver_log_pl1, re_p))
    re_p = r'^\[Avg Train Time\] ([0-9]+\.[0-9]+)$'
    t_time = float(get_value_in_quiver_log(quiver_log_pl1, re_p))
    # cache
    quiver_log_pl2 = f'{quiver_dir}/{model}-{ds}-pl2'
    re_p = r'^LOG>>> (.+)% data cached$'
    cache_ratio = int(get_value_in_quiver_log(quiver_log_pl2, re_p))
    re_p = r'^\[Extract Hit Rate\] ([0-9]+\.[0-9]+)$'
    hit_rate = int(float(get_value_in_quiver_log(quiver_log_pl2, re_p)) * 100)
    print("{:.2f}{}{:.2f}({:d}, {:d}){}{:.2f}{}{:.2f}".format(
            s_time, split_str,
            e_time, cache_ratio, hit_rate, split_str,
            t_time, split_str,
            e2e_time
        ), end=end_str)


if __name__ == '__main__':
    arguments   = parse_args()
    quiver_dir = arguments['quiver_dir']
    if (mock):
        print('model\tdataset\tquiver')
    for model in model_list:
        for dataset in dataset_list:
            if (mock):
                print(f'{model}\t{dataset}\t', end='')
            else:
                print('\n&&{', end='')
            for system in system_list:
                parse_result_quiver(quiver_dir, model, dataset,
                                        split_in, split_end)
