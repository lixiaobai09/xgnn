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
    print("model\tdataset\tmethod\tsample_time\ttrain_time")
    opt_methods = ["BASE", "+MSG", "+MSF", "+OGA", "+PSA"]
    for dataset in dataset_list:
        prefix_str = f"xgnn_graphsage_{dataset}_"
        if (dataset in ["tw", "pa"]):
            prefix_str = f"xgnn_gcn_{dataset}_"
        for i in range(1, 6):
            file_name = directory + "/" + prefix_str + str(i) + ".log"
            # print("file_name: ", file_name)
            if (dataset in ["tw", "pa"]):
                print(f"gcn\t{dataset}\t{opt_methods[i-1]}", end="")
            else:
                print(f"graphsage\t{dataset}\t{opt_methods[i-1]}", end="")
            ret = parse_result_sgnn(
                    file_name,
                    ["sample_no_mark", "mark_cache_copy_time", "train_total"])
            print("\t{:.2f}\t{:.2f}".format(ret[0], (ret[1] + ret[2])))
        print('0')
