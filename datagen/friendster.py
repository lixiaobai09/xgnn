"""
  Copyright 2022 Institute of Parallel and Distributed Systems, Shanghai Jiao Tong University

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
"""

import os

import numpy as np
import pickle, json, torch

DOWNLOAD_URL = 'https://pub-383410a98aef4cb686f0c7601eddd25f.r2.dev/graphalytics/com-friendster.tar.zst'

RAW_DATA_DIR = '/graph-learning/data-raw'
CF_RAW_DATA_DIR = f'{RAW_DATA_DIR}/com-friendster'

GNNLAB_OUTPUT_DATA_DIR = '/graph-learning/samgraph/com-friendster'

def download_data():
    print('Download data...')
    if not os.path.exists(f'{RAW_DATA_DIR}/com-friendster.tar.zst'):
        print('Start downloading...')
        assert(os.system(f'wget {DOWNLOAD_URL} -O {RAW_DATA_DIR}/com-friendster.tar.zst') == 0)
    else:
        print('Already downloaded.')

    print('Unzip data...')
    if not os.path.exists(f'{CF_RAW_DATA_DIR}/unzipped'):
        print('Start unziping...')
        assert(os.system(f'cd {RAW_DATA_DIR}; tar --use-compress-program=zstd -xf {RAW_DATA_DIR}/com-friendster.tar.zst -C {CF_RAW_DATA_DIR}') == 0)
        assert(os.system(f'touch {CF_RAW_DATA_DIR}/unzipped') == 0)
    else:
        print('Already unzipped...')

def write_split_feat_label_gnnlab():
    os.system(f'cp {CF_RAW_DATA_DIR}/train_set.bin {GNNLAB_OUTPUT_DATA_DIR}/')
    os.system(f'cp {CF_RAW_DATA_DIR}/valid_set.bin {GNNLAB_OUTPUT_DATA_DIR}/')
    os.system(f'cp {CF_RAW_DATA_DIR}/test_set.bin {GNNLAB_OUTPUT_DATA_DIR}/')

    # file0 = np.load(f'{CF_RAW_DATA_DIR}/raw/data.npz')
    # file1 = np.load(f'{CF_RAW_DATA_DIR}/raw/node-label.npz')

    # features = file0['node_feat']
    # label = file1['node_label']

    # features.astype('float32').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/feat.bin')
    # label.astype('uint64').tofile(f'{GNNLAB_OUTPUT_DATA_DIR}/label.bin')

def gen_undir_graph_cpu():
    os.system('g++ -std=c++11 -pthread -fopenmp -O2 comfriendster_csr_generator.cc -o comfriendster_csr_generator.out')
    os.system('./comfriendster_csr_generator.out')
    os.system(f'cp {CF_RAW_DATA_DIR}/indptr.bin {GNNLAB_OUTPUT_DATA_DIR}/indptr.bin')
    os.system(f'mv {CF_RAW_DATA_DIR}/indices.bin {GNNLAB_OUTPUT_DATA_DIR}/indices.bin')

def write_gnnlab_meta():
    print('Writing meta file...')
    with open(f'{GNNLAB_OUTPUT_DATA_DIR}/meta.txt', 'w') as f:
        f.write('{}\t{}\n'.format('NUM_NODE', 65608366))
        f.write('{}\t{}\n'.format('NUM_EDGE', 3612134270))
        f.write('{}\t{}\n'.format('FEAT_DIM', 140))
        f.write('{}\t{}\n'.format('NUM_CLASS', 100))
        f.write('{}\t{}\n'.format('NUM_TRAIN_SET', 1000000))
        f.write('{}\t{}\n'.format('NUM_VALID_SET', 200000))
        f.write('{}\t{}\n'.format('NUM_TEST_SET', 100000))

if __name__ == '__main__':
    assert(os.system(f'mkdir -p {CF_RAW_DATA_DIR}') == 0)
    assert(os.system(f'mkdir -p {GNNLAB_OUTPUT_DATA_DIR}') == 0)

    download_data()
    gen_undir_graph_cpu()
    write_split_feat_label_gnnlab()
    write_gnnlab_meta()
