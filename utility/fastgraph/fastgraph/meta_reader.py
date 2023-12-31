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


class MetaReader(object):
    def __init__(self):
        pass

    def read(self, folder):
        meta = {'FEAT_DATA_TYPE' : 'F32'}
        with open(os.path.join(folder, 'meta.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split()
                assert len(line) == 2
                if line[0] == 'FEAT_DATA_TYPE':
                    meta[line[0]] = line[1]
                else:
                    meta[line[0]] = int(line[1])

        meta_keys = meta.keys()

        assert('NUM_NODE' in meta_keys)
        assert('NUM_EDGE' in meta_keys)
        assert('FEAT_DIM' in meta_keys)
        assert('NUM_CLASS' in meta_keys)
        assert('NUM_TRAIN_SET' in meta_keys)
        assert('NUM_VALID_SET' in meta_keys)
        assert('NUM_TEST_SET' in meta_keys)

        return meta
