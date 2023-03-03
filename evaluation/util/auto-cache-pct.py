import subprocess
import os, sys, re
import pathlib
import dataclasses
from typing import List
import logging
logging.basicConfig(level=logging.DEBUG)

class Cmd:
    def __init__(self, cmd) -> None:
        if '--cache-percentage' in cmd:
            match = re.search(r'--cache-percentage ([0-9](\.[0-9]+)?)', cmd)
            self.ori_cache_pct = float(match.group(1))
            self.cmd_tp = re.sub(r'(--cache-percentage) [0-9](\.[0-9]+)?', r'\1 {} ', cmd)
        else:
            self.cmd_tp = f'{cmd} --cache-percentage {{}}'
            self.ori_cache_pct = None
    
    def get(self, cache_pct):
        return self.cmd_tp.format(cache_pct)
    
    def is_part_cache(self):
        return '--part-cache' in self.cmd_tp

    @property
    def dataset(self) -> str: 
        match = re.search(r'--dataset ([0-9a-zA-Z\-]+)', self.cmd_tp)
        return match.group(1)
    
    @property
    def num_worker(self) -> int:
        mathc = re.search(r'--num-worker ([0-9])', self.cmd_tp)
        return int(mathc.group(1))


def extract_cmds(script_path):
    with open(script_path, 'r') as F:
        script = F.read()
        script = script.replace('$(dirname $0)', str(pathlib.Path(script_path).absolute().parent))
        script = re.sub(r'set -x', '', script)
        script = re.sub(r'(^ *)(python)', r'\1echo \2', script, flags=re.MULTILINE)
        script = re.sub(r'2?> ?.*\.(log|err)', '', script)
        script = re.sub(r'(^.*mkdir.*$)', '', script, flags=re.MULTILINE)
    proc = subprocess.run(args=script, shell=True, executable='/bin/bash', capture_output=True)
    cmds = proc.stdout.decode().strip('\n').split('\n')
    return [Cmd(cmd) for cmd in cmds if 'sgnn' in cmd]


def find_cache_pct_impl(cmd:Cmd):
    @dataclasses.dataclass
    class GraphInfo:
        graph_sz:float = None
        feat_sz:float = None
    graphs = {
        'twitter' : GraphInfo(5.63, 39.72 ),
        'papers100M' : GraphInfo(6.43, 52.96 ),
        'uk-2006-05' : GraphInfo(11.34, 74.14 ),
        'com-friendster' : GraphInfo(13.70, 34.22 ),
        'products' : GraphInfo(0.47, 0.91 )
    }    

    class UnablePredCacheErr(Exception): ...
    def _get_next_cache_pct(cache_pct, gpu_mem):
        full_mem = 15.8
        if gpu_mem >= full_mem:
            return cache_pct
        if cmd.is_part_cache() and cmd.num_worker in {2, 4, 8}:
            avail_mem = (full_mem - gpu_mem) * min(4, cmd.num_worker)
            return min(1, avail_mem / graphs[cmd.dataset].feat_sz + cache_pct)
        elif cmd.is_part_cache() and cmd.num_worker == 1 or not cmd.is_part_cache():
            avail_mem = full_mem - gpu_mem
            return min(1, avail_mem / graphs[cmd.dataset].feat_sz + cache_pct)
        else:
            raise UnablePredCacheErr("cannot predict next cache pct")

    def _try_cache_pct(cmd:str):
        proc = subprocess.run(args=cmd.split(), env=os.environ, capture_output=True)
        output = proc.stdout.decode()
        if 'epoch_time:total' in output:
            gpu_memorys = re.findall(r'\| GPU memory ([0-9]+\.[0-9]+)', output)
            gpu_memory = max([float(m) for m in gpu_memorys])
            err = None
        else:
            gpu_memory = None
            errout = proc.stderr.decode()
            try:
                err = re.search(r'\[.* E .*\].*', errout, flags=re.MULTILINE).group(0)
            except:
                err = None
            if err is None:
                try:
                    err = re.search(r'.*out of memory.*', errout, flags=re.MULTILINE).group(0)
                except:
                    err = errout
        return proc, gpu_memory, err

    logging.info(f'\x1b[32;1m[TRY CMD]\x1b[0m: {cmd.get(cmd.ori_cache_pct)}')

    left = 0
    right = 101 if cmd.is_part_cache() else 31
    pred_cache_pct = None
    while left + 1 < right:
        if pred_cache_pct is None:
            cache_pct_int = (left + right) // 2
        else:
            cache_pct_int = pred_cache_pct
        cache_pct = cache_pct_int / 100
        try_cmd = cmd.get(cache_pct)
        proc, gpu_memory, err = _try_cache_pct(try_cmd)
        msg = f'error: \n{err}' if err is not None else f'gpu memory usage {gpu_memory}'
        logging.info(f'\x1b[35;1m[TRY RESULT]\x1b[0m: cache_pct {cache_pct_int}% in [{left}, {right}), {msg}')
        if not err:
            left = cache_pct_int
            try:
                nxt_cache_pct = _get_next_cache_pct(cache_pct, gpu_memory)
                pred_cache_pct = int(100 * nxt_cache_pct)
                pred_cache_pct = min(right - 1, max(left, pred_cache_pct))
                nxt_bin_mid = (left + right) // 2
                logging.debug(f'pred_next_cache_pct {pred_cache_pct} binary_search_next_cache_pct {nxt_bin_mid}')
                if pred_cache_pct <= nxt_bin_mid:
                    pred_cache_pct = None
            except UnablePredCacheErr as e:
                pred_cache_pct = None
        else:
            right = cache_pct_int
            pred_cache_pct = None

    # check result
    cache_pct = left
    chk_cnter = 0
    fail_cnter = 0
    while True:
        proc, gpu_memory, err = _try_cache_pct(cmd.get(cache_pct / 100))
        logging.info(f'\x1b[38;5;208;1m[CHECK RESULT {chk_cnter}]\x1b[0m: cache_pct {cache_pct} gpu_memory {gpu_memory} error {err}')
        if not err and (gpu_memory >= 15 or cache_pct == 100):
            return cache_pct / 100, gpu_memory
        elif err and cache_pct == 0:
            return cache_pct / 100, gpu_memory
        elif not err and (fail_cnter > 0):
            return cache_pct / 100, gpu_memory
        if err:
            cache_pct -= 1
            fail_cnter += 1
        else:
            cache_pct += 1
        chk_cnter += 1


def find_cache_pct(cmds:List[Cmd]):
    cache_pcts = []
    gpu_mems = []
    def _output_cache_pct():
        sys.stdout.flush()
        print(30*'=')
        pct_arr = ""
        search_res = [f'[CMD] cache-percentage memory-usage\n']
        for i, (cmd, pct, mem) in enumerate(zip(cmds, cache_pcts, gpu_mems)):
            print(f'[{i:3}] {cmd.get(cmd.ori_cache_pct)}')
            search_res.append(f'[{i:3}] {pct:16} {mem:12}\n')
            if (i + 1) % 10 == 0:
                pct_arr += f'{pct:4}\n'
            elif i % 10 == 0:
                pct_arr += f'{pct:8} '
            else:
                pct_arr += f'{pct:4} '
        print(f'\n{"".join(search_res)}\n')
        print(f'cache_pcts=(\n{pct_arr}\n)')

    for cmd in cmds:
        cache_pct, gpu_mem = find_cache_pct_impl(cmd)
        cache_pcts.append(cache_pct)
        gpu_mems.append(gpu_mem)
    _output_cache_pct()



if __name__ == '__main__':
    script = sys.argv[1]
    cmds = extract_cmds(script)
    find_cache_pct(cmds)
    
