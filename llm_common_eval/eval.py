import re
import sys
import json
import time
import torch
import platform
from json.decoder import JSONDecodeError
from collections import defaultdict
from colorama import Fore, Style
from .utils import setup_endpoint, init_logging_prefix, filter_by_key
from .utils import reset_vram_monitor, get_vram_peak


def set_seed(seed):
    if seed is None: return
    import random
    import numpy as np
    from transformers import set_seed
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # comment per https://discuss.pytorch.org/t/torch-deterministic-algorithms-error/125200/5
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def collate_passthrough(batch_data):
    return batch_data


def do_inference(model_setting, batch_data, data_adapter, n_trials, multi_turn):
    reset_vram_monitor()
    vram_base, vram_peak = get_vram_peak(), None
    trials_and_turns = [[] for _ in batch_data]
    custom_data = filter_by_key([data_adapter(x) for x in batch_data],
        lambda k: k not in ['input', '_output_process', '_example'])
    for trial in range(n_trials):
        turn = 0
        keep_going = True
        while keep_going: # multi-turns
            print(f'[Inference trial#{trial+1}, turn#{turn+1}]')
            # input texts
            adapt_data = [
                data_adapter(x, hist=trials_and_turns[b])
                if len(trials_and_turns[b]) > 0 else
                data_adapter(x) # for back compatibility
                for b, x in enumerate(batch_data)
            ]
            # input batched texts
            inp_text = list(map(lambda x: x['input'], adapt_data))
            # input example texts
            exp_data = []
            for x in adapt_data:
                if x is not None and '_example' in x:
                    exp_data.append(x['_example'](x))
                else:
                    exp_data.append(None)
            # do inference
            args = model_setting.copy()
            args.pop('inference_fn')
            print(f'{Fore.BLUE}{inp_text}{Style.RESET_ALL}')
            keep_going = False
            time_begin = time.time()
            try:
                result = model_setting['inference_fn'](inp_text, exp_data, **args)
            except torch.cuda.OutOfMemoryError as e:
                print(f'{Fore.RED}[Out of VRAMM]{Style.RESET_ALL}', e)
                result = dict(input_tokens=[None], outputs=[None])
                vram_peak = -1
            time_end = time.time()
            # add timecost and post-processing
            for b, out_dict in enumerate(result['outputs']):
                if out_dict is None: continue
                time_cost = time_end - time_begin
                out_dict["time_cost"] = time_cost
                if '_output_process' in adapt_data[0]:
                    out_dict = adapt_data[0]['_output_process'](out_dict)
                trials_and_turns[b].append({
                    'trial': trial,
                    'turn': turn,
                    'input': inp_text[b],
                    'input_tokens': result['input_tokens'][b],
                    **out_dict
                })
                keep_going = True
            if not multi_turn: keep_going = False
            turn += 1
    vram_peak = get_vram_peak() if vram_peak is None else float('inf')
    return trials_and_turns, custom_data, vram_base, vram_peak


def data_generator(dataset, **kwargs):
    import os
    from torch.utils.data import DataLoader
    def genn():
        if dataset is None: # interactive
            while True:
                print("[process fd] /proc/{0}/fd/0".format(os.getpid()))
                prompt = input('Input: ')
                yield [dict(prompt=prompt)]
        elif isinstance(dataset, str): # manual input
            yield [dict(prompt=dataset)]
        else:
            for data in DataLoader(dataset, **kwargs):
                yield data
    return genn


def evaluate(model_setting, dataset, data_adapter, metrics, batch_size=1,
    collate_fn=collate_passthrough, skip_until=0, multi_turn=False,
    manual_seed=None, log_endpoint=None, run_name=None, slow_mode=False):
    # initialize
    n_trials = max([m.n_trials for m in metrics])
    set_seed(manual_seed)
    log_fs = setup_endpoint(log_endpoint)
    if run_name: # useful in Colab
        prefix, report_file = init_logging_prefix(log_fs, run_name)
    else:
        prefix, report_file = init_logging_prefix(log_fs, sys.argv[0])
    print('[listdir root]', log_fs.listdir('/'))
    # data loader
    N = len(dataset) if dataset is not None else -1
    data_gen = data_generator(dataset,
        collate_fn=collate_fn, batch_size=batch_size)
    # run through data
    for i, batch_data in enumerate(data_gen()):
        row_base = i * batch_size
        # test whether to skip inference by looking at the log file
        log_path = f'{prefix}/{n_trials}trial-row{row_base}'
        skip_infer_this_batch = False
        if row_base < skip_until:
            continue
        elif log_fs.exists(f'{log_path}.json'):
            if slow_mode:
                with log_fs.open(f'{log_path}.json', 'r') as fh:
                    try:
                        log = json.load(fh)
                    except JSONDecodeError:
                        pass # need to re-generate log file in slow mode
                    else:
                        skip_infer_this_batch = True
            else:
                skip_infer_this_batch = True
        else:
            # touch a file to flag the work is being done
            with log_fs.open(f'{log_path}.json', 'w') as fh:
                fh.flush()
        # perform inference only if the previous step is not skipped
        fresh_log = {}
        if skip_infer_this_batch:
            print(f'[Inference skipped] {log_path} + {batch_size} / {N}')
        else:
            print(f'[Inference] {log_path} + {batch_size} / {N}')
            trials_and_turns, custom_data, vram_base, vram_peak = do_inference(
                model_setting, batch_data, data_adapter, n_trials, multi_turn
            )
            for b, (hist, cd) in enumerate(zip(trials_and_turns, custom_data)):
                row = row_base + b
                log_path = f'{prefix}/{n_trials}trial-row{row}'
                log = {
                    "exe_node": platform.node(),
                    "vram_base": vram_base,
                    "vram_peak": vram_peak,
                    "input": hist[0]['input'] if hist else None,
                    "input_tokens": hist[0]['input_tokens'] if hist else None,
                    "output_trials": hist,
                    **cd
                }
                with log_fs.open(f'{log_path}.json', 'w') as fh:
                    json.dump(log, fh, indent=2)
                    fh.flush()
                fresh_log[log_path] = log
        # evaluate
        for row in range(row_base, row_base + batch_size):
            log_path = f'{prefix}/{n_trials}trial-row{row}'
            if log_path in fresh_log:
                # use fresh logs to avoid reading back from files
                log = fresh_log[log_path]
            else:
                # read back logs from json files
                with log_fs.open(f'{log_path}.json', 'r') as fh:
                    try:
                        log = json.load(fh)
                    except JSONDecodeError:
                        # Multiple evaluation scripts are running?
                        print('[Empty log] Be sure to re-run evaluation!')
                        break
            # skip printing input_tokens and out_tokens
            print_log = filter_by_key(log,
                lambda k: k not in ['input_tokens', 'out_tokens'])
            # skip printing anything starts with an underscore
            print_log = filter_by_key(print_log,
                lambda k: not re.match(r"^_.*", k))
            print('[Evaluating]', log_path, json.dumps(print_log, indent=2))
            if len(log['output_trials']) > 0:
                for metric in metrics:
                    if not (metric.add_json_sample(log) is False):
                        print('[Running metric]', metric.report())
    report = dict([(metric.name, metric.report()) for metric in metrics])
    # done
    with log_fs.open(report_file, 'r') as fh:
        run_json = json.load(fh)
    with log_fs.open(report_file, 'w') as fh:
        json.dump({**run_json, 'report': report}, fh, indent=2)
        fh.flush()
    return report
