import re
import sys
import json
import time
import platform
from json.decoder import JSONDecodeError
from collections import defaultdict
from .utils import setup_endpoint, init_logging_prefix, filter_by_key


def set_seed(seed):
    if seed is None: return
    import torch
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
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def collate_passthrough(batch_data):
    return batch_data


def do_inference_trials(model_setting, adapt_data, n_trials, logger):
    # input texts
    inp_data = list(map(lambda x: x['input'], adapt_data))
    # input example texts
    if '_example' in adapt_data[0]:
        list(map(lambda x: x.update({'example': x['_example'](x)}), adapt_data))
        example_data = list(map(lambda x: x['example'], adapt_data))
    else:
        example_data = None
    # other customer
    custom_data = filter_by_key(adapt_data,
        lambda k: k not in ['input', '_output_process', '_example'])
    output_trials = [[] for _ in inp_data]
    for t in range(n_trials):
        print(f'[Inference trial#{t+1}]')
        args = model_setting.copy()
        args.pop('inference_fn')
        time_begin = time.time()
        result = model_setting['inference_fn'](inp_data, example_data, **args)
        time_end = time.time()
        for b, out in enumerate(result['outputs']):
            time_cost = time_end - time_begin
            out["time_cost"] = time_cost
            if '_output_process' in adapt_data[0]:
                processed = adapt_data[0]['_output_process'](out)
            else:
                processed = {}
            output_trials[b].append({**out, **processed})
    for b, (inp, inp_tokens, outs, custom) in enumerate(zip(
        inp_data, result['input_tokens'], output_trials, custom_data)):
        log = {
            "exe_node": platform.node(),
            "input": inp,
            "input_tokens": inp_tokens,
            "output_trials": outs,
            **custom
        }
        logger(b, log)


def data_generator(dataset, **kwargs):
    from torch.utils.data import DataLoader
    def genn():
        if dataset is None: # interactive
            while True:
                prompt = input('Input: ')
                yield [dict(prompt=prompt)]
        else:
            for data in DataLoader(dataset, **kwargs):
                yield data
    return genn


def evaluate(model_setting, dataset, data_adapter, metrics,
    batch_size=1, collate_fn=collate_passthrough, skip_until=0,
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
        adapt_data = [data_adapter(x) for x in batch_data]
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
        fresh_logs = {}
        def logger(batch, log):
            row = row_base + batch
            log_path = f'{prefix}/{n_trials}trial-row{row}'
            fresh_logs[log_path] = log
            with log_fs.open(f'{log_path}.json', 'w') as fh:
                json.dump(log, fh, indent=2)
                fh.flush()
        if skip_infer_this_batch:
            print(f'[Inference skipped] {log_path} + {batch_size} / {N}')
        else:
            print(f'[Inference] {log_path} + {batch_size} / {N}')
            do_inference_trials(model_setting, adapt_data, n_trials, logger)
        # evaluate
        for row in range(row_base, row_base + batch_size):
            log_path = f'{prefix}/{n_trials}trial-row{row}'
            if log_path in fresh_logs:
                # use fresh logs to avoid reading back from files
                log = fresh_logs[log_path]
            else:
                # read back logs from json files
                with log_fs.open(f'{log_path}.json', 'r') as fh:
                    try:
                        log = json.load(fh)
                    except JSONDecodeError:
                        # Multiple evaluation scripts are running?
                        print('[Empty log] Be sure to re-run evaluation!')
                        break
            display_log = filter_by_key(log,
                lambda k: not re.match(r"(.*)_tokens$", k))
            print('[Evaluating]', log_path, json.dumps(display_log, indent=2))
            for metric in metrics:
                if not metric.add_json_sample(log) is False:
                    print('[Running metric]', metric.report())
    report = dict([(metric.name, metric.report()) for metric in metrics])
    # done
    with log_fs.open(report_file, 'r') as fh:
        run_json = json.load(fh)
    with log_fs.open(report_file, 'w') as fh:
        json.dump({**run_json, 'report': report}, fh, indent=2)
        fh.flush()
    return report
