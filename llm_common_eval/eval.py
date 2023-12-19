import sys
import json
import time
import platform
from json.decoder import JSONDecodeError
from collections import defaultdict
from .utils import setup_endpoint, init_logging_prefix, filterout_dict_by_key


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


def do_inference_trials(model_setting, adapt_batch, n_trials, logger):
    input_batch = list(map(lambda x: x['input'], adapt_batch))
    label_batch = map(lambda x: x['label'], adapt_batch)
    out_trials_batch = [[] for _ in input_batch]
    inp_tokens = None
    for t in range(n_trials):
        print(f'[Inference trial#{t+1}]')
        args = model_setting.copy()
        args.pop('inference_fn')
        time_begin = time.time()
        inp_tokens, outputs = model_setting['inference_fn'](input_batch, **args)
        time_end = time.time()
        for b, out in enumerate(outputs):
            time_cost = time_end - time_begin
            out["time_cost"] = time_cost
            out_trials_batch[b].append(out)
    for b, (inp, outs, label) in enumerate(
        zip(input_batch, out_trials_batch, label_batch)):
        log = {
            "exe_node": platform.node(),
            "input": inp,
            "input_tokens": inp_tokens,
            "output_trials": outs,
            "label": label
        }
        logger(b, log)


def evaluate(model_setting, dataset, data_adapter, metrics,
    batch_size=1, collate_fn=collate_passthrough, skip_until=0,
    manual_seed=None, log_endpoint=None, run_name=None, slow_mode=False):
    # initialize
    n_trials = max([m.n_trials for m in metrics])
    set_seed(manual_seed)
    log_fs = setup_endpoint(log_endpoint)
    if run_name: # useful in Colab
        prefix = init_logging_prefix(log_fs, run_name)
    else:
        prefix = init_logging_prefix(log_fs, sys.argv[0])
    print('[listdir root]', log_fs.listdir('/'))
    # data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,
        collate_fn=collate_fn, batch_size=batch_size)
    N = len(dataset)
    # run through data
    for i, batch_data in enumerate(dataloader):
        row_base = i * batch_size
        adapt_batch = [data_adapter(x) for x in batch_data]
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
            do_inference_trials(model_setting, adapt_batch, n_trials, logger)
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
            display_log = filterout_dict_by_key(log, r"(.*)_tokens$")
            print('[Evaluating]', log_path, json.dumps(display_log, indent=2))
            for metric in metrics:
                metric.add_json_sample(log)
                print('[Running metric]', metric.report())
    return list(metric.report() for metric in metrics)
