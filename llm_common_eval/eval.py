import json
from json.decoder import JSONDecodeError
from collections import defaultdict
from .utils import setup_endpoint, init_logging_prefix


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


def evaluate(model_setting, dataset, data_adapter, metrics,
    batch_size=1, collate_fn=collate_passthrough, skip_until=0,
    manual_seed=None, log_endpoint=None):
    # initialize
    n_trials = max([m.n_trials for m in metrics])
    set_seed(manual_seed)
    log_fs = setup_endpoint(log_endpoint)
    prefix = init_logging_prefix(log_fs)
    print('[listdir]', log_fs.listdir('/'))
    # data loader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset,
        collate_fn=collate_fn, batch_size=batch_size)
    N = len(dataset)
    # run through data
    for i, batch_data in enumerate(dataloader):
        row_base = i * batch_size
        # test whether to skip by looking at the log file
        log_path = f'{prefix}/{n_trials}trial-row{row_base}'
        skip_infer_this_batch = False
        if row_base < skip_until:
            continue
        elif log_fs.exists(f'{log_path}.json'):
            print(f'[Skipping] {log_path} + {batch_size} / {N}')
            skip_infer_this_batch = True
        else:
            # touch a file to flag the work is being done
            with log_fs.open(f'{log_path}.json', 'w') as fh:
                fh.flush()
        # perform inference trials
        adapt_batch = [data_adapter(x) for x in batch_data]
        input_batch = list(map(lambda x: x['input'], adapt_batch))
        label_batch = map(lambda x: x['label'], adapt_batch)
        out_trials_batch = [[] for _ in input_batch]
        # only if the previous step is not skipped
        if not skip_infer_this_batch:
            for t in range(n_trials):
                print(f'[Trial#{t+1}] {log_path} + {batch_size} / {N}')
                args = model_setting.copy()
                args.pop('inference_fn')
                outputs = model_setting['inference_fn'](input_batch, **args)
                for b, output in enumerate(outputs):
                    out_trials_batch[b].append(output)
            for b, (inp, outs, label) in enumerate(
                zip(input_batch, out_trials_batch, label_batch)):
                row = row_base + b
                log_path = f'{prefix}/{n_trials}trial-row{row}'
                log = {"input": inp, "output_trials": outs, "label": label}
                with log_fs.open(f'{log_path}.json', 'w') as fh:
                    json.dump(log, fh, indent=2)
                    fh.flush()
        # read back json and evaluate
        for row in range(row_base, row_base + batch_size):
            log_path = f'{prefix}/{n_trials}trial-row{row}'
            with log_fs.open(f'{log_path}.json', 'r') as fh:
                try:
                    log = json.load(fh)
                except JSONDecodeError:
                    # Multiple evaluation scripts are running?
                    print('[Empty log] Be sure to re-run evaluation!')
                    break
                print('[Log]', log_path, json.dumps(log, indent=2))
            for metric in metrics:
                metric.add_json_sample(log)
                print('[Running metric]', metric.report())
    return list(metric.report() for metric in metrics)
