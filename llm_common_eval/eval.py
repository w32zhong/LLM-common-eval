from collections import defaultdict


def collate_passthrough(batch_data):
    return batch_data


def evaluate(model_setting, dataset, data_adapter, metrics,
    batch_size=1, collate_fn=collate_passthrough):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)
    running_metrics = defaultdict(float)
    count = 0
    for i, batch_data in enumerate(dataloader):
        print(f'[Evaluating] {i * batch_size} / {len(dataset)}')
        adapt_batch = [data_adapter(x) for x in batch_data]
        inferrence_args = model_setting.copy()
        inferrence_args.pop('inference_fn')
        inp_batch = list(map(lambda x: x['input'], adapt_batch))
        out_batch = model_setting['inference_fn'](inp_batch, **inferrence_args)
        label_batch = map(lambda x: x['label'], adapt_batch)
        for metric, metric_fn in metrics.items():
            for inp, out, label in zip(inp_batch, out_batch, label_batch):
                judge = float(metric_fn(inp, out, label))
                running_metrics[metric] += judge
                print('[Input]', inp)
                print('[Output]', out)
                print(f'[Judge] label: {label}, {metric}: {judge}')
                count += 1.0
            val = running_metrics[metric]
            print(f'[Metrics] {metric}: {val} / {count} = {val / count:.3f}%')
    return running_metrics
