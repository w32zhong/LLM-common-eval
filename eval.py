def collate_passthrough(batch_data):
    return batch_data


def evaluate(model_setting, dataset, ds_adapter, score_fn,
    batch_size=1, collate_fn=collate_passthrough):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)
    total_score = 0
    total_count = 0
    for i, batch_data in enumerate(dataloader):
        print(f'[Evaluating] {i} / {len(dataset)}')
        adapt_batch = [ds_adapter(x) for x in batch_data]
        inferrence_args = model_setting.copy()
        inferrence_args.pop('inference_fn')
        inp_batch = list(map(lambda x: x[0], adapt_batch))
        out_batch = model_setting['inference_fn'](inp_batch, **inferrence_args)
        label_batch = map(lambda x: x[1], adapt_batch)
        for inp, out, label in zip(inp_batch, out_batch, label_batch):
            total_score += float(score_fn(inp, out, label))
            total_count += 1.0
    return total_score / total_count
