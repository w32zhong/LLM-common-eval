import sklearn
from datasets import load_dataset, load_metric
SuperGLUE_list = 'boolq cb copa multirc record rte wic wsc'
SuperGLUE_data, SuperGLUE_metric = {}, {}
for split in SuperGLUE_list.split():
    SuperGLUE_data[split] = load_dataset("super_glue", split)
    SuperGLUE_metric[split] = load_metric("super_glue", split)

    print(split)
    print(SuperGLUE_data[split]['test'])
    if split == 'multirc':
        predictions = [
            dict(idx={"paragraph": 0, "question": 0, "answer": 0}, prediction="1"),
            dict(idx={"paragraph": 0, "question": 0, "answer": 1}, prediction="0"),
        ]
        references = [0, 1]
    elif split == 'record':
        predictions = [
            dict(idx={"passage": 1, "query": 1}, prediction_text="Hugo"),
            dict(idx={"passage": 1, "query": 2}, prediction_text="Simon Bolivar"),
        ]
        references = [
            dict(idx={"passage": 1, "query": 1}, answers=["Hugo Chavez"]),
            dict(idx={"passage": 1, "query": 2}, answers=["Bolivar", "Simon Bolivar"]),
        ]
    else:
        predictions = [0, 1]
        references = [0, 1]
    metric = SuperGLUE_metric[split]
    results = metric.compute(predictions=predictions, references=references)
    print(results)
