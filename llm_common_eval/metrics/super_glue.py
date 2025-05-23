from .common import MetricBase

try:
    from datasets import load_metric
    # For super_glue metrics implementation, refer to:
    # https://github.com/huggingface/datasets/tree/main/metrics/super_glue
except:
    pass


class Default_metrics(MetricBase):
    def __init__(self, name, dataset, *, out_key='prediction', label_key='label'):
        super().__init__(name)
        self.out_key = out_key
        self.label_key = label_key
        self.predictions = []
        self.references = []
        self.metric = load_metric("super_glue", dataset)

    def add_json_sample(self, j):
        prediction = j['output_trials'][0][self.out_key]
        reference = j[self.label_key]

        self.samples.append(j)
        self.predictions.append(prediction)
        self.references.append(reference)

    def report(self):
        results = self.metric.compute(
            predictions=self.predictions,
            references=self.references
        )
        return results


class MultiRC_metrics(MetricBase):
    def __init__(self, name, *, idx_key='idx', out_key='prediction', label_key='label'):
        super().__init__(name)
        self.idx_key = idx_key
        self.out_key = out_key
        self.label_key = label_key
        self.predictions = []
        self.references = []
        self.metric = load_metric("super_glue", 'multirc')

    def add_json_sample(self, j):
        prediction = dict(
            idx=j[self.idx_key],
            prediction=j['output_trials'][0][self.out_key]
        )
        reference = j[self.label_key]

        self.samples.append(j)
        self.predictions.append(prediction)
        self.references.append(reference)

    def report(self):
        results = self.metric.compute(
            predictions=self.predictions,
            references=self.references
        )
        return results


class ReCoRD_metrics(MetricBase):
    def __init__(self, name, *, idx_key='idx', out_key='prediction_text', label_key='label'):
        super().__init__(name)
        self.idx_key = idx_key
        self.out_key = out_key
        self.label_key = label_key
        self.predictions = []
        self.references = []
        self.metric = load_metric("super_glue", 'record')

    def add_json_sample(self, j):
        prediction = dict(
            idx=j[self.idx_key],
            prediction_text=j['output_trials'][0][self.out_key]
        )
        reference = dict(
            idx=j[self.idx_key],
            answers=j[self.label_key]
        )

        self.samples.append(j)
        self.predictions.append(prediction)
        self.references.append(reference)

    def report(self):
        results = self.metric.compute(
            predictions=self.predictions,
            references=self.references
        )
        return results
