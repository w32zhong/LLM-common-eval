import statistics
from collections import defaultdict
from .common import MetricBase

try:
    from bert_score import get_model as bert_score_get_model
    from bert_score import get_tokenizer as bert_score_get_tokenizer
    from bert_score import bert_cos_score_idf
    from bert_score.utils import model2layers as bert_score_model2layers
except:
    pass


# ref: https://github.com/Tiiiger/bert_score/blob/dbcf6db37e8bd6ff68446f06b0ba5d0763b62d20/bert_score/score.py#L21
class BERTScore(MetricBase):
    def __init__(self, name, lang='en', device='cpu',
        pred_key='out_text', ref_key='label'):
        super().__init__(name)
        self.pred_key = pred_key
        self.ref_key = ref_key
        self.metrics = ['precision', 'recall', 'f1']
        lang2model = defaultdict(lambda: "bert-base-multilingual-cased")
        lang2model.update({
            "en": "roberta-large",
            "zh": "bert-base-chinese",
            "tr": "dbmdz/bert-base-turkish-cased",
            "en-sci": "allenai/scibert_scivocab_uncased",
        })
        model_type = lang2model[lang]
        layers = bert_score_model2layers[model_type]
        self.model = bert_score_get_model(model_type, layers, False)
        self.model.to(device)
        self.tokenizer = bert_score_get_tokenizer(model_type, False)
        self.idf_dict = defaultdict(lambda: 1.0)
        self.device = device

    def calc_score(self, cand, ref):
        precision, recall, f1 = bert_cos_score_idf(self.model,
            [ref], [cand], self.tokenizer, self.idf_dict,
            verbose=False, device=self.device,
            batch_size=1, all_layers=False).tolist()[0]
        return dict(precision=precision, recall=recall, f1=f1)

    def add_json_sample(self, j):
        scores = [
            self.calc_score(out[self.pred_key], j[self.ref_key])
            for out in j['output_trials'][:self.n_trials]
        ]
        for score in scores:
            self.samples.append(score)

    def report(self):
        samples = len(self.samples)
        mean_scores = dict([
            (metric, statistics.mean([x[metric] for x in self.samples]))
            for metric in self.metrics
        ])
        return dict(name=self.name,
            samples=samples,
            mean_scores=mean_scores
        )
