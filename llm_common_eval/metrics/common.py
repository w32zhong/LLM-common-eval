import re
import math
import statistics
import itertools
from collections import defaultdict

from rouge_score import rouge_scorer

from bert_score import get_model as bert_score_get_model
from bert_score import get_tokenizer as bert_score_get_tokenizer
from bert_score import bert_cos_score_idf
from bert_score.utils import model2layers as bert_score_model2layers


class MetricBase():
    def __init__(self, name):
        self.name = name
        self.samples = []
        self.n_trials = 1

    def add_json_sample(self, j):
        raise NotImplemented

    def report(self):
        raise NotImplemented


class TokenStats():
    def __init__(self, name):
        self.name = name
        self.samples = []
        self.n_trials = 1

    def add_json_sample(self, j):
        inp_tokens = j['input_tokens']
        out_tokens = [
            output["out_tokens"]
            for output in j['output_trials']
        ]
        time_costs = [
            float(output["time_cost"])
            for output in j['output_trials']
        ]
        self.samples.append((inp_tokens, out_tokens, time_costs))

    def avg_trials(self, trials, val_fn):
        return statistics.mean(val_fn(x) for x in trials)

    def stats(self, name, samples):
        return {
            f'avg_{name}': statistics.mean(samples),
            f'median_{name}': statistics.median(samples),
            f'max_{name}': max(samples),
            f'min_{name}': min(samples),
            f'sum_{name}': sum(samples)
        }

    def report(self):
        inp_tokens_stats = self.stats(
            'input_tokens',
            [len(s[0]) for s in self.samples]
        )
        out_tokens_stats = self.stats(
            'output_tokens',
            [self.avg_trials(s[1], len) for s in self.samples]
        )
        time_costs_stats = self.stats(
            'time_cost',
            [self.avg_trials(s[2], float) for s in self.samples]
        )
        time_cost_per_token = (time_costs_stats['sum_time_cost']
            / out_tokens_stats['sum_output_tokens'])
        return dict(name=self.name,
            **inp_tokens_stats,
            **out_tokens_stats,
            **time_costs_stats,
            time_cost_per_token=time_cost_per_token
        )


class Accuracy():
    def __init__(self, name, judge, *, n_trials=1, label_key='label'):
        self.name = name
        self.judge = judge
        self.n_trials = n_trials
        self.positives = 0
        self.label_key = label_key
        self.samples = []

    def add_json_sample(self, j):
        assert len(j['output_trials']) >= self.n_trials
        res_trials = [
            bool(self.judge(
                j['input'],
                output["out_text"],
                j[self.label_key]
            ))
            for output in j['output_trials'][:self.n_trials]
        ]
        self.samples.append(res_trials)
        self.vote(res_trials)

    def vote(self, res_trials):
        if res_trials[0]:
            self.positives += 1

    def report(self):
        percent = self.positives / len(self.samples)
        samples = len(self.samples)
        positives = self.positives
        return dict(name=self.name,
            percent=percent,
            samples=samples,
            positives=positives
        )


class AccuracyPassAnyK(Accuracy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, res_trials):
        if any(res_trials):
            self.positives += 1


class AccuracyMajorityInK(Accuracy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, res_trials):
        if sum(res_trials) > (len(res_trials) / 2):
            self.positives += 1


def if_output_contain_label(inp, out, label):
    return label in out


def if_output_contain_uncased(uncased_list, inp, out, label):
    for uncased in uncased_list:
        if re.search(uncased, out, re.IGNORECASE):
            return True
    return False


class Perplexity(MetricBase):
    def __init__(self, name, loss_key='loss'):
        super().__init__(name)
        self.loss_key = loss_key

    def calc_ppl(self, loss):
        # loss: (1/N) sum_i^N log p(label_i)
        # ref: https://huggingface.co/docs/transformers/perplexity
        return math.exp(loss)

    def add_json_sample(self, j):
        perplexities = [
            self.calc_ppl(output[self.loss_key])
            for output in j['output_trials'][:self.n_trials]
        ]
        mean_ppl = statistics.mean(perplexities)
        self.samples.append(mean_ppl)

    def report(self):
        samples = len(self.samples)
        mean_ppl = statistics.mean(self.samples)
        return dict(name=self.name,
            samples=samples,
            mean_ppl=mean_ppl
        )


class ROUGE(MetricBase):
    def __init__(self, name, pred_key='out_text', ref_key='label'):
        super().__init__(name)
        self.pred_key = pred_key
        self.ref_key = ref_key
        self.metrics = ['rouge1', 'rouge2', 'rougeL', 'rougeLsum']
        self.rouge = rouge_scorer.RougeScorer(self.metrics, use_stemmer=True)

    def calc_rouge(self, pred, ref):
        scores = self.rouge.score(pred, ref)
        #print(pred, ref, scores) ### DEBUG
        scores = dict([(k, v.fmeasure) for k, v in scores.items()])
        return scores

    def add_json_sample(self, j):
        scores = [
            self.calc_rouge(out[self.pred_key], j[self.ref_key])
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
