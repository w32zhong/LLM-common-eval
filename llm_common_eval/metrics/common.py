import re
import statistics
import itertools


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
