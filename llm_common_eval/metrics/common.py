import statistics
import itertools


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
            f'avg {name}': statistics.mean(samples),
            f'median {name}': statistics.median(samples),
            f'max {name}': max(samples),
            f'min {name}': min(samples),
            f'sum {name}': sum(samples)
        }

    def report(self):
        inp_tokens_stats = self.stats(
            'input tokens',
            [len(s[0]) for s in self.samples]
        )
        out_tokens_stats = self.stats(
            'output tokens',
            [self.avg_trials(s[1], len) for s in self.samples]
        )
        time_costs_stats = self.stats(
            'time cost',
            [self.avg_trials(s[2], float) for s in self.samples]
        )
        time_cost_per_token = (time_costs_stats['sum time cost']
            / out_tokens_stats['sum output tokens'])
        return dict(name=self.name,
            **inp_tokens_stats,
            **out_tokens_stats,
            **time_costs_stats,
            time_cost_per_token=time_cost_per_token
        )


class Accuracy():
    def __init__(self, name, judge, n_trials=1):
        self.name = name
        self.judge = judge
        self.n_trials = n_trials
        self.positives = 0
        self.samples = []

    def add_json_sample(self, j):
        assert len(j['output_trials']) >= self.n_trials
        res_trials = [
            bool(self.judge(j['input'], output["out_text"], j['label']))
            for output in j['output_trials'][:self.n_trials]
        ]
        self.samples.append(res_trials)
        self.vote(res_trials)

    def vote(self, res_trials):
        raise NotImplemented

    def report(self):
        value = self.positives / float(len(self.samples))
        return dict(name=self.name, value=value)


class AccuracyPassAnyK(Accuracy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, res_trials):
        if any(res_trials):
            self.positives += 1.0


class AccuracyMajorityInK(Accuracy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def vote(self, res_trials):
        if sum(res_trials) > (len(res_trials) / 2):
            self.positives += 1.0


def if_output_contain_label(inp, out, label):
    return label in out
