#log = {"input": inp, "output_trials": outs, "label": label}


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
            bool(self.judge(j['input'], output, j['label']))
            for output in j['output_trials'][:self.n_trials]
        ]
        self.samples.append(res_trials)
        return self.vote(res_trials)

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
