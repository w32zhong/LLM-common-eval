####################
# stop criteria list
####################
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

common_stops = ['<|endoftext|>', '<|im_end|>', '</s>', '<|end|>']
newline_stops = ['\n']
double_newline_stops = ['\n\n']
newsect_stops = ['##']
code_stops = ['"""', "'''"]

class KeywordsStopper(StoppingCriteria):
    def __init__(self, tokenizer, keywords):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.prompt_lengths = 0
        self.tokens = self.compile_stop_tokens()

    def is_stop(self, text):
        for kw in self.keywords:
            if kw in text:
                return True
        return False

    def compile_stop_tokens(self):
        tokens = []
        for kw in self.keywords:
            code = self.tokenizer.encode(kw)
            if len(code) == 1:
                tokens.append(code[0])
        return tokens

    def rm_stop(self, text):
        for kw in self.keywords:
            text = text.replace(kw, '')
        return text

    def __call__(self, input_ids, scores, **kwargs):
        batch_text = []
        for b, ids in enumerate(input_ids):
            ids = ids[self.prompt_lengths[b]:]
            text = self.tokenizer.decode(ids)
            batch_text.append(text)
        batch_stop = (self.is_stop(b) for b in batch_text)
        return all(batch_stop)

    def make_list(self, prompt_lengths):
        self.prompt_lengths = prompt_lengths
        return StoppingCriteriaList([self])


class KeywordsAndRepeatingStopper(KeywordsStopper):
    def __init__(self, tokenizer, keywords, accept_prob=0.001, max_span=1600):
        super().__init__(tokenizer, keywords)
        self.accept_prob = accept_prob
        self.max_span = max_span

    def __call__(self, input_ids, scores, **kwargs):
        batch_text = []
        for b, ids in enumerate(input_ids):
            ids = ids[self.prompt_lengths[b]:]
            text = self.tokenizer.decode(ids)
            batch_text.append(text)
        batch_stop = (self.is_stop(b) for b in batch_text)
        batch_repeat = (self.is_repeating(b) for b in batch_text)
        return all(batch_stop) or all(batch_repeat)

    @staticmethod
    def find_repeater(s):
        i = (s+s)[1:-1].find(s)
        if i == -1:
            return s
        else:
            return s[:i+1]

    def is_repeating(self, text):
        #with open('test/repeat.txt', 'r') as fh:
        #    text = fh.read()
        for k in range(min(len(text), self.max_span)):
            subtext = ''.join(text[-k:].strip().split())
            repeater = self.find_repeater(subtext)
            count = subtext.count(repeater)
            prob = 0.8 ** (len(repeater) * count)
            if count > 1 and prob < self.accept_prob:
                print('\n[repeats]', repeater, '*', count, 'prob=', prob)
                return True
        return False


####################
# git information
####################
import os
import subprocess


def get_git_revision():
    try:
        dirname = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=dirname
        ).decode('ascii').strip()
    except:
        return 'N/A'


def get_git_diff():
    try:
        dirname = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(
            ['git', 'diff', 'HEAD'], cwd=dirname
        ).decode('utf-8').strip()
    except:
        return 'N/A'


#####################
# logging filesystem
#####################
import configparser
from datetime import datetime
import re
import os
import sys
import fs.osfs


# reference interface class
class LogFS():
    def listdir(self, path):
        raise NotImplemented

    def exits(self, path):
        raise NotImplemented

    def makedir(self, path, recreate=True):
        raise NotImplemented

    def open(self, path, mode):
        raise NotImplemented


class LocalLogFS(fs.osfs.OSFS):
    def __init__(self, root_path):
        super().__init__(root_path=root_path)

    def makedir(self, path, recreate=True):
        return super().makedir(path, recreate=recreate)


try:
    import s3fs
    class RemoteS3LogFS(LogFS):
        def __init__(self, endpoint_url, bucket):
            self.fs = s3fs.S3FileSystem(
                client_kwargs=dict(
                    endpoint_url=endpoint_url
                )
            )
            self.bucket = bucket

        def listdir(self, path):
            return self.fs.ls(self.bucket + '/' + path)

        def makedir(self, path, recreate=True):
            return self.fs.mkdir(self.bucket + '/' + path, create_parents=recreate)

        def open(self, path, mode):
            return self.fs.open(self.bucket + '/' + path, mode=mode)

        def exists(self, path):
            return self.fs.exists(self.bucket + '/' + path)
except:
    pass

###################
# logging endpoint
###################
import json


def read_s3_credential(path='~/.aws/credentials'):
    cfg = configparser.ConfigParser(allow_no_value=True)
    cfg.read(os.path.expanduser(path))
    return cfg # return empty list if it does not exists


def setup_endpoint(endpoint):
    cfg = read_s3_credential()
    if endpoint in cfg:
        bucket = cfg[endpoint]['bucket']
        endpoint_url = cfg[endpoint]['endpoint_url']
        fs = RemoteS3LogFS(endpoint_url, bucket)
    elif endpoint == '/tmp':
        fs = LocalLogFS('/tmp')
    else:
        fs = LocalLogFS('./logs')
    return fs


def init_logging_prefix(log_fs, script_path):
    script_name = os.path.basename(script_path)
    script_name = re.sub(r'\.py$', '', script_name)
    log_fs.makedir(script_name)

    timestamp = f'{datetime.now():%Y-%m-%d_%H:%M:%S%z}'
    git_rev = get_git_revision()
    report_file = f'{script_name}/report-{timestamp}.run'
    with log_fs.open(report_file, 'w') as fh:
        json.dump({
            'timestamp': timestamp,
            'git_diff': get_git_diff(),
            'git_rev': git_rev
        }, fh, indent=2)
        fh.flush()

    return script_name, report_file


#####################
# nested dict filter
#####################
def filter_by_key(d, criteria_fn):
    if isinstance(d, dict):
        return {
            k: filter_by_key(v, criteria_fn)
            for k, v in d.items() if criteria_fn(k)
        }
    elif isinstance(d, list):
        return [filter_by_key(item, criteria_fn) for item in d]
    else:
        return d


#####################
# few-shot generator
#####################
def dataset_group_by_col(ds, col):
    groups = {key: [] for key in ds.unique(col)}
    ds.map(lambda key, i: groups[key].append(i),
        with_indices=True, input_columns=col)
    groups = {key: ds.select(indices) for key, indices in groups.items()}
    return groups


def generate_support_set(ds, col, k_shots=3):
    groups = dataset_group_by_col(ds, col)
    min_group_size = min(len(groups[k]) for k in groups.keys())
    assert min_group_size >= k_shots, "Please feed in more data!"
    iters = [
        iter(groups[uniq_label]) for uniq_label in sorted(ds.unique(col))
    ]
    support_set = []
    n_way = len(groups.keys())
    support_size = n_way * k_shots
    while True:
        for ds_iter in iters:
            support_set.append(next(ds_iter))
        if len(support_set) >= support_size:
            support_set = support_set[:support_size]
            break
    return support_set


#####################
# assert and return
#####################
def assert_and_return(x, assertion):
    assert assertion
    return x


#####################
# string processors
#####################
def remove_by_list_of_strings(x, alist):
    for string in alist:
        x = x.replace(string, '')
    return x


def truefalse_to_onezero(output):
    if re.search('true', output, re.IGNORECASE):
        return 1
    else:
        return 0

def extract_by_list_of_strings(x, alist):
    for string in alist:
        if string in x:
            return string
    return alist[0]


def string_span_wrapper(x, span, wrapper=('[[', ']]')):
    wrapp_word = x[span[0]:span[1]]
    if isinstance(wrapp_word, str):
        return x[:span[0]] + wrapper[0] + wrapp_word + wrapper[1] + x[span[1]:]
    else:
        return x[:span[0]] + [wrapper[0]] + wrapp_word + [wrapper[1]] + x[span[1]:]


def string_spans_wrapper(x, spans, wrapper=('[[', ']]')):
    accum = 0
    sorted_spans = sorted(spans)
    for span in sorted_spans:
        span = (span[0] + accum, span[1] + accum)
        x = string_span_wrapper(x, span, wrapper=wrapper)
        if isinstance(x, str):
            accum += len(wrapper[0]) + len(wrapper[1])
        else:
            accum += 2
    return x


#####################
# CUDA VRAM Monitor
#####################
def reset_vram_monitor():
    import torch
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()

def get_vram_peak():
    import torch
    vram = 0
    for i in range(torch.cuda.device_count()):
        vram += torch.cuda.max_memory_allocated(i)
        #vram += torch.cuda.max_memory_reserved(i)
    return vram / (1024 ** 3)
