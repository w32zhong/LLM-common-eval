####################
# stop criteria list
####################
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList

common_stops = ['<|endoftext|>', '<|im_end|>', '</s>']
newline_stops = ['\n']
double_newline_stops = ['\n\n']
newsect_stops = ['##']

class KeywordsStopper(StoppingCriteria):
    def __init__(self, tokenizer, keywords):
        self.keywords = keywords
        self.tokenizer = tokenizer

    def is_stop(self, text):
        for kw in self.keywords:
            if kw in text:
                return True
        return False

    def __call__(self, input_ids, scores, **kwargs):
        batch_text = self.tokenizer.batch_decode(input_ids)
        batch_stop = (self.is_stop(b) for b in batch_text)
        return all(batch_stop)

    @staticmethod
    def make_list(tokenizer, keywords):
        return StoppingCriteriaList([
            KeywordsStopper(tokenizer, keywords)
        ])


####################
# git information
####################
import os
import subprocess


def get_git_revision():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=dirname
    ).decode('ascii').strip()


def get_git_diff():
    dirname = os.path.dirname(os.path.abspath(__file__))
    return subprocess.check_output(
        ['git', 'diff', 'HEAD'], cwd=dirname
    ).decode('utf-8').strip()


#####################
# logging filesystem
#####################
import configparser
import fs.osfs
import s3fs
from datetime import datetime
import re
import os
import sys


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


###################
# logging endpoint
###################
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
    else:
        print('Warnining: Cloud not find endpoint, use local logging.')
        fs = LocalLogFS('./logs')
    return fs


def init_logging_prefix(log_fs, script_path):
    script_name = os.path.basename(script_path)
    script_name = re.sub(r'\.py$', '', script_name)
    log_fs.makedir(script_name)

    timestamp = f'{datetime.now():%Y-%m-%d_%H:%M:%S%z}'
    git_rev = get_git_revision()
    with log_fs.open(f'{script_name}/{timestamp}_{git_rev}.run', 'w') as fh:
        fh.write(get_git_diff())
        fh.flush()

    return script_name


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
    return x[:span[0]] + wrapper[0] + wrapp_word + wrapper[1] + x[span[1]:]


def string_spans_wrapper(x, spans, wrapper=('[[', ']]')):
    #raise NotImplemented
    return x
