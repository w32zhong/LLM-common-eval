# Arg parse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_version', type=str, default='3.0.0')
parser.add_argument('--skip_until', type=int, default=0)
args = parser.parse_args()

# Load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
hgf_repo = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(hgf_repo,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hgf_repo, trust_remote_code=True)
genconfig = GenerationConfig.from_pretrained(hgf_repo)

# Set
import sys
sys.path.insert(0, '.')
import llm_common_eval as lce
genconfig.update(
    do_sample=False,
    max_length=4096
)
stop_list = lce.common_stops + lce.newsect_stops + lce.code_stops
model_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.models.common.hgf_inference_1batch,
    "generation_cfg": genconfig,
    "stopper": lce.KeywordsStopper(tokenizer, stop_list),
    "streamer": None # TextStreamer(tokenizer)
}

# Evaluate
from functools import partial
from datasets import load_dataset
ds = load_dataset("cnn_dailymail", args.dataset_version)
report = lce.evaluate(model_settings, ds['test'],
    data_adapter=lambda j: {
        'input': lce.mistral_model.prompt_instruct_QA(
            lce.Summarization_task.Qv1_multi_sentences_0shot(j['article'])
        ),
        'label': lce.assert_and_return(j['highlights'], lambda x: isinstance(x, str)),
        '_example': lambda k: k['input'] + ' ' + k['label']
    },
    metrics=[
        lce.Perplexity('perplexity'),
        lce.ROUGE('ROUGE score'),
        lce.bert_score.BERTScore('BERT score', device='cuda'),
        lce.TokenStats('token stats')
    ],
    log_endpoint='non_exists', # will fallback to filesystem current directory.
    manual_seed=42, # not meaningful for greedy strategy here, but just keep it.
    run_name=None,
    skip_until=args.skip_until,
    slow_mode=True
)

# Report
import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
