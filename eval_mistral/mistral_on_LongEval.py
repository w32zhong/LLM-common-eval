# Arg parse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
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
    max_length=8192
)
stop_list = lce.common_stops + lce.newsect_stops + lce.code_stops
model_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.models.common.hgf_inference_1batch,
    "generation_cfg": genconfig,
    "stopper": lce.KeywordsStopper(tokenizer, stop_list),
    "streamer": TextStreamer(tokenizer) # set to None to be less verbose!
}

# Evaluate
from datasets import load_dataset
args.dataset in ['w32zhong/longeval_LineRetrieval', 'w32zhong/longeval_TopicRetrieval', 'w32zhong/PassKeyRetrieval']
ds = load_dataset(args.dataset)
data_adapters = {
    "w32zhong/longeval_LineRetrieval": lambda j: {
        'input': lce.mistral_model.prompt_instruct_QA(j['prompt']),
        'label': str(j['expected_number'])
    },
    "w32zhong/longeval_TopicRetrieval": lambda j: {
        'input': lce.mistral_model.prompt_instruct_QA(j['prompt']),
        'label': j['topics'][0]
    },
    "w32zhong/PassKeyRetrieval": lambda j: {
        'input': lce.mistral_model.prompt_instruct_QA(j['prompt']),
        'label': str(j['pass_key'])
    },
}
report = lce.evaluate(model_settings, ds['test'],
    data_adapter=data_adapters[args.dataset],
    metrics=[
        lce.AccuracyPassAnyK('accuracy', judge=lce.if_output_contain_label_uncased),
        lce.TokenStats('token stats')
    ],
    log_endpoint='non_exists!', # will fallback to filesystem current directory.
    manual_seed=42,
    run_name=f'mistral_on_{args.dataset}'.replace('/', '_'),
    skip_until=0,
    slow_mode=True
)

# Report
import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
