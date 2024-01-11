# Arg parse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--shots', type=int, required=True)
args = parser.parse_args()
print(args)

# Load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
hgf_repo = "NousResearch/Llama-2-7b-chat-hf"
model = AutoModelForCausalLM.from_pretrained(hgf_repo,
    device_map="auto",
    #attn_implementation="flash_attention_2",
    load_in_4bit=True,
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
stop_list = lce.common_stops + lce.double_newline_stops
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
ds = load_dataset("snli")
ds = ds.filter(lambda j: j["label"] != -1)
support_set = lce.generate_support_set(
    ds['train'].select(range(8000)),
    'label',
    k_shots=args.shots
)
report = lce.evaluate(model_settings, ds['test'],
    data_adapter=lambda j: {
        'input': lce.llama2_model.prompt_QA(
            lce.NLI_task.Qv1_fewshot(
                j['hypothesis'], j['premise'],
                [(j['hypothesis'], j['premise'], j['label']) for j in support_set]
            )
        ),
        'label': str(j['label'])
    },
    metrics=[
        lce.AccuracyPassAnyK('accuracy', judge=lce.if_output_contain_label),
        lce.TokenStats('token stats')
    ],
    log_endpoint='non_exists!', # will fallback to filesystem current directory.
    manual_seed=42,
    run_name=None,
    skip_until=0,
    slow_mode=True
)

# Report
import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
