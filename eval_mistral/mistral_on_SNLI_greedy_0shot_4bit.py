# Load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
from transformers import BitsAndBytesConfig
hgf_repo = "mistralai/Mistral-7B-Instruct-v0.2"
model = AutoModelForCausalLM.from_pretrained(hgf_repo,
    device_map={'': 0},
    torch_dtype=torch.float16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    ),
    trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(hgf_repo, trust_remote_code=True)
genconfig = GenerationConfig.from_pretrained(hgf_repo)

# Set
import sys
sys.path.insert(0, '.')
import llm_common_eval as lce
genconfig.update(
    do_sample=False,
    max_length=2048
)
stop_list = lce.common_stops + lce.double_newline_stops
phi2_settings = {
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
report = lce.evaluate(phi2_settings, ds['test'],
    data_adapter=lambda j: {
        'input': lce.mistral_model.prompt_instruct_QA(
            lce.NLI_task.Qv1_0shot(j['hypothesis'], j['premise'])
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
