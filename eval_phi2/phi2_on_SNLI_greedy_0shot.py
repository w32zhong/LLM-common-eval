# Load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
hgf_repo = "microsoft/phi-2"
torch.set_default_device("cpu")
model = AutoModelForCausalLM.from_pretrained(hgf_repo,
    torch_dtype=torch.float32,
    flash_attn=True, # flash attention
    flash_rotary=True, # rotary embedding w/ flash_attn
    fused_dense=True, # operation fusion
    #device_map="cuda",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(hgf_repo, trust_remote_code=True)
genconfig = GenerationConfig.from_pretrained(hgf_repo)

# Set
import sys
sys.path.insert(0, '.')
import llm_common_eval as lce
genconfig.update(
    do_sample=False,
    max_new_tokens=128
)
stop_list = lce.common_stops + lce.double_newline_stops
phi2_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.models.common.hgf_inference_1batch,
    "generation_cfg": genconfig,
    "stopper": lce.KeywordsStopper(tokenizer, stop_list),
    "streamer": None
}

# Evaluate
from datasets import load_dataset
ds = load_dataset("snli")
ds = ds.filter(lambda j: j["label"] != -1)
report = lce.evaluate(phi2_settings, ds['test'],
    data_adapter=lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLI_task.Qv1_0shot(j['hypothesis'], j['premise'])
        ),
        'label': str(j['label'])
    },
    metrics=[
        lce.AccuracyPassAnyK('accuracy', judge=lce.if_output_contain_label, n_trials=1),
        lce.TokenStats('token stats')
    ],
    log_endpoint='my_cloudflare_r2', # will fallback to filesystem current directory.
    manual_seed=42, # no need for greedy generation, but keep here for demonstration.
    run_name=None,
    skip_until=0,
    slow_mode=False
)

# Report
import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
