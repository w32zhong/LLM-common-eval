# Load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
hgf_repo = "microsoft/phi-2"
torch.set_default_device("cpu")
model = AutoModelForCausalLM.from_pretrained(hgf_repo,
    torch_dtype=torch.float32, # only FP32 is supported for CPU
    flash_attn=True, # flash attention
    flash_rotary=True, # rotary embedding w/ flash_attn
    fused_dense=True, # operation fusion
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(hgf_repo, trust_remote_code=True)
genconfig = GenerationConfig.from_pretrained(hgf_repo)

# Set
import sys
sys.path.insert(0, '.')
import llm_common_eval as lce
genconfig.update(
    do_sample=True,
    max_new_tokens=12
)
phi2_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.phi2_model.hgf_inference_1batch,
    "generation_cfg": genconfig,
    "stoplist": lce.KeywordsStopper.make_list(tokenizer,
        lce.common_stops + lce.double_newline_stops),
    "streamer": TextStreamer(tokenizer) # set to None to be less verbose!
}

# Evaluate
from datasets import load_dataset
ds = load_dataset("snli")
ds = ds.filter(lambda j: j["label"] != -1)
report = lce.evaluate(phi2_settings, ds['test'].select(range(5)), # demo a few examples
    data_adapter=lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLI_task.Qv1_0shot(j['hypothesis'], j['premise'])
        ),
        'label': str(j['label'])
    },
    metrics=[
        lce.AccuracyPassAnyK('pass@3', judge=lce.if_output_contain_label, n_trials=3),
        lce.AccuracyMajorityInK('maj@3', judge=lce.if_output_contain_label, n_trials=3),
        lce.TokenStats('token stats')
    ],
    log_endpoint='non_exists!', # will fallback to filesystem current directory.
    manual_seed=42,
    run_name="debug",
    skip_until=0,
    slow_mode=True
)

# Report
import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
