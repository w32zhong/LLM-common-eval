# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
from transformers import GenerationConfig

torch.set_default_device("cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
    torch_dtype=torch.float32, # FP16
    flash_attn=True, # flash attention
    flash_rotary=True, # rotary embedding w/ flash_attn
    fused_dense=True, # operation fusion
    trust_remote_code=True
)
gen_config = GenerationConfig.from_pretrained("microsoft/phi-2",
    do_sample=True,
    max_new_tokens=12
)

# Evaluate
from datasets import load_dataset
import sys
sys.path.insert(0, '.')
import llm_common_eval as lce

phi2_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.phi2_model.hgf_inference_1batch,
    "generation_cfg": gen_config,
    "stoplist": lce.KeywordsStopper.make_list(tokenizer,
        lce.common_stops + lce.double_newline_stops),
    "streamer": TextStreamer(tokenizer)
}

#report = lce.evaluate(phi2_settings, load_dataset("snli")['test'],
report = lce.evaluate(phi2_settings, load_dataset("snli")['test'].select(range(5)),
    data_adapter=lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLI_task.Qv1_0shot(j['hypothesis'], j['premise'])
        ),
        'label': str(j['label'])
    },
    metrics=[
        lce.AccuracyPassAnyK('pass@3', judge=lce.if_output_contain_label, n_trials=3),
        lce.AccuracyMajorityInK('maj@3', judge=lce.if_output_contain_label, n_trials=3)
    ],
    log_endpoint='my_cloudflare_r2', # will fallback to filesystem current directory.
    manual_seed=42
)

import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
