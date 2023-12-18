# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TextStreamer
from transformers import GenerationConfig

torch.set_default_device("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
    torch_dtype=torch.float16,
    flash_attn=True, # flash attention
    flash_rotary=True, # rotary embedding w/ flash_attn
    fused_dense=True, # operation fusion
    #device_map="cuda",
    trust_remote_code=True
)
gen_config = GenerationConfig.from_pretrained("microsoft/phi-2",
    do_sample=False,
    max_length=2048
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
    "stoplist": lce.KeywordsStopper.make_list(tokenizer, lce.common_stops),
    "streamer": None # TextStreamer(tokenizer)
}

ds = load_dataset("snli")
support_set = lce.generate_support_set(ds['train'], label, k_shots=3)

report = lce.evaluate(phi2_settings, ds['test'],
    data_adapter=lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLI_task.Qv1_fewshot(j['hypothesis'], j['premise'], support_set)
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

import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
