# Arg parse
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()

# Load
hgf_repo = "microsoft/phi-2"
from vllm import LLM, SamplingParams
vllm_model = LLM(model=hgf_repo, trust_remote_code=True, dtype='float16', gpu_memory_utilization=0.9)

# Set
import sys
sys.path.insert(0, '.')
import llm_common_eval as lce
model_settings = {
    "vllm_model": vllm_model,
    "sampling_params": SamplingParams(
        temperature=0.0,
        best_of=1,
        top_p=1,
        top_k=-1,
        use_beam_search=False,
        max_tokens=10,
        presence_penalty=0,
        frequency_penalty=0,
    ),
    "inference_fn": lce.models.common.vllm_inference_1batch,
}

# Evaluate
from datasets import load_dataset
ds = load_dataset("snli")
ds = ds.filter(lambda j: j["label"] != -1)
report = lce.evaluate(model_settings, ds['test'],
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
    log_endpoint='non_exists', # will fallback to filesystem current directory.
    manual_seed=42, # no need for greedy generation, but keep here for demonstration.
    run_name=None,
    skip_until=0,
    slow_mode=True
)

# Report
import json
print('=' * 20, 'Report', '=' * 20)
print(json.dumps(report, indent=2))
