# Load
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig
from transformers import TextStreamer
hgf_repo = "microsoft/phi-2"
model = AutoModelForCausalLM.from_pretrained(hgf_repo,
    torch_dtype=torch.float32,
    flash_attn=True, # flash attention
    flash_rotary=True, # rotary embedding w/ flash_attn
    fused_dense=True, # operation fusion
    #device_map="auto",
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
    max_length=2048
)
phi2_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.phi2_model.hgf_inference_1batch,
    "generation_cfg": genconfig,
    "stoplist": lce.KeywordsStopper.make_list(tokenizer, lce.common_stops),
    "streamer": TextStreamer(tokenizer) # set to None to be less verbose!
}

# Evaluate
from functools import partial
from datasets import load_dataset
SuperGLUE_list = 'boolq cb copa multirc record rte wic wsc'
SuperGLUE_select = 'record'
assert SuperGLUE_select in SuperGLUE_list.split()
ds = load_dataset("super_glue", SuperGLUE_select)
SuperGLUE_default_adapter=lambda j: {
}
SuperGLUE_multirc_adapter=lambda j: {
    'input': lce.phi2_model.prompt_QA(
        lce.NLU_task.Qv1_MultiRC_0shot(j['paragraph'], j['question'], j['answer'])
    ),
    'label': lce.assert_and_return(j['label'], lambda x: x in [0, 1]),
    'idx': j['idx'],
    '_output_process': (lambda o: {
        'prediction': lce.NLU_task.Qv1_MultiRC_output_process(o['out_text'])
    })
}
SuperGLUE_record_adapter=lambda j: {
    'input': lce.phi2_model.prompt_QA(
        lce.NLU_task.Qv1_ReCoRD_0shot(j['passage'], j['query'], j['entities'])
    ),
    'label': lce.assert_and_return(j['answers'], lambda x: isinstance(x, list)),
    'idx': j['idx'],
    '_output_process': (lambda o: {
        'prediction_text': lce.NLU_task.Qv1_ReCoRD_output_process(
            lce.utils.remove_by_list_of_strings(o['out_text'], lce.common_stops)
        )
    })
}
SuperGLUE_adapters = {
    "multirc": SuperGLUE_multirc_adapter,
    "record": SuperGLUE_record_adapter,
}
SuperGLUE_metrics = {
    'multirc': [
        lce.metrics.super_glue.MultiRC_metrics('MultiRC metrics'),
        lce.Accuracy('valid output',
            judge=partial(lce.if_output_contain_uncased, ['true', 'false'])),
        lce.TokenStats('token stats')
    ],
    'record': [
        lce.metrics.super_glue.ReCoRD_metrics('ReCoRD metrics'),
        lce.Accuracy('valid output',
            judge=partial(lce.if_output_contain_uncased, ['@placeholder'])),
        lce.TokenStats('token stats')
    ]
}
report = lce.evaluate(phi2_settings, ds['validation'].select(range(5)),
    data_adapter=SuperGLUE_adapters[SuperGLUE_select],
    metrics=SuperGLUE_metrics[SuperGLUE_select],
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
