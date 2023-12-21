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
SuperGLUE_select = 'copa' # change this!
assert SuperGLUE_select in SuperGLUE_list.split()
ds = load_dataset("super_glue", SuperGLUE_select)
SuperGLUE_adapters = {
    "boolq": lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLU_task.Qv1_BoolQ_0shot(j['passage'], j['question'])
        ),
        'label': lce.assert_and_return(j['label'], lambda x: x in [0, 1]),
        '_output_process': (lambda o: {
            'prediction': lce.utils.truefalse_to_onezero(o['out_text'])
        })
    },
    "cb": lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLI_task.Qv1_0shot(j['hypothesis'], j['premise'], label_ids={
                    'entailment': 0,
                    'neutral': 2,
                    'contradiction': 1
                }
            )
        ),
        'label': lce.assert_and_return(j['label'], lambda x: x in [0, 1, 2]),
        '_output_process': (lambda o: {
            'prediction': int(lce.utils.extract_by_list_of_strings(o['out_text'], ['1', '2', '3']))
        })
    },
    "copa": lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.Reasoning_task.Qv1_COPA_0shot(j['premise'], (j['choice1'], j['choice2']), j['question'])
        ),
        'label': lce.assert_and_return(j['label'], lambda x: x in [0, 1]),
        '_output_process': (lambda o: {
            'prediction': int(lce.utils.extract_by_list_of_strings(o['out_text'], ['1', '2'])) - 1
        })
    },
    "multirc": lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLU_task.Qv1_MultiRC_0shot(j['paragraph'], j['question'], j['answer'])
        ),
        'label': lce.assert_and_return(j['label'], lambda x: x in [0, 1]),
        'idx': j['idx'],
        '_output_process': (lambda o: {
            'prediction': lce.utils.truefalse_to_onezero(o['out_text'])
        })
    },
    "record": lambda j: {
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
    },
}
SuperGLUE_metrics = {
    'boolq': [
        lce.super_glue.Default_metrics('BoolQ metrics', SuperGLUE_select),
        lce.Accuracy('valid output',
            judge=partial(lce.if_output_contain_uncased, ['true', 'false'])),
        lce.TokenStats('token stats')
    ],
    'cb': [
        lce.super_glue.Default_metrics('CommitmentBank metrics', SuperGLUE_select),
        lce.Accuracy('valid output',
            judge=partial(lce.if_output_contain_uncased, ['1', '2', '3'])),
        lce.TokenStats('token stats')
    ],
    'copa': [
        lce.super_glue.Default_metrics('COPA metrics', SuperGLUE_select),
        lce.Accuracy('valid output',
            judge=partial(lce.if_output_contain_uncased, ['1', '2'])),
        lce.TokenStats('token stats')
    ],
    'multirc': [
        lce.super_glue.MultiRC_metrics('MultiRC metrics'),
        lce.Accuracy('valid output',
            judge=partial(lce.if_output_contain_uncased, ['true', 'false'])),
        lce.TokenStats('token stats')
    ],
    'record': [
        lce.super_glue.ReCoRD_metrics('ReCoRD metrics'),
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
