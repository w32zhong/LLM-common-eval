## About
This is a simple evaluation framework (WIP) that is meant to be easily extensible for common NLP evaluations/tasks.

The code is designed to be minimal but reuseble, it has a good abstraction and directory structure for contributor and researcher to work with.  

## Quick start
For example, to evaluate phi-2 model on SNLI:
```sh
pip install -r requirements.txt
python eval_phi2/phi2_on_SNLI_demo.py
```

Alternatively, on Google Colab or Kaggle:
```sh
! ls LLM-common-eval || git clone --depth 1 https://github.com/w32zhong/LLM-common-eval.git
%cd LLM-common-eval
! git pull
! pip install -r requirements.txt
%env PYTHONPATH=.
%env CUBLAS_WORKSPACE_CONFIG=:4096:8
```
and copy code from, e.g., `eval_phi2/phi2_on_SNLI_greedy_0shot.py` to Colab.

## Code example
Load a phi-2 model (run this once to save time! E.g., on Google Colab):
```py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
    torch_dtype=torch.float32, # only FP32 is supported for CPU
    flash_attn=True, # flash attention
    flash_rotary=True, # rotary embedding w/ flash_attn
    fused_dense=True, # operation fusion
    trust_remote_code=True
)
```

Create a phi-2 model setting to be evaluated:
```py
from transformers import GenerationConfig, TextStreamer
import llm_common_eval as lce

gen_config = GenerationConfig.from_pretrained("microsoft/phi-2",
    do_sample=True,
    max_new_tokens=12
)

phi2_settings = {
    "model": model,
    "tokenizer": tokenizer,
    "inference_fn": lce.models.common.hgf_inference_1batch,
    "generation_cfg": gen_config,
    "stopper": lce.KeywordsStopper(tokenizer, lce.common_stops),
    "streamer": TextStreamer(tokenizer) # set to None to be less verbose!
}
```

To evaluate this model setting on the SNLI dataset, 
```py
from datasets import load_dataset

report = lce.evaluate(phi2_settings, load_dataset("snli")['test'],
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
```

## Logging
Logging is necessary to skip already evaluated data rows.
It also saves compute when re-calculating evaluation metrics is necessary, e.g., if you've introduced a new metric but there is no need to re-generate answers.

To log to AWS S3 or S3-compatible bucket, create the `~/.aws/credentials` file.
For example:
```ini
[default]
aws_access_key_id=<your key>
aws_secret_access_key=<your secret>

[my_cloudflare_r2]
bucket=llm-common-eval
endpoint_url=https://foobarbaz.r2.cloudflarestorage.com
```

(Optionally) install aws-cli to test if your S3 bucket is working: 
```sh
python -m pip install awscli
aws s3 ls "s3://llm-common-eval/" --endpoint-url <endpoint>
# HANDY: to delete unwanted log files:
aws s3 rm "s3://llm-common-eval/ipykernel_launcher" --endpoint-url <endpoint> --recursive
```

If everything is working, pass `log_endpoint` to `evaluate()`:
```py
lce.evaluate(
    ...
    log_endpoint='my_cloudflare_r2'
)
```
The logger will fallback to local "./logs" directory if the config endpoint is not found.

## Useful evaluation options
In Colab environment, the script running is named `ipykernel_launcher.py` by default.
You have to pass the `run_name` to overwrite the script name which determines the log root directory.

In addition, to reduce S3 API usage, use the `skip_until=n` to completely skip the first `n` data rows in case you have already evaluated them!

Finally, pass in `slow_mode=True` to avoid any row to be skipped when an empty log file is presented.
This is useful to finish a final-round complete evaluation, otherwise some evaluation may be skipped for the sake of efficiency in parallel executions.

Example code:
```py
lce.evaluate(
    ...
    run_name='phi2_on_SNLI_greedy_0shot',
    skip_until=5000,
    slow_mode=True
)
```

## Extra packages
To install the newest AutoAWQ from PyPi, you need at least CUDA 12.1 installed.
```sh
conda create -n awq python=3.8
conda activate awq
conda install lxml
pip install --default-timeout=100 autoawq
pip install -r requirements.txt
```
