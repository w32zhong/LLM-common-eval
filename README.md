## About
This is a simple evaluation framework (WIP) that is meant to be easily extensible for common NLP evaluations/tasks.

The code is designed to be minimal but reuseble, it has a good abstraction and directory structure for contributor and researcher to work with.  

## Quick start
For example, to evaluate phi-2 model on SNLI:
```sh
python eval_phi2/phi2_on_SNLI.py
```

## Code example
Load a phi-2 model (run this once to save time! E.g., on Google Colab):
```py
# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_default_device("cuda")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                             torch_dtype="auto", # FP16
                                             flash_attn=True, # flash attention
                                             flash_rotary=True, # rotary embedding w/ flash_attn
                                             fused_dense=True, # operation fusion
                                             device_map="cuda",
                                             trust_remote_code=True)

```

Create a phi-2 model setting to be evaluated:
```py
import sys
sys.path.insert(0, '.')
from llm_common_eval import inference_phi2_1batch

phi2_settings = {
    "model": model,
    "max_length": 512,
    "tokenizer": tokenizer,
    "inference_fn": inference_phi2_1batch,
    "debug": True
}
```

To evaluate this model setting on the SNLI dataset, 
```py
# Evaluate
from datasets import load_dataset
from llm_common_eval import *

evaluate(phi2_settings, load_dataset("snli")['test'],
    ds_adapter=lambda j: (
        prompt_phi2_QA(question_NLI_0shot(j['hypothesis'], j['premise'])),
        str(j['label'])
    ),
    score_fn=output_contain_label
)
```
