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
import llm_common_eval as lce

phi2_settings = {
    "model": model,
    "max_length": 128,
    "tokenizer": tokenizer,
    "inference_fn": lce.phi2_model.hgf_inference_1batch,
    "debug": False
}
```

To evaluate this model setting on the SNLI dataset, 
```py
from datasets import load_dataset

simple_test_data = load_dataset("snli")['test'].select(range(3))

lce.evaluate(phi2_settings, simple_test_data,
    data_adapter=lambda j: {
        'input': lce.phi2_model.prompt_QA(
            lce.NLI_task.Qv1_0shot(j['hypothesis'], j['premise'])
        ),
        'label': str(j['label'])
    },
    metrics={
        'accuracy': lce.positive_if_output_contain_label
    }
)
```
