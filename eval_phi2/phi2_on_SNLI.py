# Load model
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_default_device("cpu")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", 
                                             torch_dtype=torch.float32, # FP16
                                             flash_attn=True, # flash attention
                                             flash_rotary=True, # rotary embedding w/ flash_attn
                                             fused_dense=True, # operation fusion
                                             trust_remote_code=True)

# Evaluate
from datasets import load_dataset
import sys
sys.path.insert(0, '.')
from llm_common_eval import *

phi2_settings = {
    "model": model,
    "max_length": 128,
    "tokenizer": tokenizer,
    "inference_fn": inference_phi2_1batch,
    "debug": True
}

evaluate(phi2_settings, load_dataset("snli")['test'],
    ds_adapter=lambda j: (
        prompt_phi2_QA(question_NLI_0shot(j['hypothesis'], j['premise'])),
        str(j['label'])
    ),
    score_fn=output_contain_label
)
