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
import llm_common_eval as lce

phi2_settings = {
    "model": model,
    "max_length": 128,
    "tokenizer": tokenizer,
    "inference_fn": lce.phi2_model.hgf_inference_1batch,
    "debug": False
}

lce.evaluate(phi2_settings, load_dataset("snli")['test'],
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
