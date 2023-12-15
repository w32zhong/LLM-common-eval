def prompt_phi2_QA(question):
    return f'''Instruct: {question}\nOutput:'''

def inference_phi2_1batch(inp_data, model=None, tokenizer=None, max_length=2048, debug=False):
    prompt = inp_data[0]
    inp_tokens = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
    inp_length = len(inp_tokens['input_ids'][0])
    out_tokens = model.generate(**inp_tokens, max_length=max_length)
    if debug: print(tokenizer.decode(out_tokens[0]))
    out_text = tokenizer.decode(out_tokens[0][inp_length:])
    return [out_text]
