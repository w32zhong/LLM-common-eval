def prompt_QA(question):
    return f'''Instruct: {question}\nOutput:'''


def hgf_inference_1batch(inp_data, model=None, tokenizer=None,
    streamer=None, stoplist=None, generation_cfg=None, debug=False):
    prompt = inp_data[0]
    inp_tokens = tokenizer(prompt, return_tensors="pt",
        return_attention_mask=False)
    inp_length = len(inp_tokens['input_ids'][0])
    out_tokens = model.generate(**inp_tokens, generation_config=generation_cfg,
        streamer=streamer, stopping_criteria=stoplist)
    if debug: print(tokenizer.decode(out_tokens[0]))
    out_text = tokenizer.decode(out_tokens[0][inp_length:])
    return [out_text]
