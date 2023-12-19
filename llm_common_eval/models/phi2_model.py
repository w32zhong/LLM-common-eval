def prompt_QA(question):
    return f'''Instruct: {question}\nOutput:'''


def hgf_inference_1batch(inp_data, model=None, tokenizer=None,
    streamer=None, stoplist=None, generation_cfg=None, debug=False):
    prompt = inp_data[0]
    inputs = tokenizer(prompt, return_tensors="pt",
        return_attention_mask=False)
    inp_tokens = inputs['input_ids'][0]
    inp_length = len(inp_tokens)
    res_tokens = model.generate(**inputs,
        generation_config=generation_cfg,
        streamer=streamer,
        stopping_criteria=stoplist
    )
    if debug: print(tokenizer.decode(res_tokens[0]))
    out_tokens = res_tokens[0][inp_length:]
    out_text = tokenizer.decode(out_tokens)
    return inp_tokens.tolist(), [
        dict(out_text=out_text, out_tokens=out_tokens.tolist())
    ]
