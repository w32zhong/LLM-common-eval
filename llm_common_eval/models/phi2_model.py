def prompt_QA(question):
    return f'''Instruct: {question}\nOutput:'''


def hgf_inference_1batch(inp_data, model=None, tokenizer=None,
    streamer=None, stoplist=None, generation_cfg=None, debug=False):
    prompt = inp_data[0]
    inputs = tokenizer(prompt, return_tensors="pt",
        return_attention_mask=False)
    inputs.to(model.device)
    inp_tokens = inputs['input_ids'][0]
    inp_length = len(inp_tokens)
    res_tokens = model.generate(**inputs,
        generation_config=generation_cfg,
        streamer=streamer,
        stopping_criteria=stoplist([inp_length])
    )
    if debug: print(tokenizer.decode(res_tokens[0]))
    out_tokens = res_tokens[0][inp_length:]
    out_text = tokenizer.decode(out_tokens)
    return dict(
        input_tokens=[inp_tokens.tolist()],
        outputs=[dict(out_text=out_text, out_tokens=out_tokens.tolist())]
    )
