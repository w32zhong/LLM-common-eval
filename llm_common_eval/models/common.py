import torch


def hgf_inference_1batch(inp_data, exp_data, model=None, tokenizer=None,
    streamer=None, stopper=None, generation_cfg=None, debug=False):
    prompt = inp_data[0]
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs.to(model.device)
    inp_tokens = inputs['input_ids'][0]
    inp_length = len(inp_tokens)
    with torch.no_grad():
        if exp_data is not None:
            prompt = exp_data[0]
            example_toks = tokenizer(prompt, return_tensors="pt")
            example_toks.to(model.device)
            example_inp_ids = example_toks['input_ids']
            example_trg_ids = example_inp_ids.clone()
            example_trg_ids[..., :inp_length] = -100
            loss = model(example_inp_ids, labels=example_trg_ids).loss
            loss = loss.item()
        else:
            loss = None

        stoplist = stopper.make_list(prompt_lengths=[inp_length])
        res_tokens = model.generate(**inputs,
            generation_config=generation_cfg,
            streamer=streamer,
            stopping_criteria=stoplist
        )
        if debug: print(tokenizer.decode(res_tokens[0]))
        out_tokens = res_tokens[0][inp_length:]
        out_text = tokenizer.decode(out_tokens)
        return dict(
            input_tokens=[inp_tokens.tolist()],
            outputs=[dict(
                out_text=stopper.rm_stop(out_text),
                out_tokens=out_tokens.tolist(),
                loss=loss
            )]
        )


def vllm_inference_1batch(inp_data, exp_data, vllm_model=None,
    sampling_params=None):
    breakpoint()
    pass
