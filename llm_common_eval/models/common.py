import torch


def hgf_inference_1batch(inp_text, exp_data, model=None, tokenizer=None,
    streamer=None, stopper=None, generation_cfg=None, debug=False,
    model_stats_getter=lambda m: None):
    prompt = inp_text[0]
    if prompt is None:
        out_dict = None
        inp_tokens = None
    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs.to(model.device)
        inp_tokens = inputs['input_ids'][0].tolist()
        inp_length = len(inp_tokens)
        with torch.no_grad():
            if exp_data[0] is not None:
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
        out_dict = dict(
            out_text=stopper.rm_stop(out_text),
            out_tokens=out_tokens.tolist(),
            _model_stats=model_stats_getter(model),
            loss=loss
        )
    return dict(
        input_tokens=[inp_tokens],
        outputs=[out_dict]
    )


def vllm_inference_1batch(inp_text, exp_data, vllm_model=None,
    sampling_params=None):
    prompt = inp_text[0]
    if prompt is None:
        out_dict = None
        inp_tokens = None
    else:
        output = vllm_model.generate(prompt, sampling_params, use_tqdm=False)[0]
        inp_tokens = output.prompt_token_ids
        out_text = output.outputs[0].text
        out_tokens = output.outputs[0].token_ids
        loss = None if exp_data[0] is None else output.outputs[0].cumulative_logprob
        out_dict = dict(
            out_text=out_text,
            out_tokens=out_tokens,
            loss=loss
        )
    return dict(
        input_tokens=[inp_tokens],
        outputs=[out_dict]
    )
