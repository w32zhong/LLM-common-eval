import re


def prompt_QA(Q, tokenizer=None):
    if tokenizer:
        messages = [
            {"role": "user", "content": Q}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = prompt.replace('<s>', '')
        prompt = prompt.replace('</s>', '')
    else:
        prompt = f'[INST] {Q} [/INST]'
    return prompt


def prompt_fewshots_QA(Q, support_set, tokenizer=None):
    if tokenizer:
        messages = []
        for example_Q, example_A in support_set:
            messages.append(
                {"role": "user", "content": example_Q}
            )
            messages.append(
                {"role": "assistant", "content": example_A}
            )
        messages.append(
            {"role": "user", "content": Q}
        )
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = prompt.replace('<s>', '')
        prompt = prompt.replace('</s>', '')
    else:
        prompt = ''
        for example_Q, example_A in support_set:
            prompt += f'[INST] {Q} [/INST] '
            prompt += f'{example_A} '
        prompt += f'[INST] {Q} [/INST]'
    return prompt


def prompt_multi_turn_conv(tokenizer, j, hist=None):
    turns = j['turns']
    prev_response = [] if hist is None else [h['out_text'] for h in hist]
    if len(turns) <= len(prev_response):
        prompt = None
    else:
        if tokenizer is not None:
            messages = [
                {"role": "user", "content": turns[0]}
            ]
            for i, response in enumerate(prev_response):
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": turns[1 + i]})
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt = re.sub(f'^{re.escape(tokenizer.bos_token)}', '', prompt)
            prompt = re.sub(f'{re.escape(tokenizer.eos_token)}$', '', prompt)
        else:
            prompt = f'[INST] {turns[0]} [/INST] '
            for i, response in enumerate(prev_response):
                prompt += f'{response} '
                prompt += f'[INST] {turns[1 + i]} [/INST] '
    return dict(input=prompt)
