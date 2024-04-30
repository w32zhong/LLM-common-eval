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
        raise NotImplemented
    return prompt
