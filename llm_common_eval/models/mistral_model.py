def prompt_instruct_QA(Q, tokenizer=None):
    if tokenizer:
        messages = [
            {"role": "user", "content": Q},
            {"role": "assistant", "content": ""},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompt = prompt.replace('</s>', '')
    else:
        prompt = f'<s>[INST] {Q} [/INST]'
    return prompt
