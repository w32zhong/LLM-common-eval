# Copied from https://github.com/epfml/landmark-attention/blob/main/llama/run_test.py
import random


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 2000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def gen_data():
    num_tests = 50
    for n in [0, 100, 500, 1000, 5000, 8000, 10000, 12000, 14000, 18000, 20000, 25000, 38000]:
        for i in range(num_tests):
            prompt_text, pass_key = generate_prompt(n)
            yield {
                'n_garbage': n,
                'prompt': prompt_text,
                'pass_key': pass_key
            }


from datasets import Dataset, DatasetDict
dataset = DatasetDict({'test': Dataset.from_generator(gen_data)})
dataset.push_to_hub("w32zhong/PassKeyRetrieval")
