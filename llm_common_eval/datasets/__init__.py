import os
import sys
from datasets import Dataset, load_dataset
from typing import Optional

script_dir = os.path.dirname(os.path.realpath(__file__))


# extract mt-bench function to avoid load a bunch of dependencies there
def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                questions.append(json.loads(line))
    questions = questions[begin:end]
    return questions


def mt_bench():
    sys.path.insert(0, f'{script_dir}/mt_bench/fastchat/llm_judge')
    #from common import load_questions
    question_file = f'{script_dir}/mt_bench/fastchat/llm_judge/data/mt_bench/question.jsonl'
    questions = load_questions(question_file, None, None)
    def generator():
        for q in questions: yield q
    return Dataset.from_generator(generator)


def human_eval():
    sys.path.insert(0, f'{script_dir}/human_eval/human_eval')
    from data import write_jsonl, read_problems
    problems = read_problems()
    def generator():
        for q in problems.values(): yield q
    return Dataset.from_generator(generator)

def share_gpt():
    dataset_path = dict(
        path="Aeala/ShareGPT_Vicuna_unfiltered",
        data_files=["ShareGPT_V4.3_unfiltered_cleaned_split.json"],
        revision='8b0048ad6ae8c22f46a78c15559dec98feef5539'
    )
    return load_dataset(**dataset_path)['train']
