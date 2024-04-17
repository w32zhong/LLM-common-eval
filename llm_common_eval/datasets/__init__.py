import os
import sys
from datasets import Dataset

script_dir = os.path.dirname(os.path.realpath(__file__))


def mt_bench():
    sys.path.insert(0, f'{script_dir}/mt_bench/fastchat/llm_judge')
    from common import load_questions
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
