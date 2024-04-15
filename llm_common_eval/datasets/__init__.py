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
