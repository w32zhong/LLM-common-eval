import urllib.request
import json


def longeval_for_TopicRetrieval():
    data_url = 'https://raw.githubusercontent.com/DachengLi1/LongChat/longeval/longeval/evaluation/topics/testcases/{num_topics}_topics.jsonl'
    for num_topics in [5, 10, 15, 20, 25]:
        urllib.request.urlretrieve(data_url.format(num_topics=num_topics), 'tmp.jsonl')
        with open('tmp.jsonl', 'r') as json_file:
            json_list = list(json_file)
            for test_case in json_list:
                test_case = json.loads(test_case)
                yield {
                    "num_topics": num_topics,
                    'prompt': test_case['prompt'],
                    'topics': test_case['topics'],
                }


def longeval_for_LineRetrieval():
    data_url = 'https://raw.githubusercontent.com/DachengLi1/LongChat/longeval/longeval/evaluation/lines/testcases/{num_lines}_lines.jsonl'
    for num_lines in [200, 300, 400, 500, 600, 680]:
        urllib.request.urlretrieve(data_url.format(num_lines=num_lines), 'tmp.jsonl')
        with open('tmp.jsonl', 'r') as json_file:
            json_list = list(json_file)
            for test_case in json_list:
                test_case = json.loads(test_case)
                yield {
                    'num_lines': num_lines,
                    'prompt': test_case['prompt'],
                    'expected_number': test_case['expected_number'],
                }


from datasets import Dataset, DatasetDict
dataset = DatasetDict({'test': Dataset.from_generator(longeval_for_TopicRetrieval)})
dataset.push_to_hub("w32zhong/longeval_TopicRetrieval")

dataset = DatasetDict({'test': Dataset.from_generator(longeval_for_LineRetrieval)})
dataset.push_to_hub("w32zhong/longeval_LineRetrieval")
