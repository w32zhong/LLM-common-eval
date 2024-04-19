import json
with open('./data/mt_bench/model_answer/foo.jsonl', 'w') as fh:
    for i in range(81, 161):
        fh.write(json.dumps({
            "question_id": i,
            "choices": [
                {"turns": ["hey!", "hello!"]}
            ]
        }))
        fh.write('\n')
