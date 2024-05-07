import os
import json


def log_files(inp_logdir):
    for file in os.listdir(inp_logdir):
        file_path = os.path.join(inp_logdir, file)
        with open(file_path, 'r') as fh:
            try:
                json_log = json.load(fh)
                test = json_log['output_trials']
            except (KeyError, json.decoder.JSONDecodeError) as e:
                print('[skip KeyError]', file)
                continue
            yield file, json_log


def covert(inp_logdir, jsonl_name, out_dir='./data/mt_bench/model_answer'):
    out_jsonl = os.path.join(out_dir, jsonl_name)
    with open(out_jsonl, 'w') as fh:
        for file, json_log in log_files(inp_logdir):
            turns = list(map(lambda j: j['out_text'], json_log['output_trials']))
            json_out = {
                "question_id": json_log['question_id'],
                "choices": [
                    {"turns": turns}
                ]
            }
            #print('[convert]', file)
            fh.write(json.dumps(json_out))
            fh.write('\n')


if __name__ == '__main__':
    import fire
    fire.Fire(covert)
