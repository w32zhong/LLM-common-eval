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


def covert(inp_logdir, jsonl_name, out_dir='./human_eval/data'):
    out_jsonl = os.path.join(out_dir, jsonl_name)
    with open(out_jsonl, 'w') as fh:
        for file, json_log in log_files(inp_logdir):
            completion =  json_log['output_trials'][0]['out_text']
            json_out = {
                'task_id': json_log['problem']['task_id'],
                'completion': completion
            }
            fh.write(json.dumps(json_out))
            fh.write('\n')


if __name__ == '__main__':
    import fire
    fire.Fire(covert)
