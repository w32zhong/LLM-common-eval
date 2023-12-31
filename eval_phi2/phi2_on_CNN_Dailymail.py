def main(
    log_endpoint='non_exists!',
    slow_mode=False,
    skip_until=0,
    devices="0",
    runname='CNN_Daily',
    dataset_version='3.0.0'):

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    devices = str(devices) if isinstance(devices, int) else ','.join(map(str, devices))
    os.environ["CUDA_VISIBLE_DEVICES"] = devices

    # Load
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers import GenerationConfig
    from transformers import TextStreamer
    hgf_repo = "microsoft/phi-2"
    model = AutoModelForCausalLM.from_pretrained(hgf_repo,
        torch_dtype=torch.float16,
        flash_attn=True, # flash attention
        flash_rotary=True, # rotary embedding w/ flash_attn
        fused_dense=True, # operation fusion
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(hgf_repo, trust_remote_code=True)
    genconfig = GenerationConfig.from_pretrained(hgf_repo)

    # Set
    import sys
    sys.path.insert(0, '.')
    import llm_common_eval as lce
    genconfig.update(
        do_sample=False,
        max_length=2048
    )
    stop_list = lce.common_stops + lce.newsect_stops + lce.code_stops
    phi2_settings = {
        "model": model,
        "tokenizer": tokenizer,
        "inference_fn": lce.models.common.hgf_inference_1batch,
        "generation_cfg": genconfig,
        "stopper": lce.KeywordsStopper(tokenizer, stop_list),
        "streamer": None # TextStreamer(tokenizer)
    }

    # Evaluate
    from functools import partial
    from datasets import load_dataset
    ds = load_dataset("cnn_dailymail", dataset_version)
    report = lce.evaluate(phi2_settings, ds['test'],
        data_adapter=lambda j: {
            'input': lce.phi2_model.prompt_QA(
                lce.Summarization_task.Qv1_multi_sentences_0shot(j['article'])
            ),
            'label': lce.assert_and_return(j['highlights'], lambda x: isinstance(x, str)),
            '_example': lambda k: k['input'] + ' ' + k['label']
        },
        metrics=[
            lce.Perplexity('perplexity'),
            lce.ROUGE('ROUGE score'),
            lce.bert_score.BERTScore('BERT score', device='cuda'),
            lce.TokenStats('token stats')
        ],
        log_endpoint=log_endpoint, # will fallback to filesystem current directory.
        manual_seed=42, # not meaningful for greedy strategy here, but just keep it.
        run_name=runname,
        skip_until=skip_until,
        slow_mode=slow_mode
    )

    # Report
    import json
    print('=' * 20, 'Report', '=' * 20)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    import fire, os
    os.environ["PAGER"] = 'cat'
    fire.Fire(main)
