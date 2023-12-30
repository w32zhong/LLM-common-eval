source llm_common_eval/batch_eval.sh

export CUDA_VISIBLE_DEVICES=5
detached_experiment exp1 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot.py --shots 3
export CUDA_VISIBLE_DEVICES=6
detached_experiment exp2 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot.py --shots 1
waitfor_experiments exp1 exp2

export CUDA_VISIBLE_DEVICES=2
detached_experiment exp3 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot_4bit.py --shots 5
export CUDA_VISIBLE_DEVICES=3
detached_experiment exp4 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot_4bit.py --shots 3
export CUDA_VISIBLE_DEVICES=4
detached_experiment exp5 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot_4bit.py --shots 1
waitfor_experiments exp3 exp4 exp5
