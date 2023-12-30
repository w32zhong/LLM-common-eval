source llm_common_eval/batch_eval.sh

export CUDA_VISIBLE_DEVICES=5
detached_experiment exp1 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot.py --shots 3
export CUDA_VISIBLE_DEVICES=6
detached_experiment exp2 \
    python eval_mistral/mistral_on_SNLI_greedy_fewshot.py --shots 1
waitfor_experiments exp1 exp2
