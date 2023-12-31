source ./llm_common_eval/batch_eval.sh

# Dataset validation size (ordered):
# cb        56
# copa      100
# wsc       104
# rte       277
# wic       638
# boolq     3.27K
# multirc   4.85K
# record    10K

detached_experiment exp1 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset cb
waitfor_experiments exp1

detached_experiment exp2 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset copa
waitfor_experiments exp2
detached_experiment exp3 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset wsc
waitfor_experiments exp3

detached_experiment exp4 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset rte
detached_experiment exp5 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset wic
waitfor_experiments exp4 exp5

detached_experiment exp6 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset boolq
detached_experiment exp7 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset multirc
waitfor_experiments exp6 exp7

detached_experiment exp8 \
    python eval_mistral/mistral_on_SuperGLUE_all.py --dataset record
waitfor_experiments exp8
